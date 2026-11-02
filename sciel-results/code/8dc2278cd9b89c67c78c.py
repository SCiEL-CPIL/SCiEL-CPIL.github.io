import numpy as np
import time

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # Target rank to search for
    target_rank = 23
    # Fallback rank
    fallback_rank = n * m * p
    
    # Construct the target tensor T
    # T[i, j, k, l] is not the right shape.
    # The tensor is T_{(i,j), (j,k), (k,i)}
    # Shape: (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def try_find_decomposition(rank, time_limit=10):
        start_time = time.time()
        np.random.seed(42)
        # Initialize U, V, W
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Scale to help convergence
        U /= np.sqrt(rank)
        V /= np.sqrt(rank)
        W /= np.sqrt(rank)
        
        lr = 0.01
        # Simple gradient descent
        for step in range(1000):
            if time.time() - start_time > time_limit:
                break
            
            # Compute current decomposition
            # C = sum_r U[:, r] V[:, r]^T W[:, r]
            # This is a tensor of shape (n*m, m*p, p*n)
            # We can compute the error efficiently
            # E_{a,b,c} = T_{a,b,c} - sum_r U_{a,r} V_{b,r} W_{c,r}
            
            # Gradient with respect to U_{a,r}:
            # dE/du_{a,r} = - sum_{b,c} E_{a,b,c} V_{b,r} W_{c,r}
            # We can compute this using matrix multiplications
            # Let's compute the "contraction" of E with V and W
            # E is (n*m, m*p, p*n)
            # V is (m*p, rank)
            # W is (p*n, rank)
            
            # Reshape E to (n*m, m*p, p*n)
            # We want G_U (n*m, rank)
            # G_U[a, r] = sum_{b, c} E[a, b, c] V[b, r] W[c, r]
            # This is equivalent to:
            # Temp1 = np.einsum('abc,br->acr', E, V) # (n*m, p*n, rank)
            # G_U = np.einsum('acr,cr->ar', Temp1, W) # (n*m, rank)
            
            # Similarly for V and W
            # G_V[b, r] = sum_{a, c} E[a, b, c] U[a, r] W[c, r]
            # Temp2 = np.einsum('abc,ar->bcr', E, U) # (m*p, p*n, rank)
            # G_V = np.einsum('bcr,cr->br', Temp2, W)
            
            # G_W[c, r] = sum_{a, b} E[a, b, c] U[a, r] V[b, r]
            # Temp3 = np.einsum('abc,ar->bcr', E, U) # Reuse Temp2? No, indices differ
            # Temp3 = np.einsum('abc,br->acr', E, V) # Reuse Temp1
            # G_W = np.einsum('acr,ar->cr', Temp1, U)
            
            # Compute E
            # E = T - sum_r U_r V_r^T W_r
            # This is expensive to compute fully every step.
            # But n*m=9, m*p=9, p*n=9. 9x9x9 = 729 entries.
            # Rank 23.
            # 23 iterations of outer product.
            # It's small enough.
            
            E = T.copy()
            for r in range(rank):
                E -= np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            
            # Compute gradients
            Temp1 = np.einsum('abc,br->acr', E, V)
            G_U = np.einsum('acr,cr->ar', Temp1, W)
            
            Temp2 = np.einsum('abc,ar->bcr', E, U)
            G_V = np.einsum('bcr,cr->br', Temp2, W)
            
            G_W = np.einsum('acr,ar->cr', Temp1, U)
            
            # Update
            U -= lr * G_U
            V -= lr * G_V
            W -= lr * G_W
            
            # Check convergence
            err = np.linalg.norm(E)
            if err < 1e-9:
                break
        
        # Round to half-integers
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        # Check exactness
        E_rounded = T.copy()
        for r in range(rank):
            E_rounded -= np.einsum('a,b,c->abc', U_rounded[:, r], V_rounded[:, r], W_rounded[:, r])
        
        if np.linalg.norm(E_rounded) < 1e-9:
            return U_rounded, V_rounded, W_rounded
        else:
            return None

    # Try to find a lower rank decomposition
    result = try_find_decomposition(target_rank, time_limit=10)
    
    if result:
        U, V, W = result
    else:
        # Fallback to schoolbook
        rank = fallback_rank
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
    
    # Convert to lists as required by interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
