import numpy as np
import time

DIMENSIONS = [2, 2, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank) = (4, rank)
    # V: (m*p, rank) = (14, rank)
    # W: (p*n, rank) = (14, rank)
    
    # Build target tensor T of shape (4, 14, 14)
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook decomposition (rank 28)
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    best_U, best_V, best_W = schoolbook()
    best_rank = n * m * p
    
    def check_decomposition(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose T."""
        rank = U.shape[1]
        # Compute sum_r U_r ⊗ V_r ⊗ W_r
        # T_reconstructed[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_rec, T, atol=tol)
    
    def try_rank(r, max_iters=2000, n_restarts=5):
        """Try to find a rank-r decomposition."""
        nonlocal best_U, best_V, best_W, best_rank
        if r >= best_rank:
            return False
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                return False
            
            # Random initialization
            U = np.random.randn(n*m, r) * 0.1
            V = np.random.randn(m*p, r) * 0.1
            W = np.random.randn(p*n, r) * 0.1
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return False
                
                # Update U: for each a, U[a,r] = T[a,:,:] contracted with V and W
                # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
                # Fix V, W: U[a,r] = sum_{b,c} T[a,b,c] V[b,r] W[c,r] / sum_{b,c} V[b,r]^2 W[c,r]^2
                # This is a least squares problem.
                
                # For each a: T[a,:,:] = sum_r U[a,r] (V[:,r] ⊗ W[:,r])
                # Let M_a be a matrix of shape (m*p * p*n, r) where M_a[:,r] = V[:,r] ⊗ W[:,r]
                # Then T[a].reshape(-1) = M_a @ U[a,:]
                
                # Compute M_a
                # M_a[:, r] = V[:, r] ⊗ W[:, r] (Kronecker product)
                # Shape: (m*p * p*n, r) = (196, r)
                
                # This is expensive. Let's use a different approach.
                
                # Alternative: use gradient descent
                T_rec = np.einsum('ar,br,cr->abc', U, V, W)
                err = T_rec - T
                loss = np.sum(err**2)
                
                if loss < 1e-18:
                    if check_decomposition(U, V, W):
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        best_rank = r
                        return True
                    break
                
                # Gradients
                # dL/dU[a,r] = 2 * err[a,:,:] contracted with V[:,r] and W[:,r]
                # = 2 * sum_{b,c} err[a,b,c] V[b,r] W[c,r]
                dU = 2 * np.einsum('abc,br,cr->ar', err, V, W)
                dV = 2 * np.einsum('abc,ar,cr->br', err, U, W)
                dW = 2 * np.einsum('abc,ar,br->cr', err, U, V)
                
                # Adaptive learning rate
                lr = 0.01 / (1 + it * 0.01)
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
            
            # Final check
            T_rec = np.einsum('ar,br,cr->abc', U, V, W)
            if np.allclose(T_rec, T, atol=1e-9):
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                best_rank = r
                return True
        
        return False
    
    # Try ranks from low to high
    for r in range(14, best_rank):
        if time.time() > deadline:
            break
        if try_rank(r):
            break
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
