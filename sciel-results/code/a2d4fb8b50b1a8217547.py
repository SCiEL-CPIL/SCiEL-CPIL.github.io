"""Matrix multiplication 2,3,5 tensor decomposition - rank minimization."""
import numpy as np
from itertools import product
import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Schoolbook rank = n*m*p = 30
    rank_schoolbook = n * m * p
    
    # Try to find a lower rank decomposition
    # The tensor T has entries T[(i,j), (j,k), (i,k)] = 1
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[(i,j),(j,k),(i,k)]
    
    # Let's try to find rank r < 30 by solving the nonlinear system
    # For small r, we can use numerical optimization
    
    # First, let's establish the tensor
    # T is a 3-tensor with dimensions (n*m, m*p, p*n) = (6, 15, 10)
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some valid (i,j,k)
    
    def tensor_entry(a, b, c):
        """Return 1 if (a,b,c) corresponds to a valid matmul term, 0 otherwise."""
        i, j = divmod(a, m)
        j2, k = divmod(b, p)
        k2, i2 = divmod(c, n)
        if i == i2 and j == j2 and k == k2:
            return 1
        return 0
    
    # Build the full tensor as a 3D array
    T = np.zeros((n*m, m*p, p*n))
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                T[a, b, c] = tensor_entry(a, b, c)
    
    # Try to find minimal rank by attempting to decompose
    # For rank r, we have r columns in U (6×r), V (15×r), W (10×r)
    # The constraint is: for all (a,b,c), sum over c_idx of U[a,c_idx]*V[b,c_idx]*W[c,c_idx] = T[a,b,c]
    
    # This is a nonlinear system. Let's try a gradient-based approach for small r.
    
    def try_rank(r, max_iter=5000, lr=0.01):
        """Try to find a rank-r decomposition using gradient descent."""
        if time.time() > deadline - 5:
            return None
        
        # Initialize with random values
        np.random.seed(42)
        U = np.random.randn(n*m, r) * 0.1
        V = np.random.randn(m*p, r) * 0.1
        W = np.random.randn(p*n, r) * 0.1
        
        # Scale to help convergence
        U *= (n*m * r) ** 0.25
        V *= (m*p * r) ** 0.25
        W *= (p*n * r) ** 0.25
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                return None
            
            # Compute the current decomposition
            # D[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # This is expensive to compute fully, so let's compute the error
            # and gradients efficiently
            
            # Compute D as a 3D tensor
            D = np.zeros((n*m, m*p, p*n))
            for r_idx in range(r):
                D += np.einsum('ar,br,cr->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
            
            # Error
            E = D - T
            loss = np.sum(E ** 2)
            
            if loss < 1e-10:
                return U, V, W
            
            if it % 100 == 0 and loss > 1e-5:
                # Check if we're making progress
                if it > 100 and loss > 1.0:
                    return None
            
            # Compute gradients
            # dL/dU[a,r] = 2 * sum_{b,c} E[a,b,c] * V[b,r] * W[c,r]
            # dL/dV[b,r] = 2 * sum_{a,c} E[a,b,c] * U[a,r] * W[c,r]
            # dL/dW[c,r] = 2 * sum_{a,b} E[a,b,c] * U[a,r] * V[b,r]
            
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for r_idx in range(r):
                # dU[:, r_idx] = 2 * E @ (V[:, r_idx] * W[:, r_idx]) reshaped
                # E[a,b,c] * V[b,r] * W[c,r] summed over b,c
                # = (E * (V[:, r_idx, None, None] * W[None, :, r_idx])) summed over b,c
                # Let's compute this as: for each a, sum_{b,c} E[a,b,c] * V[b,r] * W[c,r]
                # = (E[a,:,:] * V[:, r_idx][:, None] * W[:, r_idx][None, :]).sum()
                
                # More efficiently:
                # temp1[a,b] = sum_c E[a,b,c] * W[c,r]
                # dU[a,r] = 2 * sum_b temp1[a,b] * V[b,r]
                
                temp1 = np.einsum('abc,c->ab', E, W[:, r_idx])
                dU[:, r_idx] = 2 * np.einsum('ab,b->a', temp1, V[:, r_idx])
                
                temp2 = np.einsum('abc,b->ac', E, V[:, r_idx])
                dW[:, r_idx] = 2 * np.einsum('ac,c->a', temp2, U[:, r_idx])
                
                temp3 = np.einsum('abc,a->bc', E, U[:, r_idx])
                dV[:, r_idx] = 2 * np.einsum('bc,c->b', temp3, W[:, r_idx])
            
            # Gradient descent step
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        # Check final loss
        D = np.zeros((n*m, m*p, p*n))
        for r_idx in range(r):
            D += np.einsum('ar,br,cr->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
        loss = np.sum((D - T) ** 2)
        
        if loss < 1e-8:
            return U, V, W
        return None
    
    # Try ranks from 24 to 29
    best_result = None
    for r in range(24, rank_schoolbook):
        if time.time() > deadline - 15:
            break
        result = try_rank(r)
        if result is not None:
            best_result = (r, result)
            break
    
    if best_result is not None:
        r, (U, V, W) = best_result
        # Convert to lists with rounded values (integer or half-integer)
        U_list = [[round(U[i, j] * 2) / 2 for j in range(r)] for i in range(n*m)]
        V_list = [[round(V[i, j] * 2) / 2 for j in range(r)] for i in range(m*p)]
        W_list = [[round(W[i, j] * 2) / 2 for j in range(r)] for i in range(p*n)]
        
        # Verify the rounded solution
        U_np = np.array(U_list)
        V_np = np.array(V_list)
        W_np = np.array(W_list)
        D = np.zeros((n*m, m*p, p*n))
        for r_idx in range(r):
            D += np.einsum('ar,br,cr->abc', U_np[:, r_idx], V_np[:, r_idx], W_np[:, r_idx])
        if np.max(np.abs(D - T)) < 1e-6:
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook rank 30
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
