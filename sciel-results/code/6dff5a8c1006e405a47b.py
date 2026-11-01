"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 40
    
    # Build the tensor: T[i*m+j, j*p+k, k*n+i] = 1
    # Shape: (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    best_rank = schoolbook_rank
    best_U = None
    best_V = None
    best_W = None
    
    # Try to find a lower rank decomposition
    # Strategy: use alternating least squares or similar to minimize rank
    # We'll try ranks from some lower bound up to schoolbook_rank
    
    # Theoretical lower bound: at least max(n*m, m*p, p*n) / something
    # Actually, rank >= max(n*m, m*p, n*p) is not necessarily true
    # A better lower bound: rank >= n*m*p / min(n*m, m*p, n*p) ... not quite
    
    # Let's try ranks from 20 to 39
    for target_rank in range(20, schoolbook_rank):
        if time.time() > DEADLINE:
            break
        
        # Initialize with random values
        np.random.seed(target_rank * 1000 + 42)
        U = np.random.randn(n*m, target_rank)
        V = np.random.randn(m*p, target_rank)
        W = np.random.randn(p*n, target_rank)
        
        # Alternating least squares
        converged = False
        for it in range(500):
            if time.time() > DEADLINE:
                break
            
            # Update U: for each column r, U[:,r] should satisfy
            # sum_r U[i,r] * V[j,r] * W[k,r] = T[i,j,k]
            # For fixed r, U[:,r] = argmin sum_{j,k} (T[i,j,k] - sum_r U[i,r]*V[j,r]*W[k,r])^2
            # This is a linear system
            
            # Actually, let's use a simpler approach: gradient descent or ALS
            # For ALS, fix V and W, solve for U
            # T[i,j,k] = sum_r U[i,r] * (V[j,r]*W[k,r])
            # For each r: U[:,r] = (V[:,r] outer W[:,r])^+ * T_reshaped... 
            # This is getting complex. Let me use a different approach.
            
            # Use the fact that we can vectorize
            # T as (n*m, m*p*p*n) matrix where T_mat[i, j*(p*n)+k] = T[i,j,k]
            # Then T_mat = U * (V kron W)^T ... not quite
            
            # Let me use a direct approach: minimize ||T - sum_r U_r outer V_r outer W_r||_F
            # using gradient descent
            
            # Compute current approximation
            approx = np.zeros_like(T)
            for r in range(target_rank):
                approx += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
            
            error = np.linalg.norm(T - approx, 'fro')
            if error < 1e-6:
                converged = True
                break
            
            # Gradient descent step
            lr = 0.01
            for r in range(target_rank):
                # d/dU[i,r] = sum_{j,k} (approx[i,j,k] - T[i,j,k]) * V[j,r] * W[k,r]
                grad_U = np.einsum('ijk,j,k->i', T - approx, V[:,r], W[:,r])
                grad_V = np.einsum('ijk,i,k->j', T - approx, U[:,r], W[:,r])
                grad_W = np.einsum('ijk,i,j->k', T - approx, U[:,r], V[:,r])
                U[:,r] -= lr * grad_U
                V[:,r] -= lr * grad_V
                W[:,r] -= lr * grad_W
        
        # Check if we found a good solution
        approx = np.zeros_like(T)
        for r in range(target_rank):
            approx += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
        error = np.linalg.norm(T - approx, 'fro')
        
        if error < 1e-4:
            # Try to round to exact values
            # Round to nearest half-integer
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            approx_r = np.zeros_like(T)
            for r in range(target_rank):
                approx_r += np.einsum('i,j,k->ijk', U_rounded[:,r], V_rounded[:,r], W_rounded[:,r])
            error_r = np.linalg.norm(T - approx_r, 'fro')
            
            if error_r < 1e-8:
                best_rank = target_rank
                best_U = U_rounded
                best_V = V_rounded
                best_W = W_rounded
                break
            else:
                # Use the unrounded solution but try to make it exact
                # Check if it's close enough to be considered valid
                if error < 1e-4:
                    best_rank = target_rank
                    best_U = U
                    best_V = V
                    best_W = W
    
    # If we found a better rank, use it; otherwise use schoolbook
    if best_rank < schoolbook_rank and best_U is not None:
        # Convert to lists of integers or half-integers
        U_list = [[int(x * 2) / 2 for x in row] for row in best_U]
        V_list = [[int(x * 2) / 2 for x in row] for row in best_V]
        W_list = [[int(x * 2) / 2 for x in row] for row in best_W]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Schoolbook decomposition
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
