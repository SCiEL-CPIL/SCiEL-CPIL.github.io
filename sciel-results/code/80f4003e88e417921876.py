"""Program to find rank-minimal exact tensor decomposition for 3x3 * 3x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin

    n, m, p = DIMENSIONS
    # A is n x m, B is m x p, C is n x p
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)  <-- Note: W index is k*n + i (output C is n x p, but trace convention says W[k*n+i, r])
    
    # The tensor equation is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i, j, k.
    
    # We can view this as computing C = A * B where A is 3x3 and B is 3x6.
    # We can split B into two 3x3 matrices: B = [B1, B2].
    # Then C = A * B1 + A * B2.
    # This is equivalent to two independent 3x3 * 3x3 multiplications.
    # The rank of the 3x3 * 3x3 tensor is known to be 23 (Landsberg).
    # So the total rank would be 23 + 23 = 46.
    
    # Try to find a rank-46 decomposition by combining two rank-23 decompositions.
    # Since explicit rank-23 coefficients are complex, try a numerical search.
    
    # First, return the schoolbook solution as a fallback.
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    # Try to find a better rank using iterative optimization
    # Start with rank 46 and try to reduce
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Try ranks from 46 down to 36
    for target_rank in range(46, 35, -1):
        if time.time() > deadline - 10:
            break
            
        # Try multiple random initializations
        found = False
        for seed in [42, 123, 456, 789, 1000, 2000, 3000]:
            if time.time() > deadline - 15:
                break
                
            np.random.seed(seed)
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # Iteratively optimize to satisfy the tensor equation
            # The equation is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i, j, k
            
            # Use alternating least squares
            max_iters = 500
            for iteration in range(max_iters):
                if time.time() > deadline - 10:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        # For each (i,j), we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
                        # So: U[i*m+j, :] @ M = 1 where M[k, r] = V[j*p+k, r] * W[k*n+i, r]
                        M = np.outer(V[j*p:j*p+p, :], W[i:i+p, :])  # shape (p, target_rank)
                        try:
                            sol, residuals, rank_m, sv = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                            U[i * m + j, :] = sol
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        # For each (j,k), we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i
                        # So: V[j*p+k, :] @ M = 1 where M[i, r] = U[i*m+j, r] * W[k*n+i, r]
                        M = np.outer(U[:, j], W[k*n:(k+1)*n, :])  # shape (n, target_rank)
                        try:
                            sol, residuals, rank_m, sv = np.linalg.lstsq(M.T, np.ones(n), rcond=None)
                            V[j * p + k, :] = sol
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        # For each (k,i), we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
                        # So: W[k*n+i, :] @ M = 1 where M[j, r] = U[i*m+j, r] * V[j*p+k, r]
                        M = np.zeros((m, target_rank))
                        for jj in range(m):
                            M[jj, :] = U[i*m+jj, :] * V[jj*p+k, :]
                        try:
                            sol, residuals, rank_m, sv = np.linalg.lstsq(M.T, np.ones(m), rcond=None)
                            W[k * n + i, :] = sol
                        except:
                            pass
                
                # Check convergence every 20 iterations
                if iteration % 20 == 0:
                    error = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                                error += (val - 1.0) ** 2
                    if error < 1e-8:
                        found = True
                        break
            
            if found:
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                break
        
        if found:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
