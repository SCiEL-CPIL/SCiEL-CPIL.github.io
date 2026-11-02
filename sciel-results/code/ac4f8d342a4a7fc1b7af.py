"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # 110s deadline, leaving 10s margin
    
    # Start with schoolbook as fallback
    rank_schoolbook = n * m * p
    U_fb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    
    # Try to find lower rank decompositions
    # Use random search with local refinement
    np.random.seed(42)
    random.seed(42)
    
    for target_rank in range(55, 27, -1):
        if time.time() > deadline - 30:
            break
        
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Scale to make entries reasonable
            U /= np.sqrt(target_rank)
            V /= np.sqrt(target_rank)
            W /= np.sqrt(target_rank)
            
            # Check if this decomposes the tensor
            # The tensor T[i,j,k] = 1 if the multiplication is correct
            # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta(i,j,k) for the multiplication tensor
            # Actually, the multiplication tensor has T[i,j,k] = 1 for all valid (i,j,k)
            # Wait, no. The matrix multiplication tensor is defined as:
            # T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
            # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
            # And the decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
            
            # Check by reconstructing
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        err = abs(val - 1.0)
                        if err > max_err:
                            max_err = err
            
            if max_err < 1e-6:
                # Found a valid decomposition
                best_rank = target_rank
                best_U = [[float(x) for x in U[r]] for r in range(n*m)]
                best_V = [[float(x) for x in V[r]] for r in range(m*p)]
                best_W = [[float(x) for x in W[r]] for r in range(p*n)]
                break
        
        if best_rank == target_rank:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
