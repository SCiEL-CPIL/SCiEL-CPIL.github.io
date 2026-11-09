"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import random
import time

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Target rank based on known results for 2x3x4 matrix multiplication
    # The rank is known to be 11 (Bini's algorithm or similar)
    target_rank = 11
    
    # Try to find a rank-11 decomposition
    best_result = None
    
    # Schoolbook decomposition as fallback (rank 24)
    def schoolbook():
        rank = n * m * p
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
    
    # Verify a decomposition
    def verify_decomposition(U, V, W, tol=1e-10):
        n, m, p = DIMENSIONS
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        
        # Check dimensions
        if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
            return False
        
        # Verify: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1 if (i,j,k) is valid
        # and 0 otherwise. Actually, the tensor entry T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # For matrix multiplication, T[i,j,k] = 1 if this is a valid product term, else 0
        # Actually for 2x3 * 3x4, the output is 2x4, so the tensor has entries for (i,j,k) where
        # i in [0,2), j in [0,3), k in [0,4), and the product C[i,k] = sum_j A[i,j]*B[j,k]
        # So the tensor T[i,j,k] = 1 for all valid (i,j,k)
        
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    # For matrix multiplication, all entries should be 1
                    err = abs(val - 1.0)
                    max_err = max(max_err, err)
        
        return max_err < tol, max_err
    
    # Random search for low-rank decomposition
    def random_search(rank, n_iter=10000):
        for _ in range(n_iter):
            if time.time() > deadline:
                break
            
            # Generate random factors
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Scale to help convergence
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            valid, err = verify_decomposition(U, V, W)
            if valid:
                return U.tolist(), V.tolist(), W.tolist()
        
        return None
    
    # Try structured approaches based on known algorithms
    # For 2x3x4, try rank 11, 12, 13, 14
    
    for rank in [11, 12, 13, 14, 15, 16]:
        if time.time() > deadline:
            break
        
        # Multiple random restarts
        found = random_search(rank, n_iter=5000)
        if found:
            U, V, W = found
            best_result = {"U": U, "V": V, "W": W}
            break
    
    if best_result is None:
        # Fallback to schoolbook
        best_result = schoolbook()
    
    return best_result
# EVOLVE-BLOCK-END
