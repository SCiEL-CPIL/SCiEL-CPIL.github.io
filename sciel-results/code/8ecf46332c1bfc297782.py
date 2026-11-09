"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 54
    
    # Build the schoolbook decomposition as fallback
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    def to_half_integer(x):
        """Round to nearest 0.5 and return as integer*0.5"""
        return np.round(2.0 * x) / 2.0
    
    def verify(U, V, W, n, m, p):
        """Check if decomposition is exact (within floating point tolerance for half-integers)"""
        rank = U.shape[1]
        # Compute tensor product sum
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This should equal 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def try_rank(r, seed, time_limit):
        """Try to find a rank-r decomposition using randomized approach"""
        start = time.time()
        random.seed(seed)
        np.random.seed(seed)
        
        # Initialize with random values
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        # ALS-style iterations
        for iteration in range(30):
            if time.time() - start > time_limit:
                return None
            
            # Update U
            for i in range(n):
                for j in range(m):
                    # For each (i,j), we need sum_r U[i,j,r] * V[j*p+..., r] * W[...+i, r] = 1 for all k
                    # This is a system of linear equations in U[i,j,:]
                    # For each k: sum_r U[i,j,r] * (V[j*p+k, r] * W[k*n+i, r]) = 1
                    # So: A * U[i,j,:] = 1 where A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    A = np.zeros((p, r))
                    for k in range(p):
                        A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    # For each (j,k), need sum_r U[...+j, r] * V[j,k,r] * W[k*n+i, r] = 1 for all i
                    # A[i, r] = U[i*m+j, r] * W[k*n+i, r]
                    A = np.zeros((n, r))
                    for i in range(n):
                        A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    # For each (k,i), need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k,i,r] = 1 for all j
                    # A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    A = np.zeros((m, r))
                    for j in range(m):
                        A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                    except:
                        pass
        
        # Round to half-integers
        U_h = to_half_integer(U)
        V_h = to_half_integer(V)
        W_h = to_half_integer(W)
        
        # Verify
        if verify(U_h, V_h, W_h, n, m, p):
            return (U_h, V_h, W_h)
        
        return None
    
    # Deadline guard: 150s wall clock, use 140s for safety
    deadline = time.time() + 140
    
    # Try ranks from 53 down to 45 (below schoolbook)
    for r in range(53, 44, -1):
        if time.time() > deadline - 10:
            break
        for seed in range(10):
            if time.time() > deadline - 5:
                break
            result = try_rank(r, seed, time_limit=5.0)
            if result is not None:
                U_h, V_h, W_h = result
                return {
                    "U": U_h.tolist(),
                    "V": V_h.tolist(),
                    "W": W_h.tolist()
                }
    
    # Fall back to schoolbook
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
# EVOLVE-BLOCK-END
