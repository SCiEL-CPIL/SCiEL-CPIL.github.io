"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Target: find rank-minimal decomposition
    # Start with schoolbook rank = n*m*p = 80
    schoolbook_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Known: 4x4x4 has rank 49. For 4x4x4x5, we can try ranks between 49 and 80.
    
    # Strategy: Use iterative tensor decomposition
    # The tensor T[i,j,k] = 1 for all valid (i,j,k)
    # We need: T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    
    best_result = None
    best_rank = schoolbook_rank
    
    # First, create schoolbook as fallback
    def make_schoolbook():
        rank = schoolbook_rank
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    U_s, V_s, W_s, r_s = make_schoolbook()
    best_result = {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    best_rank = r_s
    
    # Try ranks from low to high
    # For 4x4x4x5, try ranks: 56, 60, 64, 68, 72, 76
    # Actually, let's try a range and see what works
    
    for target_rank in [56, 60, 64, 68, 72, 76, 80]:
        if time.time() > deadline - 30:
            break
        if target_rank >= best_rank:
            continue
            
        success = try_decompose(n, m, p, target_rank, deadline)
        if success:
            U, V, W, r = success
            if r < best_rank:
                best_rank = r
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return best_result

def try_decompose(n, m, p, target_rank, deadline):
    """Try to find a rank-target_rank decomposition using iterative method."""
    
    # Initialize with random or structured initialization
    np.random.seed(42)
    
    max_iter = 500
    tol = 1e-10
    
    # Initialize U, V, W
    U = np.random.randn(n*m, target_rank) * 0.1
    V = np.random.randn(m*p, target_rank) * 0.1
    W = np.random.randn(p*n, target_rank) * 0.1
    
    # Use ALS-like approach
    for iteration in range(max_iter):
        if time.time() > deadline - 10:
            return None
        
        # Update U: for each (i,j), solve least squares
        for i in range(n):
            for j in range(m):
                # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                # For fixed (i,j): sum_k T[i,j,k] * V[j*p+k,r] * W[k*n+i,r] 
                # We want U[i*m+j,:] such that:
                # sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1 for all k
                # This is: U[i*m+j,:] * M = ones(p) where M[r,k] = V[j*p+k,r] * W[k*n+i,r]
                M = np.zeros((target_rank, p))
                for k in range(p):
                    M[:, k] = V[j*p+k, :] * W[k*n+i, :]
                # Solve: U[i*m+j,:] * M = ones(p)
                # i.e., M.T @ U[i*m+j,:] = ones(p)
                try:
                    U[i*m+j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        # Update V: for each (j,k), solve least squares
        for j in range(m):
            for k in range(p):
                # For fixed (j,k): sum_i T[i,j,k] * U[i*m+j,r] * W[k*n+i,r] = 1 for all i
                # M[r,i] = U[i*m+j,r] * W[k*n+i,r]
                M = np.zeros((target_rank, n))
                for i in range(n):
                    M[:, i] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    V[j*p+k, :] = np.linalg.lstsq(M.T, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        # Update W: for each (k,i), solve least squares
        for k in range(p):
            for i in range(n):
                # For fixed (k,i): sum_j T[i,j,k] * U[i*m+j,r] * V[j*p+k,r] = 1 for all j
                # M[r,j] = U[i*m+j,r] * V[j*p+k,r]
                M = np.zeros((target_rank, m))
                for j in range(m):
                    M[:, j] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    W[k*n+i, :] = np.linalg.lstsq(M.T, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        # Check convergence
        if iteration % 10 == 0:
            error = compute_error(n, m, p, U, V, W)
            if error < tol:
                return U, V, W, target_rank
    
    # Final check
    error = compute_error(n, m, p, U, V, W)
    if error < 1e-6:
        return U, V, W, target_rank
    
    return None

def compute_error(n, m, p, U, V, W):
    """Compute the maximum error of the decomposition."""
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err
# EVOLVE-BLOCK-END

