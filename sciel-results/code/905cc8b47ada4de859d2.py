import numpy as np
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # safety margin
    
    # Target: find rank r < n*m*p = 100
    # Try various ranks and use ALS to find exact decomposition
    
    best_result = None
    best_rank = n * m * p  # schoolbook
    
    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, rank
    
    best_result, best_rank = schoolbook()
    
    # Try to find lower rank via ALS
    # The constraint: for all i,j,k: sum_t W[k*n+i,t] * U[i*m+j,t] * V[j*p+k,t] = 1
    
    def check_residual(U, V, W, r):
        """Check if the decomposition is exact."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(W[k * n + i, :] * U[i * m + j, :] * V[j * p + k, :])
                    max_err = max(max_err, abs(val - 1.0))
        return max_err
    
    def als_step(r, U, V, W, max_iters=200, tol=1e-10):
        """ALS optimization for the 3-way tensor constraint."""
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Update W: for each (k,i), solve linear system
            # sum_t W[k*n+i,t] * U[i*m+j,t] * V[j*p+k,t] = 1 for all j
            # This is: W[k*n+i,:] @ diag(U[i*m+j,:]*V[j*p+k,:]) ... 
            # Actually for fixed (k,i), we need: for each j, sum_t W[k*n+i,t] * (U[i*m+j,t] * V[j*p+k,t]) = 1
            # Let A_j[t] = U[i*m+j,t] * V[j*p+k,t], then sum_t W[k*n+i,t] * A_j[t] = 1 for all j
            # This is a linear system: A @ W[k*n+i,:]^T = 1 where A[j,t] = U[i*m+j,t]*V[j*p+k,t]
            
            for i in range(n):
                for k in range(p):
                    # Build system: for each j, sum_t W[k*n+i,t] * U[i*m+j,t] * V[j*p+k,t] = 1
                    A = np.outer(U[i * m, :], V[k * p, :])  # shape (m, r)
                    # A @ w = 1 where w = W[k*n+i,:]
                    try:
                        w, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        W[k * n + i, :] = w
                    except:
                        pass
            
            # Update U: for each (i,j), solve
            # sum_t U[i*m+j,t] * W[k*n+i,t] * V[j*p+k,t] = 1 for all k
            for i in range(n):
                for j in range(m):
                    A = np.outer(W[:, i], V[j * p, :])  # shape (p, r)
                    try:
                        u, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        U[i * m + j, :] = u
                    except:
                        pass
            
            # Update V: for each (j,k), solve
            # sum_t V[j*p+k,t] * U[i*m+j,t] * W[k*n+i,t] = 1 for all i
            for j in range(m):
                for k in range(p):
                    A = np.outer(U[:, j], W[k * n, :])  # shape (n, r)
                    try:
                        v, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                        V[j * p + k, :] = v
                    except:
                        pass
            
            # Check convergence
            err = check_residual(U, V, W, r)
            if err < tol:
                return U, V, W, err
        
        return U, V, W, check_residual(U, V, W, r)
    
    # Try ranks from 99 down to some lower bound
    # Lower bound: at least max(n*p, m*p, n*m) = max(20, 25, 20) = 25
    # But actually the rank must be at least the "slice rank" lower bound
    # For now, try a range
    
    for r in range(best_rank - 1, 24, -1):
        if time.time() > deadline - 10:
            break
        
        success = False
        # Try multiple random initializations
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(attempt * 1000 + r)
            U = np.random.randn(n * m, r) * 0.1
            V = np.random.randn(m * p, r) * 0.1
            W = np.random.randn(p * n, r) * 0.1
            
            U, V, W, err = als_step(r, U, V, W, max_iters=100)
            
            if err < 1e-6:
                # Found a valid decomposition!
                best_rank = r
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                success = True
                break
        
        if success:
            # Try to go even lower
            continue
        else:
            # If we can't find one at this rank, try lower (but might be harder)
            # Actually, if rank r doesn't work, lower ranks might also not work
            # But due to numerical issues, let's keep trying a few more
            if r < best_rank - 5:
                break
    
    if time.time() > deadline - 5:
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
