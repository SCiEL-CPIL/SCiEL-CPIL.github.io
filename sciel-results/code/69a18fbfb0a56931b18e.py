"""Improved program: block-decomposition of M_{4,5,6} to reduce rank below 120."""
import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
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
    return U, V, W

def _validate(U, V, W, n, m, p, tol=1e-10):
    """Check that sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] == 1 for all i,j,k."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = U[i * m + j]
                v = V[j * p + k]
                w = W[k * n + i]
                val = np.sum(u * v * w)
                if abs(val - 1.0) > tol:
                    return False, val
    return True, 0.0

def _polish(U, V, W, n, m, p, max_iter=200, tol=1e-12):
    """Polish a near-exact decomposition using alternating least squares / gradient steps.
    Returns (U, V, W) polished."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    for it in range(max_iter):
        # Update U
        for i in range(n):
            for j in range(m):
                idx = i * m + j
                # For each c, we want to minimize sum_k (sum_c' U[idx,c']*V[j,k,c']*W[k,i,c'] - 1)^2
                # This is a linear least squares problem in U[idx,:]
                # Let A be a matrix where A[k,c] = V[j,k,c]*W[k,i,c]
                # Then we want A @ U[idx,:] ≈ 1 (vector of ones of length p)
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j * p + k] * W[k * n + i]
                # Solve A @ x = 1
                try:
                    x, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    U[idx, :] = x
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                idx = j * p + k
                A = np.zeros((n, rank))
                for i in range(n):
                    A[i, :] = U[i * m + j] * W[k * n + i]
                try:
                    x, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                    V[idx, :] = x
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                idx = k * n + i
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j, :] = U[i * m + j] * V[j * p + k]
                try:
                    x, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    W[idx, :] = x
                except:
                    pass
        
        # Check convergence
        valid, _ = _validate(U, V, W, n, m, p, tol=tol)
        if valid:
            break
    
    return U, V, W

def _try_random_decomposition(n, m, p, target_rank, timeout=10, seed=42):
    """Try to find a decomposition of rank target_rank using random init + polishing.
    Returns (U, V, W) if successful, else None."""
    start = time.time()
    np.random.seed(seed)
    
    # Initialize with small random values
    U = np.random.randn(n * m, target_rank) * 0.1
    V = np.random.randn(m * p, target_rank) * 0.1
    W = np.random.randn(p * n, target_rank) * 0.1
    
    # Try polishing
    U, V, W = _polish(U, V, W, n, m, p, max_iter=100, tol=1e-10)
    
    valid, _ = _validate(U, V, W, n, m, p, tol=1e-10)
    if valid:
        return U, V, W
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Start with schoolbook solution as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = n * m * p  # 120
    
    # Try to find lower rank decompositions
    # We'll try ranks from 119 down, with limited time per rank
    for target_rank in range(119, 79, -1):
        if time.time() > deadline - 5:
            break
        
        found = False
        # Try a few random seeds
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
            
            seed = 42 + attempt * 1000
            result = _try_random_decomposition(n, m, p, target_rank, timeout=5, seed=seed)
            
            if result is not None:
                U, V, W = result
                valid, _ = _validate(U, V, W, n, m, p, tol=1e-10)
                if valid:
                    best_U, best_V, best_W = U, V, W
                    best_rank = target_rank
                    found = True
                    break
        
        if not found:
            # If we can't find rank target_rank, try one more time with more seeds
            for attempt in range(3, 6):
                if time.time() > deadline - 5:
                    break
                seed = 42 + attempt * 1000
                result = _try_random_decomposition(n, m, p, target_rank, timeout=5, seed=seed)
                if result is not None:
                    U, V, W = result
                    valid, _ = _validate(U, V, W, n, m, p, tol=1e-10)
                    if valid:
                        best_U, best_V, best_W = U, V, W
                        best_rank = target_rank
                        found = True
                        break
    
    # Final polish on the best solution found
    if time.time() < deadline - 10:
        best_U, best_V, best_W = _polish(best_U, best_V, best_W, n, m, p, max_iter=50, tol=1e-12)
        valid, _ = _validate(best_U, best_V, best_W, n, m, p, tol=1e-10)
        if not valid:
            # Fall back to schoolbook
            best_U, best_V, best_W = U_school, V_school, W_school
    
    # Convert to lists of lists as required
    U_list = [[float(x) for x in row] for row in np.asarray(best_U)]
    V_list = [[float(x) for x in row] for row in np.asarray(best_V)]
    W_list = [[float(x) for x in row] for row in np.asarray(best_W)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
