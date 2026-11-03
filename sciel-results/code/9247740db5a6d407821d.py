import numpy as np
import time

DIMENSIONS = [5, 5, 5]

def _validate_decomposition(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    # The matrix multiplication tensor T[i,j,k] = 1 for all i,j,k
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U[u_idx], V[v_idx] * W[w_idx])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 5x5 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    Score is the rank (lower is better).
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s. We return by 140s.
    deadline = start_time + 140.0
    
    # 1. Initialize with schoolbook decomposition (Rank 125)
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook), dtype=float)
    V_school = np.zeros((m * p, rank_schoolbook), dtype=float)
    W_school = np.zeros((p * n, rank_schoolbook), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    best_rank = rank_schoolbook
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # 2. Attempt to find a lower rank (124) using refined alternating minimization
    # We use a strict time budget for this search to ensure we always return a valid result.
    target_rank = 124
    search_budget = 30.0 # Seconds
    
    if time.time() < deadline - search_budget:
        try:
            U, V, W = _try_fast_alternating_minimization(n, m, p, target_rank, start_time, deadline - search_budget)
            if U is not None:
                # Round to nearest half-integer
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                # Verify the rounded solution
                if _validate_decomposition(U_r, V_r, W_r, n, m, p):
                    best_rank = target_rank
                    best_U = U_r
                    best_V = V_r
                    best_W = W_r
        except Exception:
            pass # Fall back to schoolbook

    # 3. Return best result
    def to_list(arr):
        return [[float(arr[i, j]) for j in range(arr.shape[1])] for i in range(arr.shape[0])]
    
    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }

def _try_fast_alternating_minimization(n, m, p, rank, start_time, deadline):
    """
    Attempt to find a rank decomposition using vectorized alternating minimization.
    Returns (U, V, W) or (None, None, None) if failed.
    """
    # Try a few different random seeds to increase chance of finding a solution
    seeds = [42, 123, 999, 2023]
    
    for seed in seeds:
        if time.time() > deadline - 5.0:
            return None, None, None
            
        np.random.seed(seed)
        
        # Initialize with random values from {-1, -0.5, 0, 0.5, 1}
        # This helps with the half-integer constraint later
        U = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(n*m, rank))
        V = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(m*p, rank))
        W = np.random.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=(p*n, rank))
        
        max_iters = 100
        prev_err = float('inf')
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                return None, None, None
            
            # Update U
            for i in range(n):
                for j in range(m):
                    # Construct matrix B where B[c, k] = V[j*p+k, c] * W[k*n+i, c]
                    B = np.zeros((rank, p))
                    for k in range(p):
                        B[:, k] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(B.T, np.ones(p), rcond=None)[0]
                    except:
                        return None, None, None
            
            # Update V
            for j in range(m):
                for k in range(p):
                    # Construct matrix A where A[c, i] = U[i*m+j, c] * W[k*n+i, c]
                    A = np.zeros((rank, n))
                    for i in range(n):
                        A[:, i] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(A.T, np.ones(n), rcond=None)[0]
                    except:
                        return None, None, None
            
            # Update W
            for k in range(p):
                for i in range(n):
                    # Construct matrix C where C[c, j] = U[i*m+j, c] * V[j*p+k, c]
                    C = np.zeros((rank, m))
                    for j in range(m):
                        C[:, j] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(C.T, np.ones(m), rcond=None)[0]
                    except:
                        return None, None, None
            
            # Check convergence
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        err = abs(val - 1.0)
                        if err > max_err:
                            max_err = err
                            
            if max_err < 1e-8:
                return U, V, W
                
            # Early stopping if diverging
            if max_err > prev_err * 1.1 and it > 10:
                break
            prev_err = max_err
            
    return None, None, None
