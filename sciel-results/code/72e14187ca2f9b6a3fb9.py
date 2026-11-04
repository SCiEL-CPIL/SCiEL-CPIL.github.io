"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _validate(U, V, W):
    """Validate exact decomposition using trace convention W[k*n+i, r]."""
    n, m, p = DIMENSIONS
    if not U or not V or not W:
        return False
    r = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if not all(len(row) == r for row in U):
        return False
    if not all(len(row) == r for row in V):
        return False
    if not all(len(row) == r for row in W):
        return False
    if r == 0:
        return False
    
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_np[i*m + j] * V_np[j*p + k] * W_np[k*n + i])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _alternating_minimize(U, V, W, max_iters=200, tol=1e-8):
    """Alternating least squares to refine decomposition."""
    n, m, p = DIMENSIONS
    r = U.shape[1]
    
    for it in range(max_iters):
        # Update U
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, r))
                for k in range(p):
                    M[k, :] = V[j*p + k] * W[k*n + i]
                try:
                    U[i*m + j] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, r))
                for i in range(n):
                    M[i, :] = U[i*m + j] * W[k*n + i]
                try:
                    V[j*p + k] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, r))
                for j in range(m):
                    M[j, :] = U[i*m + j] * V[j*p + k]
                try:
                    W[k*n + i] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        # Check convergence
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m + j] * V[j*p + k] * W[k*n + i])
                    error += (val - 1.0) ** 2
        error = np.sqrt(error)
        if error < tol:
            break
    
    return U, V, W, error

def _try_rank(rank, n, m, p, n_restarts=5, max_iters=100):
    """Try to find a valid decomposition of given rank."""
    best_error = np.inf
    best_result = None
    
    for restart in range(n_restarts):
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        U, V, W, error = _alternating_minimize(U, V, W, max_iters=max_iters)
        
        if error < best_error:
            best_error = error
            best_result = (U.copy(), V.copy(), W.copy())
        
        if error < 1e-8:
            # Convert to lists and validate
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            if _validate(U_list, V_list, W_list):
                return U_list, V_list, W_list
    
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # Fallback: schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try progressively higher ranks from a reasonable lower bound
    # Lower bound for matmul rank is at least max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    # But realistically, for 3x5x6, we expect rank around 40-60
    lower_bound = 30
    upper_bound = best_rank
    
    # Try ranks in increasing order
    for rank in range(lower_bound, upper_bound + 1):
        if time.time() > deadline - 10:
            break
        
        result = _try_rank(rank, n, m, p, n_restarts=3, max_iters=50)
        if result is not None:
            U_list, V_list, W_list = result
            best = {"U": U_list, "V": V_list, "W": W_list}
            best_rank = rank
            break
    
    return best
# EVOLVE-BLOCK-END
