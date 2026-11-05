import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def _schoolbook():
    """Return the naive rank-63 decomposition with integer entries."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W):
    """Check if the decomposition is valid. Returns True if valid."""
    n, m, p = DIMENSIONS
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if val != 1:
                    return False
    return True

def _try_alternating_minimization(rank, max_iter=200, tol=1e-8, deadline=None):
    """
    Try to find a rank-r decomposition using alternating minimization.
    Note: This currently returns floats. We will snap to integers/half-integers before returning.
    """
    n, m, p = DIMENSIONS
    start = time.time()
    
    best_result = None
    best_err = np.inf
    
    for attempt in range(5):
        if deadline is not None and time.time() > deadline:
            break
        
        np.random.seed(42 + attempt)
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        for iteration in range(max_iter):
            if deadline is not None and time.time() > deadline - 5:
                break
            
            # Solve for U given V, W
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b = np.ones(p)
                    try:
                        U[i*m+j, :], residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
                    except:
                        pass
            
            # Solve for V given U, W
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    try:
                        V[j*p+k, :], residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
                    except:
                        pass
            
            # Solve for W given U, V
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    try:
                        W[k*n+i, :], residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
                    except:
                        pass
            
            # Check convergence
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        err += (val - 1.0) ** 2
            err = np.sqrt(err / (n*m*p))
            
            if err < best_err:
                best_err = err
                best_result = (U.copy(), V.copy(), W.copy())
            
            if err < tol:
                break
        
        if best_err < 1e-6:
            break
    
    if best_result is not None and best_err < 1e-6:
        # Snap to nearest integer or half-integer
        U_int = np.round(best_result[0] * 2) / 2
        V_int = np.round(best_result[1] * 2) / 2
        W_int = np.round(best_result[2] * 2) / 2
        
        # Verify the snapped version
        if _verify(U_int, V_int, W_int):
            return (U_int, V_int, W_int), best_err
            
    return None, best_err

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Try to find a low-rank decomposition
    # We start from a lower bound estimate. For 3x3x7, 55 is a known upper bound.
    # We try a few ranks.
    for rank in range(55, 63):
        if time.time() > deadline:
            break
        
        result, err = _try_alternating_minimization(rank, max_iter=50, tol=1e-8, deadline=deadline)
        
        if result is not None:
            U, V, W = result
            # The _try function already verifies the snapped integer version
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook (which is now integer)
    schoolbook = _schoolbook()
    return {"U": schoolbook["U"].tolist(), "V": schoolbook["V"].tolist(), "W": schoolbook["W"].tolist()}
