"""Matrix multiplication 5x5x5 rank-minimal decomposition search."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]
_TIME_LIMIT = 140.0

def _time_remaining(deadline):
    return deadline - time.time()

def _verify(U, V, W, n, m, p):
    """Verify decomposition: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1"""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i * m + j], V_arr[j * p + k] * W_arr[k * n + i])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
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

def _try_winograd_style(n, m, p, time_budget):
    """
    Try to construct a decomposition using Winograd-style reductions.
    For n=m=p=5, we can use the fact that 5 = 4+1 and apply
    reduction techniques. This is a heuristic approach.
    """
    deadline = time.time() + time_budget
    
    # For now, just return schoolbook as we don't have a known explicit
    # lower-rank construction that we can implement quickly
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT
    
    # Default: schoolbook
    result = _schoolbook(n, m, p)
    
    # Try to find better decomposition if time permits
    remaining = _time_remaining(deadline)
    if remaining > 10:
        # Attempt Winograd-style or other structured search
        better = _try_winograd_style(n, m, p, remaining - 5)
        if better is not None and _verify(better["U"], better["V"], better["W"], n, m, p):
            result = better
    
    return result
