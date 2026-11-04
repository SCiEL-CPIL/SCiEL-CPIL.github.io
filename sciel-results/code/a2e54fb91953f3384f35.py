import numpy as np
import time

DIMENSIONS = [3, 3, 3]

def _verify_decomposition(U, V, W, n=3, m=3, p=3):
    """
    Verify that the decomposition U, V, W correctly decomposes the 3x3 matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Condition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i, j, k.
    """
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    rank = U_arr.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_arr[i * m + j] * V_arr[j * p + k] * W_arr[k * n + i])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _round_to_half_ints(U, V, W):
    """
    Round the entries of U, V, W to the nearest half-integer.
    """
    def round_matrix(M):
        return [[round(x * 2) / 2 for x in row] for row in M]
    
    return round_matrix(U), round_matrix(V), round_matrix(W)

def _als_decomposition(r, n=3, m=3, p=3, max_iter=100, time_budget=10.0):
    """
    Try to find a rank-r decomposition using ALS (Alternating Least Squares).
    """
    start = time.time()
    
    # Initialize randomly
    U = np.random.randn(n * m, r) * 0.5
    V = np.random.randn(m * p, r) * 0.5
    W = np.random.randn(p * n, r) * 0.5
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for iteration in range(max_iter):
        if time.time() - start > time_budget:
            break
        
        # Update U
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, r))
                for k in range(p):
                    M[k, :] = V[j * p + k] * W[k * n + i]
                try:
                    U_new, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    U[i * m + j, :] = U_new
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, r))
                for i in range(n):
                    M[i, :] = U[i * m + j] * W[k * n + i]
                try:
                    V_new, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(n), rcond=None)
                    V[j * p + k, :] = V_new
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, r))
                for j in range(m):
                    M[j, :] = U[i * m + j] * V[j * p + k]
                try:
                    W_new, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                    W[k * n + i, :] = W_new
                except:
                    pass
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    err += (val - 1.0) ** 2
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_err < 1e-8:
            break
    
    if best_err > 1e-6:
        return None
    
    # Round to half-integers
    U_rounded = _round_to_half_ints(best_U.tolist(), best_V.tolist(), best_W.tolist())
    
    # Verify
    if _verify_decomposition(U_rounded, best_V.tolist(), best_W.tolist()):
        return U_rounded, best_V.tolist(), best_W.tolist()
        
    # Try rounding to integers specifically?
    U_int = [[round(x) for x in row] for row in best_U.tolist()]
    V_int = [[round(x) for x in row] for row in best_V.tolist()]
    W_int = [[round(x) for x in row] for row in best_W.tolist()]
    
    if _verify_decomposition(U_int, V_int, W_int):
        return U_int, V_int, W_int

    return None

def _find_decomposition(n=3, m=3, p=3, time_limit=110):
    """
    Attempt to find a rank-minimal decomposition using ALS.
    """
    start_time = time.time()
    
    # Try ranks 23, 24, 25, 26
    for r in [23, 24, 25, 26]:
        if time.time() - start_time > time_limit - 10:
            break
            
        # Allocate budget per rank
        budget = 20.0 if r == 23 else 10.0
        restarts = 5 if r == 23 else 3
        
        for restart in range(restarts):
            if time.time() - start_time > time_limit - 5:
                break
            
            np.random.seed(hash((r, restart)) % 2**32)
            result = _als_decomposition(r, n, m, p, max_iter=50, time_budget=budget / restarts)
            if result is not None:
                U, V, W = result
                return {"U": U, "V": V, "W": W}
                
    # Fallback to schoolbook
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

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Try to find a better decomposition (rank < 27)
    result = _find_decomposition(n, m, p, time_limit=110)
    
    # Safety check: ensure we didn't exceed time
    if time.time() - start_time > 115:
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
        
    return result
