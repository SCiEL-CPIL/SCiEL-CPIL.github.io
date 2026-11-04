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

def _als_decomposition(r, n=3, m=3, p=3, max_iter=100, time_budget=10.0, init_method='random'):
    """
    Try to find a rank-r decomposition using ALS (Alternating Least Squares).
    """
    start = time.time()
    
    # Initialize based on method
    if init_method == 'structured':
        # Use a structured initialization that respects the tensor structure
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        # Initialize with small random values but with some structure
        for c in range(r):
            # Each column gets a random "phase" pattern
            phase = np.random.randn(1) * 0.3
            U[:, c] = np.random.randn(n * m) * 0.3 + phase
            V[:, c] = np.random.randn(m * p) * 0.3 + phase
            W[:, c] = np.random.randn(p * n) * 0.3 + phase
    elif init_method == 'sparse':
        # Sparse initialization - most entries zero, some non-zero
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        for c in range(r):
            # Pick a few random non-zero positions
            n_nonzero = max(1, r // 5)
            U_idx = np.random.choice(n * m, n_nonzero, replace=False)
            U[U_idx, c] = np.random.choice([-1, 1], n_nonzero) * np.random.uniform(0.5, 1.5, n_nonzero)
            V_idx = np.random.choice(m * p, n_nonzero, replace=False)
            V[V_idx, c] = np.random.choice([-1, 1], n_nonzero) * np.random.uniform(0.5, 1.5, n_nonzero)
            W_idx = np.random.choice(p * n, n_nonzero, replace=False)
            W[W_idx, c] = np.random.choice([-1, 1], n_nonzero) * np.random.uniform(0.5, 1.5, n_nonzero)
    else:
        # Random initialization
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
        
        if best_err < 1e-10:
            break
    
    if best_err > 1e-6:
        return None
    
    # Polish: do a few more iterations with tighter tolerance
    U_polish = best_U.copy()
    V_polish = best_V.copy()
    W_polish = best_W.copy()
    
    polish_iter = min(20, max_iter // 2)
    for _ in range(polish_iter):
        if time.time() - start > time_budget:
            break
        
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, r))
                for k in range(p):
                    M[k, :] = V_polish[j * p + k] * W_polish[k * n + i]
                try:
                    U_new, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    U_polish[i * m + j, :] = U_new
                except:
                    pass
        
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, r))
                for i in range(n):
                    M[i, :] = U_polish[i * m + j] * W_polish[k * n + i]
                try:
                    V_new, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                    V_polish[j * p + k, :] = V_new
                except:
                    pass
        
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, r))
                for j in range(m):
                    M[j, :] = U_polish[i * m + j] * V_polish[j * p + k]
                try:
                    W_new, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                    W_polish[k * n + i, :] = W_new
                except:
                    pass
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_polish[i*m+j] * V_polish[j*p+k] * W_polish[k*n+i])
                    err += (val - 1.0) ** 2
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U_polish.copy(), V_polish.copy(), W_polish.copy()
    
    # Try multiple rounding strategies
    def try_rounding(U_mat, V_mat, W_mat, round_func):
        U_r = round_func(U_mat)
        V_r = round_func(V_mat)
        W_r = round_func(W_mat)
        if _verify_decomposition(U_r, V_r, W_r):
            return U_r, V_r, W_r
        return None
    
    # Half-integer rounding
    result = try_rounding(best_U.tolist(), best_V.tolist(), best_W.tolist(), 
                          lambda M: [[round(x * 2) / 2 for x in row] for row in M])
    if result:
        return result
    
    # Integer rounding
    result = try_rounding(best_U.tolist(), best_V.tolist(), best_W.tolist(),
                          lambda M: [[round(x) for x in row] for row in M])
    if result:
        return result
    
    # Try rounding to nearest 0.25
    result = try_rounding(best_U.tolist(), best_V.tolist(), best_W.tolist(),
                          lambda M: [[round(x * 4) / 4 for x in row] for row in M])
    if result:
        return result
    
    return None

def _find_decomposition(n=3, m=3, p=3, time_limit=110):
    """
    Attempt to find a rank-minimal decomposition using ALS.
    """
    start_time = time.time()
    
    # Try ranks 22, 23, 24, 25, 26
    for r in [22, 23, 24, 25, 26]:
        if time.time() - start_time > time_limit - 10:
            break
            
        # Allocate budget per rank - give more time to lower ranks
        if r <= 23:
            budget = 25.0
            restarts = 8
        elif r <= 24:
            budget = 15.0
            restarts = 5
        else:
            budget = 10.0
            restarts = 3
        
        init_methods = ['random', 'structured', 'sparse']
        
        for restart in range(restarts):
            if time.time() - start_time > time_limit - 5:
                break
            
            np.random.seed(hash((r, restart)) % 2**32)
            init_method = init_methods[restart % len(init_methods)]
            result = _als_decomposition(r, n, m, p, max_iter=80, time_budget=budget / restarts, 
                                        init_method=init_method)
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
        
    # Verify the result before returning
    U_arr = np.array(result['U'])
    V_arr = np.array(result['V'])
    W_arr = np.array(result['W'])
    rank = U_arr.shape[1]
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_arr[i * m + j] * V_arr[j * p + k] * W_arr[k * n + i])
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
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
        
    return result
