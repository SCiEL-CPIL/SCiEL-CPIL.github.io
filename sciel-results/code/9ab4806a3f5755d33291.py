import time
import numpy as np

DIMENSIONS = [3, 3, 6]

def _check_half_integer(arr, tol=1e-9):
    doubled = arr * 2.0
    rounded = np.round(doubled)
    if np.max(np.abs(doubled - rounded)) < tol:
        return rounded / 2.0
    return None

def _search_3x3x3(rank, max_attempts=1000, deadline=None):
    """Search for a rank-R decomposition of 3x3x3 matmul tensor.
    U: 9 x R, V: 9 x R, W: 9 x R
    """
    n, m, p = 3, 3, 3
    for attempt in range(max_attempts):
        if deadline is not None and time.time() > deadline:
            break
        # Generate random U, V with small integers
        U_rand = np.random.randint(-1, 2, size=(n * m, rank))
        V_rand = np.random.randint(-1, 2, size=(m * p, rank))
        
        W_sol = np.zeros((p * n, rank))
        valid = True
        
        for i in range(n):
            for k in range(p):
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = U_rand[i * m + j, :] * V_rand[j * p + k, :]
                
                # Solve M @ w = 1
                try:
                    # Use least squares to find w
                    w_vec, residuals, rank_m, s = np.linalg.lstsq(M, np.ones(m), rcond=None)
                except:
                    valid = False
                    break
                
                w_check = _check_half_integer(w_vec)
                if w_check is None:
                    valid = False
                    break
                W_sol[k * n + i, :] = w_check
        
        if valid:
            # Verify
            U_arr = np.array(U_rand, dtype=float).reshape(n, m, rank)
            V_arr = np.array(V_rand, dtype=float).reshape(m, p, rank)
            W_arr = np.array(W_sol, dtype=float).reshape(p, n, rank)
            T = np.einsum('ijr,jkr,kir->ijk', U_arr, V_arr, W_arr, optimize=True)
            if np.allclose(T, np.ones((n, m, p)), atol=1e-9):
                return U_rand.tolist(), V_rand.tolist(), W_sol.tolist()
    
    return None

def _verify_decomposition_fast(U, V, W, n, m, p):
    rank = len(U[0])
    U_arr = np.array(U, dtype=float).reshape(n, m, rank)
    V_arr = np.array(V, dtype=float).reshape(m, p, rank)
    W_arr = np.array(W, dtype=float).reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U_arr, V_arr, W_arr, optimize=True)
    return np.allclose(T, np.ones((n, m, p)), atol=1e-9)

def construct():
    start_time = time.time()
    deadline = start_time + 110
    n, m, p = DIMENSIONS
    
    # Strategy 1: Try to find rank 46 (2 * 23) via 3x3x3 blocks
    # We need two independent 3x3x3 decompositions of rank 23
    core1 = _search_3x3x3(23, max_attempts=500, deadline=deadline)
    if core1 and time.time() < deadline:
        core2 = _search_3x3x3(23, max_attempts=500, deadline=deadline)
        if core2:
            R = 46
            U_total = [[0] * R for _ in range(n * m)]
            V_total = [[0] * R for _ in range(m * p)]
            W_total = [[0] * R for _ in range(p * n)]
            
            U1, V1, W1 = core1
            U2, V2, W2 = core2
            
            for i in range(3):
                for j in range(3):
                    for r in range(23):
                        U_total[i * 3 + j][r] = U1[i * 3 + j][r]
                        U_total[i * 3 + j][23 + r] = U2[i * 3 + j][r]
            
            for j in range(3):
                for k_local in range(3):
                    for r in range(23):
                        k1 = k_local
                        k2 = 3 + k_local
                        V_total[j * 6 + k1][r] = V1[j * 3 + k_local][r]
                        V_total[j * 6 + k2][23 + r] = V2[j * 3 + k_local][r]
            
            for k_local in range(3):
                for i in range(3):
                    for r in range(23):
                        k1 = k_local
                        k2 = 3 + k_local
                        W_total[k1 * 3 + i][r] = W1[k_local * 3 + i][r]
                        W_total[k2 * 3 + i][23 + r] = W2[k_local * 3 + i][r]
            
            if _verify_decomposition_fast(U_total, V_total, W_total, n, m, p):
                return {"U": U_total, "V": V_total, "W": W_total}

    # Fallback: Schoolbook rank 54
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
