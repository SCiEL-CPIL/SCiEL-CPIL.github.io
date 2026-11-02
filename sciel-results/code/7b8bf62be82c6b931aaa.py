"""Combined program: Rank-minimal search for 3x3x6 matmul, fallback to schoolbook."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

def _check_half_integer(arr, tol=1e-9):
    """Check if all entries are within tol of half-integers. Return rounded half-integers if valid."""
    doubled = arr * 2.0
    rounded = np.round(doubled)
    if np.max(np.abs(doubled - rounded)) < tol:
        return rounded / 2.0
    return None

def _search_3x3x2(rank, max_attempts=5000, deadline=None):
    """Search for a rank-R decomposition of 3x3x2 matmul tensor.
    U: 9 x R, V: 6 x R, W: 6 x R
    """
    n, m, p = 3, 3, 2
    for attempt in range(max_attempts):
        if deadline is not None and time.time() > deadline:
            break
        U_rand = np.random.randint(-1, 2, size=(n * m, rank))
        V_rand = np.random.randint(-1, 2, size=(m * p, rank))
        
        W_sol = np.zeros((p * n, rank))
        valid = True
        
        for i in range(n):
            for k in range(p):
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = U_rand[i * m + j, :] * V_rand[j * p + k, :]
                
                try:
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
            return U_rand.tolist(), V_rand.tolist(), W_sol.tolist()
    
    return None

def _search_3x3x3(rank, max_attempts=3000, deadline=None):
    """Search for a rank-R decomposition of 3x3x3 matmul tensor.
    U: 9 x R, V: 9 x R, W: 9 x R
    """
    n, m, p = 3, 3, 3
    for attempt in range(max_attempts):
        if deadline is not None and time.time() > deadline:
            break
        U_rand = np.random.randint(-1, 2, size=(n * m, rank))
        V_rand = np.random.randint(-1, 2, size=(m * p, rank))
        
        W_sol = np.zeros((p * n, rank))
        valid = True
        
        for i in range(n):
            for k in range(p):
                M = np.zeros((m, rank))
                for j in range(m):
                    M[j, :] = U_rand[i * m + j, :] * V_rand[j * p + k, :]
                
                try:
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
            return U_rand.tolist(), V_rand.tolist(), W_sol.tolist()
    
    return None

def _verify_decomposition_fast(U, V, W, n, m, p):
    """Fast verification using einsum."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float).reshape(n, m, rank)
    V_arr = np.array(V, dtype=float).reshape(m, p, rank)
    W_arr = np.array(W, dtype=float).reshape(p, n, rank)
    
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Note: W is indexed by (k, i, r) in the reshaped form (p, n, rank)
    # So W_arr[k, i, r] is correct.
    T = np.einsum('ijr,jkr,kir->ijk', U_arr, V_arr, W_arr, optimize=True)
    
    # Expected T is all 1s
    return np.allclose(T, np.ones((n, m, p)), atol=1e-9)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    # Strategy 1: Rank 45 via 3 parallel 3x3x2 blocks
    blocks = []
    for t in range(3):
        if time.time() > deadline:
            break
        result = _search_3x3x2(15, max_attempts=2000, deadline=deadline)
        if result is None:
            blocks = None
            break
        blocks.append(result)
    
    if blocks is not None and len(blocks) == 3:
        R = 45
        U_total = [[0] * R for _ in range(n * m)]
        V_total = [[0] * R for _ in range(m * p)]
        W_total = [[0] * R for _ in range(p * n)]
        
        for t in range(3):
            U_t, V_t, W_t = blocks[t]
            offset = 15 * t
            for i in range(3):
                for j in range(3):
                    for r in range(15):
                        U_total[i * 3 + j][offset + r] = U_t[i * 3 + j][r]
            for j in range(3):
                for k_local in range(2):
                    for r in range(15):
                        k_global = 2 * t + k_local
                        V_total[j * 6 + k_global][offset + r] = V_t[j * 2 + k_local][r]
            for k_local in range(2):
                for i in range(3):
                    for r in range(15):
                        k_global = 2 * t + k_local
                        W_total[k_global * 3 + i][offset + r] = W_t[k_local * 3 + i][r]
        
        if _verify_decomposition_fast(U_total, V_total, W_total, n, m, p):
            return {"U": U_total, "V": V_total, "W": W_total}

    # Strategy 2: Rank 46 via 2 parallel 3x3x3 blocks
    cores = []
    for t in range(2):
        if time.time() > deadline:
            break
        result = _search_3x3x3(23, max_attempts=2000, deadline=deadline)
        if result is None:
            cores = None
            break
        cores.append(result)
    
    if cores is not None and len(cores) == 2:
        R = 46
        U_total = [[0] * R for _ in range(n * m)]
        V_total = [[0] * R for _ in range(m * p)]
        W_total = [[0] * R for _ in range(p * n)]
        
        for t in range(2):
            U_t, V_t, W_t = cores[t]
            offset = 23 * t
            for i in range(3):
                for j in range(3):
                    for r in range(23):
                        U_total[i * 3 + j][offset + r] = U_t[i * 3 + j][r]
            for j in range(3):
                for k_local in range(3):
                    for r in range(23):
                        k_global = 3 * t + k_local
                        V_total[j * 6 + k_global][offset + r] = V_t[j * 3 + k_local][r]
            for k_local in range(3):
                for i in range(3):
                    for r in range(23):
                        k_global = 3 * t + k_local
                        W_total[k_global * 3 + i][offset + r] = W_t[k_local * 3 + i][r]
        
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
