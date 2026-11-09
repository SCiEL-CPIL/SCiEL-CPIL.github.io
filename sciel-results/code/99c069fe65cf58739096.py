"""Matrix multiplication 2x5x6 rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices: U[i*2+j, c] for A[i,j], V[i*2+j, c] for B[i,j], W[i*2+j, c] for C[i,j]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    
    # M2: (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    
    # M3: A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    
    # M4: A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    
    # M5: (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    
    # M6: (A10-A00)(B00+B10)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    
    # M7: (A01-A11)(B01+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    
    # W: reconstruct C from M1..M7
    # C00 = M1+M4-M5+M7
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
    # C01 = M3+M5
    W[1, 2] = 1; W[1, 4] = 1
    # C10 = M2+M4
    W[2, 1] = 1; W[2, 3] = 1
    # C11 = M1-M2+M3+M6
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify the decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _als_search(n, m, p, target_rank, time_limit, seed):
    """Try to find a rank-decomposition of the given rank using ALS."""
    if target_rank < max(n, m, p):
        return None
    
    start = time.time()
    rng = np.random.RandomState(seed)
    
    U = rng.randn(n * m, target_rank)
    V = rng.randn(m * p, target_rank)
    W = np.zeros((p * n, target_rank))
    # Initialize W to identity-like or random? 
    # Better to initialize all randomly.
    W = rng.randn(p * n, target_rank)
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_err = float('inf')
    
    max_iters = 200
    tol = 1e-10
    
    for it in range(max_iters):
        if time.time() - start > time_limit:
            break
        
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, target_rank))
                for k in range(p):
                    M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                try:
                    sol, _, _, _ = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                    U_new[i * m + j, :] = sol
                except:
                    U_new[i * m + j, :] = U[i * m + j, :]
        
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, target_rank))
                for i in range(n):
                    M[i, :] = U_new[i * m + j, :] * W[k * n + i, :]
                try:
                    sol, _, _, _ = np.linalg.lstsq(M.T, np.ones(n), rcond=None)
                    V_new[j * p + k, :] = sol
                except:
                    V_new[j * p + k, :] = V[j * p + k, :]
        
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, target_rank))
                for j in range(m):
                    M[j, :] = U_new[i * m + j, :] * V_new[j * p + k, :]
                try:
                    sol, _, _, _ = np.linalg.lstsq(M.T, np.ones(m), rcond=None)
                    W_new[k * n + i, :] = sol
                except:
                    W_new[k * n + i, :] = W[k * n + i, :]
        
        U, V, W = U_new, V_new, W_new
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i * m + j]
                    v_val = V[j * p + k]
                    w_val = W[k * n + i]
                    s = np.sum(u_val * v_val * w_val)
                    err += (s - 1.0) ** 2
        err = np.sqrt(err)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < tol:
            break
    
    if best_err > 1e-8:
        return None
        
    U_final = best_U
    V_final = best_V
    W_final = best_W
    
    # Verify exactness
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_final[i * m + j]
                v_val = V_final[j * p + k]
                w_val = W_final[k * n + i]
                s = np.sum(u_val * v_val * w_val)
                if abs(s - 1.0) > 1e-8:
                    return None
    
    return U_final, V_final, W_final

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Build the 54-rank solution
    total_rank = 54
    U_full = np.zeros((10, total_rank))
    V_full = np.zeros((30, total_rank))
    W_full = np.zeros((12, total_rank))
    
    col_offset = 0
    
    # Part 1: A1@B1 (2x1x6) Schoolbook
    for i in range(2):
        for k in range(6):
            U_full[i * 5, col_offset] = 1
            V_full[k, col_offset] = 1
            W_full[k * 2 + i, col_offset] = 1
            col_offset += 1
            
    # Part 2: A2@B2 (2x2x6) Strassen
    U_str, V_str, W_str = _strassen_2x2x2()
    for b in range(3):
        U_full[1, col_offset:col_offset+7] = U_str[0, :]
        U_full[2, col_offset:col_offset+7] = U_str[1, :]
        U_full[6, col_offset:col_offset+7] = U_str[2, :]
        U_full[7, col_offset:col_offset+7] = U_str[3, :]
        
        V_full[6+2*b, col_offset:col_offset+7] = V_str[0, :]
        V_full[6+2*b+1, col_offset:col_offset+7] = V_str[1, :]
        V_full[12+2*b, col_offset:col_offset+7] = V_str[2, :]
        V_full[12+2*b+1, col_offset:col_offset+7] = V_str[3, :]
        
        W_full[4*b, col_offset:col_offset+7] = W_str[0, :]
        W_full[4*b+2, col_offset:col_offset+7] = W_str[1, :]
        W_full[4*b+1, col_offset:col_offset+7] = W_str[2, :]
        W_full[4*b+3, col_offset:col_offset+7] = W_str[3, :]
        col_offset += 7
        
    # Part 3: A3@B3 (2x2x6) Strassen
    for b in range(3):
        U_full[3, col_offset:col_offset+7] = U_str[0, :]
        U_full[4, col_offset:col_offset+7] = U_str[1, :]
        U_full[8, col_offset:col_offset+7] = U_str[2, :]
        U_full[9, col_offset:col_offset+7] = U_str[3, :]
        
        V_full[18+2*b, col_offset:col_offset+7] = V_str[0, :]
        V_full[18+2*b+1, col_offset:col_offset+7] = V_str[1, :]
        V_full[24+2*b, col_offset:col_offset+7] = V_str[2, :]
        V_full[24+2*b+1, col_offset:col_offset+7] = V_str[3, :]
        
        W_full[4*b, col_offset:col_offset+7] = W_str[0, :]
        W_full[4*b+2, col_offset:col_offset+7] = W_str[1, :]
        W_full[4*b+1, col_offset:col_offset+7] = W_str[2, :]
        W_full[4*b+3, col_offset:col_offset+7] = W_str[3, :]
        col_offset += 7
        
    if _verify(U_full, V_full, W_full, n, m, p):
        best_solution = {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    else:
        # Fallback
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
        best_solution = {"U": U, "V": V, "W": W}

    # 2. Search for lower rank
    for rank in range(53, 39, -1):
        if time.time() > deadline:
            break
        for seed in [0, 1, 2]:
            if time.time() > deadline:
                break
            res = _als_search(n, m, p, rank, time_limit=5.0, seed=seed)
            if res is not None:
                U, V, W = res
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                if _verify(U_r, V_r, W_r, n, m, p):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}

    return best_solution

# EVOLVE-BLOCK-END
