import time
import numpy as np

DIMENSIONS = [2, 5, 5]
_DEADLINE = 140.0

def _schoolbook_block(n, m, p):
    """Schoolbook for n×m by m×p. Returns U(n*m, r), V(m*p, r), W(p*n, r) with r=n*m*p."""
    r = n * m * p
    U = np.zeros((n * m, r), dtype=int)
    V = np.zeros((m * p, r), dtype=int)
    W = np.zeros((p * n, r), dtype=int)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W

def _strassen_2x2():
    """Strassen's algorithm for 2×2 × 2×2. Returns U(4,7), V(4,7), W(4,7)."""
    # Strassen's 7 products:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A10)*(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U vectors (size 4: [A00, A01, A10, A11])
    # V vectors (size 4: [B00, B01, B10, B11])
    # W vectors (size 4: [C00, C01, C10, C11])
    
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00+A11)*(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)*B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A00*(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A11*(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A00+A01)*B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A10-A00)*(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01-A10)*(B10+B11)
    U[1, 6] = 1; U[2, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _verify(n, m, p, U, V, W):
    """Verify exact decomposition."""
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook_full(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    n, m, p = DIMENSIONS  # 2, 5, 5
    start_time = time.time()
    
    # Build Strassen-based decomposition with rank 46
    # Partition: A is 2×5, B is 5×5, C is 2×5
    # A columns: [0:2], [2:4], [4:5]  (sizes 2, 2, 1)
    # B rows: [0:2], [2:4], [4:5]     (sizes 2, 2, 1)
    # B cols: [0:2], [2:4], [4:5]     (sizes 2, 2, 1)
    # C cols: [0:2], [2:4], [4:5]     (sizes 2, 2, 1)
    
    # Block definitions: (A_row_start, A_row_end, A_col_start, A_col_end, 
    #                     B_row_start, B_row_end, B_col_start, B_col_end,
    #                     C_row_start, C_row_end, C_col_start, C_col_end)
    # A is 2×5: rows 0-1, cols 0-4
    # B is 5×5: rows 0-4, cols 0-4
    # C is 2×5: rows 0-1, cols 0-4
    
    blocks = [
        # (a_rs, a_re, a_cs, a_ce, b_rs, b_re, b_cs, b_ce, c_rs, c_re, c_cs, c_ce, method)
        (0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 'strassen'),  # 2x2 * 2x2 -> 2x2
        (0, 2, 0, 2, 0, 2, 2, 4, 0, 2, 2, 4, 'strassen'),  # 2x2 * 2x2 -> 2x2
        (0, 2, 0, 2, 0, 2, 4, 5, 0, 2, 4, 5, 'schoolbook'), # 2x2 * 2x1 -> 2x1
        (0, 2, 2, 4, 2, 4, 0, 2, 0, 2, 0, 2, 'strassen'),  # 2x2 * 2x2 -> 2x2
        (0, 2, 2, 4, 2, 4, 2, 4, 0, 2, 2, 4, 'strassen'),  # 2x2 * 2x2 -> 2x2
        (0, 2, 2, 4, 2, 4, 4, 5, 0, 2, 4, 5, 'schoolbook'), # 2x2 * 2x1 -> 2x1
        (0, 2, 4, 5, 4, 5, 0, 2, 0, 2, 0, 2, 'schoolbook'), # 2x1 * 1x2 -> 2x2
        (0, 2, 4, 5, 4, 5, 2, 4, 0, 2, 2, 4, 'schoolbook'), # 2x1 * 1x2 -> 2x2
        (0, 2, 4, 5, 4, 5, 4, 5, 0, 2, 4, 5, 'schoolbook'), # 2x1 * 1x1 -> 2x1
    ]
    
    # For each block, generate local U, V, W and map to global indices
    all_U = []
    all_V = []
    all_W = []
    
    for (a_rs, a_re, a_cs, a_ce, b_rs, b_re, b_cs, b_ce, 
         c_rs, c_re, c_cs, c_ce, method) in blocks:
        
        a_n = a_re - a_rs  # rows of A block
        a_m = a_ce - a_cs  # cols of A block
        b_m = b_re - b_rs  # rows of B block
        b_p = b_ce - b_cs  # cols of B block
        c_n = c_re - c_rs  # rows of C block
        c_p = c_ce - c_cs  # cols of C block
        
        if method == 'strassen':
            # Must be 2x2 * 2x2 -> 2x2
            assert a_n == 2 and a_m == 2 and b_m == 2 and b_p == 2 and c_n == 2 and c_p == 2
            u_loc, v_loc, w_loc = _strassen_2x2()
        else:
            u_loc, v_loc, w_loc = _schoolbook_block(a_n, a_m, b_p)
        
        rank_loc = u_loc.shape[1]
        
        # Map local indices to global
        u_global = np.zeros((n * m, rank_loc), dtype=int)
        v_global = np.zeros((m * p, rank_loc), dtype=int)
        w_global = np.zeros((p * n, rank_loc), dtype=int)
        
        for li in range(a_n):
            for lj in range(a_m):
                gi = a_rs + li
                gj = a_cs + lj
                u_global[gi * m + gj, :] = u_loc[li * a_m + lj, :]
        
        for lj in range(b_m):
            for lk in range(b_p):
                gj = b_rs + lj
                gk = b_cs + lk
                v_global[gj * p + gk, :] = v_loc[lj * b_p + lk, :]
        
        for lk in range(c_p):
            for li in range(c_n):
                gk = c_cs + lk
                gi = c_rs + li
                w_global[gk * n + gi, :] = w_loc[lk * c_n + li, :]
        
        all_U.append(u_global)
        all_V.append(v_global)
        all_W.append(w_global)
    
    # Concatenate along rank dimension
    U = np.hstack(all_U)
    V = np.hstack(all_V)
    W = np.hstack(all_W)
    
    total_rank = U.shape[1]
    
    # Verify
    if _verify(n, m, p, U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    return _schoolbook_full(n, m, p)
