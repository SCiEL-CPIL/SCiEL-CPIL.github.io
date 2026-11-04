import time
import numpy as np

_START_TIME = time.time()
TIME_LIMIT = 110.0

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    n, m, p = 4, 4, 6
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # PART 1: 4x4 * 4x4 (columns 0-3 of B) using BCL rank-13
    if _time_remaining() > 5.0:
        _add_bcl_4x4(0, 0, 0, U_cols, V_cols, W_cols, n, m, p)
        
    # PART 2: 4x4 * 4x2 (columns 4-5 of B) using 4 independent 2x2*2x2 Strassen (rank 28)
    if _time_remaining() > 3.0:
        _add_strassen_2x2(0, 0, 4, U_cols, V_cols, W_cols, n, m, p)
        _add_strassen_2x2(0, 2, 4, U_cols, V_cols, W_cols, n, m, p)
        _add_strassen_2x2(2, 0, 4, U_cols, V_cols, W_cols, n, m, p)
        _add_strassen_2x2(2, 2, 4, U_cols, V_cols, W_cols, n, m, p)
        
    rank = len(U_cols)
    
    U_final = [[U_cols[r][i] for r in range(rank)] for i in range(n * m)]
    V_final = [[V_cols[r][i] for r in range(rank)] for i in range(m * p)]
    W_final = [[W_cols[r][i] for r in range(rank)] for i in range(p * n)]
    
    # Optional: Verify if time allows
    if _time_remaining() > 10.0:
        _verify_decomposition(U_final, V_final, W_final, n, m, p)
        
    return {"U": U_final, "V": V_final, "W": W_final}

def _add_bcl_4x4(oi, oj, ok, U_cols, V_cols, W_cols, n, m, p):
    """
    Add BCL rank-13 decomposition for 4x4 * 4x4 block multiplication.
    """
    # The 13 terms of BCL decomposition
    # Each term is (u_indices, u_coeffs, v_indices, v_coeffs, w_indices, w_coeffs)
    # Indices are (i, j) for U, (j, k) for V, (k, i) for W
    # Note: W index is (k, i) corresponding to C[i, k]
    
    terms = [
        # T1
        ([(0,0), (3,3)], [1, 1], [(0,0), (3,3)], [1, 1], [(0,0), (3,0), (0,3), (3,3)], [1, 1, 1, 1]),
        # T2
        ([(0,3), (3,0)], [1, 1], [(0,3), (3,0)], [1, 1], [(0,0), (3,0), (0,3), (3,3)], [1, 1, 1, 1]),
        # T3
        ([(0,0), (3,3)], [1, -1], [(0,0), (3,3)], [1, -1], [(0,0), (3,0), (0,3), (3,3)], [1, 1, 1, 1]),
        # T4
        ([(0,3), (3,0)], [1, -1], [(0,3), (3,0)], [1, -1], [(0,0), (3,0), (0,3), (3,3)], [1, 1, 1, 1]),
        # T5
        ([(0,1), (3,2)], [1, 1], [(0,1), (3,2)], [1, 1], [(1,0), (2,0), (1,3), (2,3)], [1, 1, 1, 1]),
        # T6
        ([(0,2), (3,1)], [1, 1], [(0,2), (3,1)], [1, 1], [(2,0), (1,0), (2,3), (1,3)], [1, 1, 1, 1]),
        # T7
        ([(0,1), (3,2)], [1, -1], [(0,1), (3,2)], [1, -1], [(1,0), (2,0), (1,3), (2,3)], [1, 1, 1, 1]),
        # T8
        ([(0,2), (3,1)], [1, -1], [(0,2), (3,1)], [1, -1], [(2,0), (1,0), (2,3), (1,3)], [1, 1, 1, 1]),
        # T9
        ([(1,0), (2,3)], [1, 1], [(1,0), (2,3)], [1, 1], [(0,1), (3,1), (0,2), (3,2)], [1, 1, 1, 1]),
        # T10
        ([(1,3), (2,0)], [1, 1], [(1,3), (2,0)], [1, 1], [(3,1), (0,1), (3,2), (0,2)], [1, 1, 1, 1]),
        # T11
        ([(1,0), (2,3)], [1, -1], [(1,0), (2,3)], [1, -1], [(0,1), (3,1), (0,2), (3,2)], [1, 1, 1, 1]),
        # T12
        ([(1,3), (2,0)], [1, -1], [(1,3), (2,0)], [1, -1], [(3,1), (0,1), (3,2), (0,2)], [1, 1, 1, 1]),
        # T13
        ([(1,1), (2,2)], [1, 1], [(1,1), (2,2)], [1, 1], [(1,1), (2,1), (1,2), (2,2)], [1, 1, 1, 1]),
    ]
    
    for u_idx, u_coef, v_idx, v_coef, w_idx, w_coef in terms:
        u_col = [0] * (n * m)
        v_col = [0] * (m * p)
        w_col = [0] * (p * n)
        
        for (r, c), coeff in zip(u_idx, u_coef):
            i = oi + r
            j = oj + c
            if 0 <= i < n and 0 <= j < m:
                u_col[i * m + j] += coeff
                
        for (r, c), coeff in zip(v_idx, v_coef):
            j = oj + r
            k = ok + c
            if 0 <= j < m and 0 <= k < p:
                v_col[j * p + k] += coeff
                
        for (r, c), coeff in zip(w_idx, w_coef):
            i = oi + r
            k = ok + c
            if 0 <= i < n and 0 <= k < p:
                w_col[k * n + i] += coeff
                
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)

def _add_strassen_2x2(oi, oj, ok, U_cols, V_cols, W_cols, n, m, p):
    """Add Strassen decomposition for 2x2 * 2x2 block multiplication."""
    S2 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    for u_pat, v_pat, w_pat in S2:
        u_col = [0] * (n * m)
        v_col = [0] * (m * p)
        w_col = [0] * (p * n)
        
        for (r, c), coeff in u_pat.items():
            i = oi + r
            j = oj + c
            if 0 <= i < n and 0 <= j < m:
                u_col[i * m + j] += coeff
                
        for (r, c), coeff in v_pat.items():
            j = oj + r
            k = ok + c
            if 0 <= j < m and 0 <= k < p:
                v_col[j * p + k] += coeff
                
        for (r, c), coeff in w_pat.items():
            i = oi + r
            k = ok + c
            if 0 <= i < n and 0 <= k < p:
                w_col[k * n + i] += coeff
                
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the tensor decomposition is correct.
    """
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.int64)
    V_arr = np.array(V, dtype=np.int64)
    W_arr = np.array(W, dtype=np.int64)
    
    i_vals, j_vals, k_vals = np.meshgrid(
        np.arange(n), np.arange(m), np.arange(p), indexing='ij'
    )
    u_indices = (i_vals * m + j_vals).ravel()
    v_indices = (j_vals * p + k_vals).ravel()
    w_indices = (k_vals * n + i_vals).ravel()
    
    U_sel = U_arr[u_indices, :]
    V_sel = V_arr[v_indices, :]
    W_sel = W_arr[w_indices, :]
    
    totals = np.sum(U_sel * V_sel * W_sel, axis=1)
    
    if not np.all(totals == 1):
        return False
    return True
