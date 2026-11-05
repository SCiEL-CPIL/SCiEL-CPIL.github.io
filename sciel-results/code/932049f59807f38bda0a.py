import time
import numpy as np

_START_TIME = time.time()
TIME_LIMIT = 110.0

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    """
    Construct a rank-reduced tensor decomposition for 4x4 * 4x6 matrix multiplication.
    Rank 77: 49 (Strassen 4x4x4) + 28 (4x4x2 via 4x 2x2x2 Strassen).
    """
    n, m, p = 4, 4, 6
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # PART 1: 4x4 * 4x4 (columns 0-3 of B) using recursive Strassen (rank 49)
    if _time_remaining() > 5.0:
        _add_strassen_4x4(0, 0, 0, U_cols, V_cols, W_cols, n, m, p)
    
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
    
    if _verify_decomposition(U_final, V_final, W_final, n, m, p):
        return {"U": U_final, "V": V_final, "W": W_final}
    
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}

def _strassen_2x2_offsets():
    """Return list of (u_dict, v_dict, w_dict) for 2x2 Strassen."""
    return [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]

def _add_strassen_2x2(oi, oj, ok, U_cols, V_cols, W_cols, n, m, p):
    for u_pat, v_pat, w_pat in _strassen_2x2_offsets():
        u_col = [0] * (n * m)
        v_col = [0] * (m * p)
        w_col = [0] * (p * n)
        
        for (r, c), coeff in u_pat.items():
            ii = oi + r
            jj = oj + c
            if 0 <= ii < n and 0 <= jj < m:
                u_col[ii * m + jj] += coeff
        
        for (r, c), coeff in v_pat.items():
            jj = oj + r
            kk = ok + c
            if 0 <= jj < m and 0 <= kk < p:
                v_col[jj * p + kk] += coeff
        
        for (r, c), coeff in w_pat.items():
            ii = oi + r
            kk = ok + c
            if 0 <= ii < n and 0 <= kk < p:
                w_col[kk * n + ii] += coeff
        
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)

def _strassen_4x4_block_combos():
    """
    Return list of (u_block_map, v_block_map, w_block_map) for 4x4 Strassen.
    Block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1) at the 2x2 block level.
    """
    return [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),
    ]

def _add_strassen_4x4(oi, oj, ok, U_cols, V_cols, W_cols, n, m, p):
    block_offsets = {0: (0, 0), 1: (0, 2), 2: (2, 0), 3: (2, 2)}
    base_2x2 = _strassen_2x2_offsets()
    
    for u_blocks, v_blocks, w_blocks in _strassen_4x4_block_combos():
        for u_pat, v_pat, w_pat in base_2x2:
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            
            for bidx, bcoeff in u_blocks.items():
                br, bc = block_offsets[bidx]
                for (r, c), coeff in u_pat.items():
                    ii = oi + br + r
                    jj = oj + bc + c
                    if 0 <= ii < n and 0 <= jj < m:
                        u_col[ii * m + jj] += bcoeff * coeff
            
            for bidx, bcoeff in v_blocks.items():
                br, bc = block_offsets[bidx]
                for (r, c), coeff in v_pat.items():
                    jj = oj + br + r
                    kk = ok + bc + c
                    if 0 <= jj < m and 0 <= kk < p:
                        v_col[jj * p + kk] += bcoeff * coeff
            
            for bidx, bcoeff in w_blocks.items():
                br, bc = block_offsets[bidx]
                for (r, c), coeff in w_pat.items():
                    ii = oi + br + r
                    kk = ok + bc + c
                    if 0 <= ii < n and 0 <= kk < p:
                        w_col[kk * n + ii] += bcoeff * coeff
            
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)

def _verify_decomposition(U, V, W, n, m, p):
    U_arr = np.array(U, dtype=np.int64)
    V_arr = np.array(V, dtype=np.int64)
    W_arr = np.array(W, dtype=np.int64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                total = int(np.dot(U_arr[u_idx, :], V_arr[v_idx, :] * W_arr[w_idx, :]))
                if total != 1:
                    return False
    return True
