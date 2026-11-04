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
    
    def add_term(u_dict, v_dict, w_dict):
        u_col = [0] * (n * m)
        v_col = [0] * (m * p)
        w_col = [0] * (p * n)
        
        for (i, j), coeff in u_dict.items():
            u_col[i * m + j] += coeff
        for (j, k), coeff in v_dict.items():
            v_col[j * p + k] += coeff
        for (i, k), coeff in w_dict.items():
            w_col[k * n + i] += coeff
            
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)
        
    def strassen_2x2(oi, oj, ok):
        # m1
        add_term({(oi, oj): 1, (oi+1, oj+1): 1}, {(oj, ok): 1, (oj+1, ok+1): 1}, {(oi, ok): 1, (oi+1, ok+1): 1})
        # m2
        add_term({(oi+1, oj): 1, (oi+1, oj+1): 1}, {(oj, ok): 1}, {(oi+1, ok): 1, (oi+1, ok+1): -1})
        # m3
        add_term({(oi, oj): 1}, {(oj, ok+1): 1, (oj+1, ok+1): -1}, {(oi, ok+1): 1, (oi+1, ok+1): 1})
        # m4
        add_term({(oi+1, oj+1): 1}, {(oj+1, ok): 1, (oj, ok): -1}, {(oi, ok): 1, (oi+1, ok): 1})
        # m5
        add_term({(oi, oj): 1, (oi, oj+1): 1}, {(oj+1, ok+1): 1}, {(oi, ok): -1, (oi, ok+1): 1})
        # m6
        add_term({(oi+1, oj): 1, (oi, oj): -1}, {(oj, ok): 1, (oj, ok+1): 1}, {(oi+1, ok+1): 1})
        # m7
        add_term({(oi, oj+1): 1, (oi+1, oj+1): -1}, {(oj+1, ok): 1, (oj+1, ok+1): 1}, {(oi, ok): 1})

    def strassen_4x4(oi, oj, ok):
        # Recursive Strassen for 4x4x4
        # M1 = (A00+A11)(B00+B11)
        _strassen_super(oi, oj, ok, {0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1})
        # M2 = (A10+A11)B00
        _strassen_super(oi, oj, ok, {2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1})
        # M3 = A00(B01-B11)
        _strassen_super(oi, oj, ok, {0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1})
        # M4 = A11(B10-B00)
        _strassen_super(oi, oj, ok, {3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1})
        # M5 = (A00+A01)B11
        _strassen_super(oi, oj, ok, {0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1})
        # M6 = (A10-A00)(B00+B01)
        _strassen_super(oi, oj, ok, {2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1})
        # M7 = (A01-A11)(B10+B11)
        _strassen_super(oi, oj, ok, {1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})

    def _strassen_super(oi, oj, ok, u_blocks, v_blocks, w_blocks):
        strassen_2x2_terms = [
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ]
        block_offsets = {0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)}
        
        for u_pat, v_pat, w_pat in strassen_2x2_terms:
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            
            for block_idx, block_coeff in u_blocks.items():
                br, bc = block_offsets[block_idx]
                for (r, c), coeff in u_pat.items():
                    i = oi + 2*br + r
                    j = oj + 2*bc + c
                    if 0 <= i < n and 0 <= j < m:
                        u_col[i * m + j] += block_coeff * coeff
            
            for block_idx, block_coeff in v_blocks.items():
                br, bc = block_offsets[block_idx]
                for (r, c), coeff in v_pat.items():
                    j = oj + 2*br + r
                    k = ok + 2*bc + c
                    if 0 <= j < m and 0 <= k < p:
                        v_col[j * p + k] += block_coeff * coeff
            
            for block_idx, block_coeff in w_blocks.items():
                br, bc = block_offsets[block_idx]
                for (r, c), coeff in w_pat.items():
                    i = oi + 2*br + r
                    k = ok + 2*bc + c
                    if 0 <= i < n and 0 <= k < p:
                        w_col[k * n + i] += block_coeff * coeff
            
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)

    def schoolbook_4x2(oi, oj, ok):
        # Computes C[oi:oi+4, ok:ok+2] = A[oi:oi+4, oj:oj+4] * B[oj:oj+4, ok:ok+2]
        # Rank 8
        for i in range(4):
            for j in range(4):
                for k in range(2):
                    add_term({(oi+i, oj+j): 1}, {(oj+j, ok+k): 1}, {(oi+i, ok+k): 1})

    # Build 4x4x4 part (Rank 49)
    if _time_remaining() > 10.0:
        strassen_4x4(0, 0, 0)
        
    # Build 4x4x2 part (Rank 8)
    # This computes C[:, 4:6] = A[:, 0:4] * B[0:4, 4:6]
    if _time_remaining() > 5.0:
        schoolbook_4x2(0, 0, 4)
        
    rank = len(U_cols)
    
    U_final = [[U_cols[r][i] for r in range(rank)] for i in range(n*m)]
    V_final = [[V_cols[r][i] for r in range(rank)] for i in range(m*p)]
    W_final = [[W_cols[r][i] for r in range(rank)] for i in range(p*n)]
    
    # Verify
    U_arr = np.array(U_final, dtype=np.float64)
    V_arr = np.array(V_final, dtype=np.float64)
    W_arr = np.array(W_final, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                total = np.dot(U_arr[u_idx, :], V_arr[v_idx, :] * W_arr[w_idx, :])
                if abs(total - 1.0) > 1e-9:
                    # Fallback to schoolbook if verification fails
                    rank_fb = n * m * p
                    U_fb = [[0] * rank_fb for _ in range(n * m)]
                    V_fb = [[0] * rank_fb for _ in range(m * p)]
                    W_fb = [[0] * rank_fb for _ in range(p * n)]
                    column = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                U_fb[i * m + j][column] = 1
                                V_fb[j * p + k][column] = 1
                                W_fb[k * n + i][column] = 1
                                column += 1
                    return {"U": U_fb, "V": V_fb, "W": W_fb}
                    
    return {"U": U_final, "V": V_final, "W": W_final}
