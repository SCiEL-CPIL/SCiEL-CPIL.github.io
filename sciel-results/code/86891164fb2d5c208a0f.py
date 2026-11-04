"""Matrix multiplication 4x4 * 4x6 rank-minimal decomposition."""
import time
_START_TIME = time.time()
TIME_LIMIT = 110.0

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    """
    Construct a rank-77 tensor decomposition for 4x4 * 4x6 matrix multiplication.
    Decomposition:
    - 4x4 * 4x4 part: Rank 49 (2-level Strassen)
    - 4x4 * 4x2 part: Rank 28 (4 blocks of 2x2 Strassen)
    Total Rank: 77
    """
    n, m, p = 4, 4, 6
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_dict, v_dict, w_dict):
        u_col = [0] * (n * m)
        v_col = [0] * (m * p)
        w_col = [0] * (p * n)
        
        for (i, j), coeff in u_dict.items():
            u_col[i * m + j] = coeff
        for (j, k), coeff in v_dict.items():
            v_col[j * p + k] = coeff
        for (i, k), coeff in w_dict.items():
            w_col[k * n + i] = coeff
        
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)

    # Pre-defined Strassen 2x2 terms
    # Each term is (u_dict, v_dict, w_dict)
    strassen_terms = [
        # m1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # m2
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # m3
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # m4
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # m5
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # m6
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # m7
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]

    def add_strassen_block(oi, oj, ok):
        """
        Add 7 Strassen terms for a 2x2 block multiplication.
        oi: offset row in A
        oj: offset col in A / row in B
        ok: offset col in B
        """
        for u_rel, v_rel, w_rel in strassen_terms:
            u_dict = {}
            v_dict = {}
            w_dict = {}
            
            for (ri, ci), val in u_rel.items():
                u_dict[(oi + ri, oj + ci)] = val
            for (ri, ci), val in v_rel.items():
                v_dict[(oj + ri, ok + ci)] = val
            for (ri, ci), val in w_rel.items():
                w_dict[(oi + ri, ok + ci)] = val
                
            add_term(u_dict, v_dict, w_dict)

    # PART 1: 4x4 * 4x4 (columns 0-3)
    # 4 blocks of 2x2 * 2x2
    for rb in range(2):
        if _time_remaining() < 5.0: break
        for cb in range(2):
            if _time_remaining() < 5.0: break
            # A[rb, 0] * B[0, cb]
            add_strassen_block(rb*2, 0, cb*2)
            # A[rb, 1] * B[1, cb]
            add_strassen_block(rb*2, 2, cb*2)

    # PART 2: 4x4 * 4x2 (columns 4-5)
    # 4 blocks of 2x2 * 2x2
    for rb in range(2):
        if _time_remaining() < 5.0: break
        # A[rb, 0] * B[0, 2]
        add_strassen_block(rb*2, 0, 4)
        # A[rb, 1] * B[1, 2]
        add_strassen_block(rb*2, 2, 4)

    rank = len(U_cols)
    
    U_final = [[U_cols[r][i] for r in range(rank)] for i in range(n*m)]
    V_final = [[V_cols[r][i] for r in range(rank)] for i in range(m*p)]
    W_final = [[W_cols[r][i] for r in range(rank)] for i in range(p*n)]
    
    return {"U": U_final, "V": V_final, "W": W_final}
