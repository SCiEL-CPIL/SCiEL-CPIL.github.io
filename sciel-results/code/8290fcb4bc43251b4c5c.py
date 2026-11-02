import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s margin
    
    n, m, p = DIMENSIONS
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vals, v_vals, w_vals):
        if time.time() > deadline:
            return False
        U_row = [0] * (n * m)
        for idx, val in u_vals.items():
            U_row[idx] = val
        V_row = [0] * (m * p)
        for idx, val in v_vals.items():
            V_row[idx] = val
        W_row = [0] * (p * n)
        for idx, val in w_vals.items():
            W_row[idx] = val
        U_cols.append(U_row)
        V_cols.append(V_row)
        W_cols.append(W_row)
        return True
        
    def add_schoolbook(i_indices, j_indices, k_indices):
        for i in i_indices:
            for j in j_indices:
                for k in k_indices:
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    if not add_term({u_idx: 1}, {v_idx: 1}, {w_idx: 1}):
                        return False
        return True
        
    def add_strassen(i_offset, j_offset, k_offset):
        # M1
        if not add_term(
            {(i_offset+0)*m + (j_offset+0): 1, (i_offset+1)*m + (j_offset+1): 1},
            {(j_offset+0)*p + (k_offset+0): 1, (j_offset+1)*p + (k_offset+1): 1},
            {(k_offset+0)*n + (i_offset+0): 1, (k_offset+1)*n + (i_offset+1): 1}
        ): return False
        # M2
        if not add_term(
            {(i_offset+1)*m + (j_offset+0): 1, (i_offset+1)*m + (j_offset+1): 1},
            {(j_offset+0)*p + (k_offset+0): 1},
            {(k_offset+0)*n + (i_offset+1): 1, (k_offset+1)*n + (i_offset+1): -1}
        ): return False
        # M3
        if not add_term(
            {(i_offset+0)*m + (j_offset+0): 1},
            {(j_offset+1)*p + (k_offset+0): 1, (j_offset+1)*p + (k_offset+1): -1},
            {(k_offset+0)*n + (i_offset+0): 1, (k_offset+1)*n + (i_offset+1): 1}
        ): return False
        # M4
        if not add_term(
            {(i_offset+1)*m + (j_offset+1): 1},
            {(j_offset+0)*p + (k_offset+1): 1, (j_offset+0)*p + (k_offset+0): -1},
            {(k_offset+0)*n + (i_offset+0): 1, (k_offset+0)*n + (i_offset+1): 1}
        ): return False
        # M5
        if not add_term(
            {(i_offset+0)*m + (j_offset+0): 1, (i_offset+0)*m + (j_offset+1): 1},
            {(j_offset+1)*p + (k_offset+1): 1},
            {(k_offset+0)*n + (i_offset+0): -1, (k_offset+1)*n + (i_offset+0): 1}
        ): return False
        # M6
        if not add_term(
            {(i_offset+1)*m + (j_offset+0): 1, (i_offset+0)*m + (j_offset+0): -1},
            {(j_offset+0)*p + (k_offset+0): 1, (j_offset+1)*p + (k_offset+0): 1},
            {(k_offset+1)*n + (i_offset+1): 1}
        ): return False
        # M7
        if not add_term(
            {(i_offset+0)*m + (j_offset+1): 1, (i_offset+1)*m + (j_offset+1): -1},
            {(j_offset+1)*p + (k_offset+1): 1, (j_offset+0)*p + (k_offset+1): 1},
            {(k_offset+0)*n + (i_offset+0): 1}
        ): return False
        return True

    # Execute blocks
    if not add_strassen(0, 0, 0): return {"U": [], "V": [], "W": []}
    if not add_strassen(0, 0, 2): return {"U": [], "V": [], "W": []}
    if not add_strassen(0, 2, 0): return {"U": [], "V": [], "W": []}
    if not add_strassen(0, 2, 2): return {"U": [], "V": [], "W": []}
    
    if not add_schoolbook([0, 1], [0, 1], [4]): return {"U": [], "V": [], "W": []}
    if not add_schoolbook([0, 1], [2, 3], [4]): return {"U": [], "V": [], "W": []}
    if not add_schoolbook([0, 1], [4], [0, 1]): return {"U": [], "V": [], "W": []}
    if not add_schoolbook([0, 1], [4], [2, 3]): return {"U": [], "V": [], "W": []}
    if not add_schoolbook([0, 1], [4], [4]): return {"U": [], "V": [], "W": []}
    
    rank = len(U_cols)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        for i in range(n * m):
            U[i][r] = U_cols[r][i]
        for j in range(m * p):
            V[j][r] = V_cols[r][j]
        for k in range(p * n):
            W[k][r] = W_cols[r][k]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
