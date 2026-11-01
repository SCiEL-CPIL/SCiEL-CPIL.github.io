# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=5, p=6
    
    # Max possible rank estimate
    # Term 1 (2x1x6): 12
    # Term 2 (2x2x6): 3 * 7 = 21
    # Term 3 (2x2x6): 3 * 7 = 21
    # Total: 54
    
    max_rank = 54
    U = [[0] * max_rank for _ in range(n * m)]
    V = [[0] * max_rank for _ in range(m * p)]
    W = [[0] * max_rank for _ in range(p * n)]
    
    current_rank = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        nonlocal current_rank
        U[u_idx][current_rank] += coeff
        V[v_idx][current_rank] += coeff
        W[w_idx][current_rank] += coeff
        current_rank += 1
        
    # Helper to add Strassen block
    def add_strassen(i_off, j_off, k_off):
        nonlocal current_rank
        # Local indices 0,1 for A rows/cols, B rows/cols, C rows/cols
        # U index: (i_off+ai)*m + (j_off+aj)
        # V index: (j_off+bj)*p + (k_off+bk)
        # W index: (k_off+ck)*n + (i_off+ci)
        
        terms = [
            (
                {(1,0):1, (1,1):1}, 
                {(0,0):1}, 
                {(0,1):1, (1,0):1}
            ),
            (
                {(0,0):1, (0,1):1}, 
                {(1,0):1}, 
                {(0,0):-1, (1,0):-1}
            ),
            (
                {(0,0):1}, 
                {(0,1):1, (1,1):-1}, 
                {(0,1):1, (1,1):1}
            ),
            (
                {(1,1):1}, 
                {(1,0):1, (0,0):-1}, 
                {(0,0):1, (1,0):1}
            ),
            (
                {(0,0):1, (1,1):1}, 
                {(0,0):1, (1,1):1}, 
                {(0,0):1, (1,1):1}
            ),
            (
                {(0,1):1, (1,1):-1}, 
                {(1,0):1, (1,1):1}, 
                {(0,0):1}
            ),
            (
                {(0,0):1, (1,0):-1}, 
                {(0,0):1, (0,1):1}, 
                {(1,1):1}
            )
        ]
        
        for u_dict, v_dict, w_dict in terms:
            for (ai, aj), coeff in u_dict.items():
                row = (i_off + ai) * m + (j_off + aj)
                U[row][current_rank] += coeff
            for (bj, bk), coeff in v_dict.items():
                row = (j_off + bj) * p + (k_off + bk)
                V[row][current_rank] += coeff
            for (ci, ck), coeff in w_dict.items():
                row = (k_off + ck) * n + (i_off + ci)
                W[row][current_rank] += coeff
            current_rank += 1

    # Term 1: A[:, 0] * B[0, :]
    # A is 2x1, B is 1x6.
    # i in 0..1, k in 0..5
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            add_term(u_idx, v_idx, w_idx)
            
    # Term 2: A[:, 1:3] * B[1:3, :]
    # A is 2x2, B is 2x6.
    # Split B into 3 blocks of 2 cols.
    for k_block in range(0, p, 2):
        add_strassen(0, 1, k_block)
        
    # Term 3: A[:, 3:5] * B[3:5, :]
    # A is 2x2, B is 2x6.
    for k_block in range(0, p, 2):
        add_strassen(0, 3, k_block)
        
    # Trim to actual rank
    U = [row[:current_rank] for row in U]
    V = [row[:current_rank] for row in V]
    W = [row[:current_rank] for row in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
