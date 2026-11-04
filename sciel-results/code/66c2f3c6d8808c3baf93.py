"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 20
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to set a term
    def set_term(u_idx, v_idx, w_idx, col):
        U[u_idx][col] = 1
        V[v_idx][col] = 1
        W[w_idx][col] = 1
        
    col = 0
    
    # 1. Block (i=0, j=0, k=0): 1x1x1
    # i=0, j=0, k=0
    u_idx = 0 * m + 0
    v_idx = 0 * p + 0
    w_idx = 0 * n + 0
    set_term(u_idx, v_idx, w_idx, col)
    col += 1
    
    # 2. Block (i=0, j in 1..3, k=0): 1x3x1
    for j in [1, 2, 3]:
        u_idx = 0 * m + j
        v_idx = j * p + 0
        w_idx = 0 * n + 0
        set_term(u_idx, v_idx, w_idx, col)
        col += 1
        
    # 3. Block (i=1, j=0, k in 1..4): 1x1x4
    for k in [1, 2, 3, 4]:
        u_idx = 1 * m + 0
        v_idx = 0 * p + k
        w_idx = k * n + 1
        set_term(u_idx, v_idx, w_idx, col)
        col += 1
        
    # 4. Block (i=1, j in 1..3, k in 1..4): 1x3x4
    for j in [1, 2, 3]:
        for k in [1, 2, 3, 4]:
            u_idx = 1 * m + j
            v_idx = j * p + k
            w_idx = k * n + 1
            set_term(u_idx, v_idx, w_idx, col)
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
