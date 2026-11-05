"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Block decomposition approach
    # Split n=3 into 2+1
    # Split m=4 into 2+2
    # Split p=5 into 2+2+1
    
    # We will use Schoolbook for all blocks to ensure correctness.
    # Rank = 2*2*2 + 2*2*2 + 2*2*1 + 1*2*2 + 1*2*2 + 1*2*1 + 2*2*2 + 2*2*2 + 2*2*1 + 1*2*2 + 1*2*2 + 1*2*1
    # = 8+8+4+4+4+2+8+8+4+4+4+2 = 60
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_schoolbook_block(n_loc, m_loc, p_loc, i_off, j_off, k_off):
        nonlocal col
        for i in range(n_loc):
            for j in range(m_loc):
                for k in range(p_loc):
                    u_idx = (i_off + i) * m + (j_off + j)
                    v_idx = (j_off + j) * p + (k_off + k)
                    w_idx = (k_off + k) * n + (i_off + i)
                    U[u_idx][col] = 1
                    V[v_idx][col] = 1
                    W[w_idx][col] = 1
                    col += 1

    # Blocks:
    # I: 0 (size 2), 1 (size 1)
    # J: 0 (size 2), 1 (size 2)
    # K: 0 (size 2), 1 (size 2), 2 (size 1)
    
    blocks = [
        (0, 0, 0), (0, 0, 1), (0, 0, 2),
        (0, 1, 0), (0, 1, 1), (0, 1, 2),
        (1, 0, 0), (1, 0, 1), (1, 0, 2),
        (1, 1, 0), (1, 1, 1), (1, 1, 2)
    ]
    
    for bi, bj, bk in blocks:
        if bi == 0:
            n_loc, i_off = 2, 0
        else:
            n_loc, i_off = 1, 2
            
        if bj == 0:
            m_loc, j_off = 2, 0
        else:
            m_loc, j_off = 2, 2
            
        if bk == 0:
            p_loc, k_off = 2, 0
        elif bk == 1:
            p_loc, k_off = 2, 2
        else:
            p_loc, k_off = 1, 4
            
        add_schoolbook_block(n_loc, m_loc, p_loc, i_off, j_off, k_off)
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
