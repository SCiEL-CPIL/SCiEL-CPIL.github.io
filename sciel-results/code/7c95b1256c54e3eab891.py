"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=9
    # We decompose the 2x3 * 3x9 multiplication as three independent 2x3 * 3x3 block multiplications.
    # For each 2x3 * 3x3 block, we use the schoolbook rank-18 decomposition.
    # Total rank = 3 * 18 = 54.
    
    p_block = 3
    num_blocks = p // p_block  # 3
    
    cols_U = []
    cols_V = []
    cols_W = []
    
    for b in range(num_blocks):
        for i in range(n):
            for j in range(m):
                for k in range(p_block):
                    u_idx = i * m + j
                    v_idx = j * p + (b * p_block + k)
                    w_idx = (b * p_block + k) * n + i
                    
                    col_u = [0] * (n * m)
                    col_v = [0] * (m * p)
                    col_w = [0] * (p * n)
                    
                    col_u[u_idx] = 1
                    col_v[v_idx] = 1
                    col_w[w_idx] = 1
                    
                    cols_U.append(col_u)
                    cols_V.append(col_v)
                    cols_W.append(col_w)
    
    rank = len(cols_U)
    
    U = [[cols_U[c][r] for c in range(rank)] for r in range(n * m)]
    V = [[cols_V[c][r] for c in range(rank)] for r in range(m * p)]
    W = [[cols_W[c][r] for c in range(rank)] for r in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
