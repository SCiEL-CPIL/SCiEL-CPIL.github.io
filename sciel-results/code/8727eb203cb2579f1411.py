"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    u_dim = n * m
    v_dim = m * p
    w_dim = p * n
    
    rank = 14
    U = [[0] * rank for _ in range(u_dim)]
    V = [[0] * rank for _ in range(v_dim)]
    W = [[0] * rank for _ in range(w_dim)]
    
    # Strassen decomposition terms for 2x2x2
    # Format: (u_entries, v_entries, w_entries)
    # u_entries: [(i, j, coeff), ...]
    # v_entries: [(j, k, coeff), ...]
    # w_entries: [(k, i, coeff), ...]
    
    terms = [
        # 1: (a00+a11)(b00+b11)
        (
            [(0,0,1), (1,1,1)],
            [(0,0,1), (1,1,1)],
            [(0,0,1), (1,1,1)]
        ),
        # 2: (a10+a11)b00
        (
            [(1,0,1), (1,1,1)],
            [(0,0,1)],
            [(0,1,1), (1,1,-1)]
        ),
        # 3: a00(b01-b11)
        (
            [(0,0,1)],
            [(0,1,1), (1,1,-1)],
            [(1,0,1), (1,1,1)]
        ),
        # 4: a11(b10-b00)
        (
            [(1,1,1)],
            [(1,0,1), (0,0,-1)],
            [(0,0,1), (0,1,1)]
        ),
        # 5: (a00+a01)b11
        (
            [(0,0,1), (0,1,1)],
            [(1,1,1)],
            [(1,0,1), (0,0,-1)]
        ),
        # 6: (a10-a00)(b00+b01)
        (
            [(1,0,1), (0,0,-1)],
            [(0,0,1), (0,1,1)],
            [(1,1,1)]
        ),
        # 7: (a01-a11)(b10+b11)
        (
            [(0,1,1), (1,1,-1)],
            [(1,0,1), (1,1,1)],
            [(0,0,1)]
        ),
    ]
    
    # Populate U, V, W for the two 2x2x2 blocks
    for r in range(7):
        u_ent, v_ent, w_ent = terms[r]
        
        # Block 1: k in {0, 1}
        col1 = r
        for i, j, val in u_ent:
            U[i * m + j][col1] += val
        for j, k, val in v_ent:
            V[j * p + k][col1] += val
        for k, i, val in w_ent:
            W[k * n + i][col1] += val
            
        # Block 2: k in {2, 3}
        col2 = 7 + r
        for i, j, val in u_ent:
            U[i * m + j][col2] += val
        for j, k, val in v_ent:
            V[j * p + (k + 2)][col2] += val
        for k, i, val in w_ent:
            W[(k + 2) * n + i][col2] += val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
