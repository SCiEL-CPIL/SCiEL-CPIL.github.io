"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Shapes
    u_dim = n * m  # 4
    v_dim = m * p  # 8
    w_dim = p * n  # 8
    
    rank = 14
    U = [[0] * rank for _ in range(u_dim)]
    V = [[0] * rank for _ in range(v_dim)]
    W = [[0] * rank for _ in range(w_dim)]
    
    # Strassen terms for 2x2x2: (u_entries, v_entries, w_entries)
    # u_entries: list of (i, j, val)
    # v_entries: list of (j, k, val)
    # w_entries: list of (k, i, val)
    
    strassen_terms = [
        # 1: M1 = (a00+a11)(b00+b11)
        ([(0,0,1), (1,1,1)],
         [(0,0,1), (1,1,1)],
         [(0,0,1), (1,1,1)]),
        # 2: M2 = (a10+a11)b00
        ([(1,0,1), (1,1,1)],
         [(0,0,1)],
         [(0,1,1), (1,1,-1)]),
        # 3: M3 = a00(b01-b11)
        ([(0,0,1)],
         [(0,1,1), (1,1,-1)],
         [(1,0,1), (1,1,1)]),
        # 4: M4 = a11(b10-b00)
        ([(1,1,1)],
         [(1,0,1), (0,0,-1)],
         [(0,0,1), (0,1,1)]),
        # 5: M5 = (a00+a01)b11
        ([(0,0,1), (0,1,1)],
         [(1,1,1)],
         [(1,0,1), (0,0,-1)]),
        # 6: M6 = (a10-a00)(b00+b01)
        ([(1,0,1), (0,0,-1)],
         [(0,0,1), (0,1,1)],
         [(1,1,1)]),
        # 7: M7 = (a01-a11)(b10+b11)
        ([(0,1,1), (1,1,-1)],
         [(1,0,1), (1,1,1)],
         [(0,0,1)]),
    ]
    
    # Middle-product decomposition:
    # Block 1 (k=0,1): columns 0-6
    # Block 2 (k=2,3): columns 7-13
    # V-factors for block 2 are linear combinations sharing support with block 1.
    
    for r in range(7):
        u_entries, v_entries, w_entries = strassen_terms[r]
        
        # Block 1 (k=0,1): direct Strassen
        col1 = r
        for i, j, val in u_entries:
            U[i * m + j][col1] += val
        for j, k, val in v_entries:
            V[j * p + k][col1] += val
        for k, i, val in w_entries:
            W[k * n + i][col1] += val
        
        # Block 2 (k=2,3): shifted Strassen with shared V-support
        col2 = 7 + r
        for i, j, val in u_entries:
            U[i * m + j][col2] += val
        for j, k, val in v_entries:
            k_new = k + 2
            V[j * p + k_new][col2] += val
        for k, i, val in w_entries:
            k_new = k + 2
            W[k_new * n + i][col2] += val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
