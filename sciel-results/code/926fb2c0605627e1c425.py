"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for 2x2x2
    # U[a, b, r], V[b, c, r], W[c, a, r]
    # r in 0..6
    U_strassen = [
        [0, 0, 0, 0, 0, 0, 0], # a=0, b=0
        [0, 0, 0, 0, 0, 0, 0], # a=0, b=1
        [0, 0, 0, 0, 0, 0, 0], # a=1, b=0
        [0, 0, 0, 0, 0, 0, 0]  # a=1, b=1
    ]
    # Flatten U to list of lists of 7 values for each (a,b)
    # Actually, let's just define the 7 patterns directly in a loop
    
    # Define patterns for U, V, W for local indices a, b, c in {0, 1}
    # We will iterate through 4 blocks and 7 strassen steps
    
    # Block definitions:
    # block_id, i_offset, j_offset, k_offset
    # But i is always 0,1.
    # Block 1: A1 (j 0,1), B11 (j 0,1, k 0,1) -> j_off=0, k_off=0
    # Block 2: A2 (j 2,3), B21 (j 2,3, k 0,1) -> j_off=2, k_off=0
    # Block 3: A1 (j 0,1), B12 (j 0,1, k 2,3) -> j_off=0, k_off=2
    # Block 4: A2 (j 2,3), B22 (j 2,3, k 2,3) -> j_off=2, k_off=2
    
    blocks = [
        (0, 0), # j_off, k_off
        (2, 0),
        (0, 2),
        (2, 2)
    ]
    
    # Strassen step definitions
    # Each step is a tuple of (U_vals, V_vals, W_vals)
    # U_vals: dict {(a,b): val}
    # V_vals: dict {(b,c): val}
    # W_vals: dict {(c,a): val}
    
    steps = []
    
    # M1
    steps.append((
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1}
    ))
    # M2
    steps.append((
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,0): 1, (1,1): -1}
    ))
    # M3
    steps.append((
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(0,1): 1, (1,1): 1}
    ))
    # M4
    steps.append((
        {(1,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (1,0): 1}
    ))
    # M5
    steps.append((
        {(0,0): 1, (0,1): 1},
        {(1,1): 1},
        {(0,0): -1, (0,1): 1}
    ))
    # M6
    steps.append((
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (0,1): 1},
        {(1,1): 1}
    ))
    # M7
    steps.append((
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1}
    ))
    
    for b_idx, (j_off, k_off) in enumerate(blocks):
        for s_idx, (U_pat, V_pat, W_pat) in enumerate(steps):
            r = b_idx * 7 + s_idx
            
            # Fill U
            for (a, b), val in U_pat.items():
                i = a
                j = b + j_off
                U[i * m + j][r] = val
                
            # Fill V
            for (b, c), val in V_pat.items():
                j = b + j_off
                k = c + k_off
                V[j * p + k][r] = val
                
            # Fill W
            for (c, a), val in W_pat.items():
                k = c + k_off
                i = a
                W[k * n + i][r] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
