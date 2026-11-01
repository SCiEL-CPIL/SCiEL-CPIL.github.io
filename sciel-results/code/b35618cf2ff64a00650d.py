"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Using Strassen's algorithm for 2x2x2 blocks
    # There are 8 blocks of size 2x2 in the 2x16 matrix B
    num_blocks = p // 2
    rank_per_block = 7
    rank = num_blocks * rank_per_block
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define Strassen's 7 terms
    # Each term is a list of (coeff, A_idx, B_local_idx) for A and B parts
    # and (coeff, C_local_idx) for C part
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B local indices: 0:e(0,0), 1:f(0,1), 2:g(1,0), 3:h(1,1)
    # C local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    strassen_terms = [
        # Term 1: (a+d)(e+h) -> C00 += 1, C11 += 1
        {
            'A': [(1, 0), (1, 3)], 
            'B': [(1, 0), (1, 3)], 
            'C': [(1, 0), (1, 3)]
        },
        # Term 2: (c+d)e -> C10 += 1, C11 -= 1
        {
            'A': [(1, 2), (1, 3)], 
            'B': [(1, 0)], 
            'C': [(1, 2), (-1, 3)]
        },
        # Term 3: a(f-h) -> C01 += 1, C11 += 1
        {
            'A': [(1, 0)], 
            'B': [(1, 1), (-1, 3)], 
            'C': [(1, 1), (1, 3)]
        },
        # Term 4: d(g-e) -> C00 += 1, C10 += 1
        {
            'A': [(1, 3)], 
            'B': [(1, 2), (-1, 0)], 
            'C': [(1, 0), (1, 2)]
        },
        # Term 5: (a+b)h -> C00 -= 1, C01 += 1
        {
            'A': [(1, 0), (1, 1)], 
            'B': [(1, 3)], 
            'C': [(-1, 0), (1, 1)]
        },
        # Term 6: (c-a)(e+f) -> C11 += 1
        {
            'A': [(1, 2), (-1, 0)], 
            'B': [(1, 0), (1, 1)], 
            'C': [(1, 3)]
        },
        # Term 7: (b-d)(g+h) -> C00 += 1, C11 += 1
        {
            'A': [(1, 1), (-1, 3)], 
            'B': [(1, 2), (1, 3)], 
            'C': [(1, 0), (1, 3)]
        },
    ]
    
    for b in range(num_blocks):
        for s in range(rank_per_block):
            col = b * rank_per_block + s
            term = strassen_terms[s]
            
            # Fill U
            for coeff, a_idx in term['A']:
                U[a_idx][col] += coeff
            
            # Fill V
            # Map local B indices to global V indices
            # e(0) -> 2b, f(1) -> 2b+1, g(2) -> 16+2b, h(3) -> 17+2b
            for coeff, b_local_idx in term['B']:
                if b_local_idx == 0: v_idx = 2 * b
                elif b_local_idx == 1: v_idx = 2 * b + 1
                elif b_local_idx == 2: v_idx = 16 + 2 * b
                elif b_local_idx == 3: v_idx = 17 + 2 * b
                V[v_idx][col] += coeff
            
            # Fill W
            # Map local C indices to global W indices
            # C00(0) -> 4b, C01(1) -> 4b+2, C10(2) -> 4b+1, C11(3) -> 4b+3
            for coeff, c_local_idx in term['C']:
                if c_local_idx == 0: w_idx = 4 * b
                elif c_local_idx == 1: w_idx = 4 * b + 2
                elif c_local_idx == 2: w_idx = 4 * b + 1
                elif c_local_idx == 3: w_idx = 4 * b + 3
                W[w_idx][col] += coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
