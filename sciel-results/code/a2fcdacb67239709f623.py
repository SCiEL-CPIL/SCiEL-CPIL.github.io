"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 35 decomposition using Strassen's algorithm for the 2x2 block structure
    # with 5 independent vector components.
    
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for A (2x2)
    # A indices: 0: A00, 1: A01, 2: A10, 3: A11
    strassen_A = [
        [1, 0, 0, 1], # M1: A00 + A11
        [0, 0, 1, 1], # M2: A10 + A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00 + A01
        [-1, 0, 1, 0], # M6: A10 - A00
        [0, 1, 0, -1]  # M7: A01 - A11
    ]
    
    # Strassen coefficients for B blocks
    # B blocks: 00 (0-4), 01 (5-9), 10 (10-14), 11 (15-19)
    # Each M_k is a linear combination of B blocks.
    # We define for each M_k, which B blocks are involved and with what sign.
    # Then we map to specific columns t (0-4).
    
    # M1: B00 + B11
    # M2: B00
    # M3: B01 - B11
    # M4: B10 - B00
    # M5: B11
    # M6: B00 + B01
    # M7: B10 + B11
    
    # Structure: list of (block_index, sign) for each k
    # Block indices: 0->B00, 1->B01, 2->B10, 3->B11
    strassen_B_blocks = [
        [(0, 1), (3, 1)],   # M1
        [(0, 1)],           # M2
        [(1, 1), (3, -1)],  # M3
        [(2, 1), (0, -1)],  # M4
        [(3, 1)],           # M5
        [(0, 1), (1, 1)],   # M6
        [(2, 1), (3, 1)]    # M7
    ]
    
    # Output C blocks
    # C blocks: 00 (0-4), 01 (5-9), 10 (10-14), 11 (15-19)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Structure: list of (M_index, sign) for each C block
    # M indices: 0->M1, 1->M2, 2->M3, 3->M4, 4->M5, 5->M6, 6->M7
    output_C_blocks = [
        [(0, 1), (3, 1), (4, -1), (6, 1)],  # C00
        [(2, 1), (4, 1)],                   # C01
        [(1, 1), (3, 1)],                   # C10
        [(0, 1), (1, -1), (2, 1), (5, 1)]   # C11
    ]
    
    # Fill U, V, W
    for k in range(7):
        for t in range(5):
            col = k * 5 + t
            
            # U
            for i in range(4):
                U[i][col] = strassen_A[k][i]
                
            # V
            for b_idx, sign in strassen_B_blocks[k]:
                # b_idx 0->0-4, 1->5-9, 2->10-14, 3->15-19
                base = b_idx * 5
                V[base + t][col] = sign
                
            # W
            # We need to sum contributions from all M_k to the C blocks.
            # It is easier to iterate over C blocks and add contributions.
            # But we are iterating over columns (k,t).
            # So for a fixed k, we add to the relevant C blocks.
            
            # Let's precompute which C blocks M_k contributes to.
            # We can just loop through output_C_blocks and check if k is present.
            for c_idx, contributions in enumerate(output_C_blocks):
                base = c_idx * 5
                for m_idx, sign in contributions:
                    if m_idx == k:
                        W[base + t][col] += sign
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
