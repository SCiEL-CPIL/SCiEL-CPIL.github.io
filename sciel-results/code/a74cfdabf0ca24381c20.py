"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 8
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 multipliers
    # Each multiplier has (U_coeff, V_coeff, W_coeff)
    # U: indices (i, j) where i in {0,1}, j in {0,1}
    # V: indices (j, k) where j in {0,1}, k in {0,1}
    # W: indices (k, i) where k in {0,1}, i in {0,1}
    
    strassen_mults = [
        # (u_indices, v_indices, w_indices)
        # u_indices: list of (i, j, coeff)
        # v_indices: list of (j, k, coeff)
        # w_indices: list of (k, i, coeff)
        
        # M1 = A00*(B00+B11)
        [
            [(0, 0, 1)],
            [(0, 0, 1), (1, 1, 1)],
            [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1)]
        ],
        
        # M2 = A10*(B00+B01)
        [
            [(1, 0, 1)],
            [(0, 0, 1), (0, 1, 1)],
            [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1)]
        ],
        
        # M3 = (A00+A10)*(B10+B11)
        [
            [(0, 0, 1), (1, 0, 1)],
            [(1, 0, 1), (1, 1, 1)],
            [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1)]
        ],
        
        # M4 = (A01-A11)*(B10+B11)
        [
            [(0, 1, 1), (1, 1, -1)],
            [(1, 0, 1), (1, 1, 1)],
            [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1)]
        ],
        
        # M5 = A00*(B01-B11)
        [
            [(0, 0, 1)],
            [(0, 1, 1), (1, 1, -1)],
            [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1)]
        ],
        
        # M6 = (A00+A11)*(B00-B10)
        [
            [(0, 0, 1), (1, 1, 1)],
            [(0, 0, 1), (1, 0, -1)],
            [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1)]
        ],
        
        # M7 = (A10-A00)*(B11-B01)
        [
            [(1, 0, 1), (0, 0, -1)],
            [(1, 1, 1), (0, 1, -1)],
            [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1)]
        ],
    ]
    
    col = 0
    for block in range(4):  # 4 blocks of 2x2x2
        for mult_idx, (u_idx, v_idx, w_idx) in enumerate(strassen_mults):
            # For block b, the j offset is 2*block, k offset is 2*block
            # Actually, let me think about this more carefully.
            
            # We're computing C = A * B where A is 2x4, B is 4x8
            # Split into 4 blocks: for b in 0..3, A[:, 2b:2b+2] * B[2b:2b+2, :]
            # But that's not quite right. Let me reconsider.
            
            # Actually, the standard way to apply Strassen to 2x4x8:
            # Split the middle dimension (m=4) into two groups of 2.
            # Group 0: j in {0,1}, Group 1: j in {2,3}
            # For each group, we have a 2x2x8 multiplication.
            # Then split the output dimension (p=8) into four groups of 2.
            # For each sub-group, we have a 2x2x2 multiplication.
            
            # So for block b (0..3), we're handling k in {2b, 2b+1}
            # And j in {0,1} for group 0, {2,3} for group 1
            
            # Wait, I need to be more systematic. Let me use a different approach.
            
            # For each of the 8 (j_group, k_group) pairs:
            # j_group: 0 or 1 (j in {0,1} or j in {2,3})
            # k_group: 0, 1, 2, 3 (k in {0,1}, {2,3}, {4,5}, {6,7})
            
            pass
    
    # Let me redo this with a cleaner approach
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j_group in range(2):  # j in {0,1} or {2,3}
        for k_group in range(4):  # k in {0,1}, {2,3}, {4,5}, {6,7}
            j_off = 2 * j_group
            k_off = 2 * k_group
            
            for mult_idx, (u_idx, v_idx, w_idx) in enumerate(strassen_mults):
                for (i, j, cu) in u_idx:
                    U[i * m + (j_off + j)][col] += cu
                for (j, k, cv) in v_idx:
                    V[(j_off + j) * p + (k_off + k)][col] += cv
                for (k, i, cw) in w_idx:
                    W[(k_off + k) * n + i][col] += cw
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
