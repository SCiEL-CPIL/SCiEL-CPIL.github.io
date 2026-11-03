import numpy as np

def construct():
    n, m, p = 4, 4, 8
    rank = 98
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen 2x2x2 base case (rank 7)
    # Indices: 0:00, 1:01, 2:10, 3:11
    U_2 = np.zeros((4, 7))
    V_2 = np.zeros((4, 7))
    W_2 = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = -1
    
    # M3 = A00(B01-B11)
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1
    
    # M4 = A11(B10-B00)
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1
    
    # M5 = (A00+A01)B11
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1; W_2[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1

    # Block Strassen specs for 4x4x4
    # A blocks: (coeff, block_idx) where block_idx 0:00, 1:01, 2:10, 3:11
    # B blocks: (coeff, block_idx)
    # C blocks: (coeff, block_idx)
    
    # Define the 7 Strassen steps for the 4x4 level
    # Each step defines how A, B, C blocks are combined
    
    steps = [
        # k=0: M1
        {
            'A': [(1, 0), (1, 3)], # A00 + A11
            'B': [(1, 0), (1, 3)], # B00 + B11
            'C': [(1, 0), (1, 3)]  # C00 + C11
        },
        # k=1: M2
        {
            'A': [(1, 2), (1, 3)], # A10 + A11
            'B': [(1, 0)],         # B00
            'C': [(1, 2), (-1, 3)] # C10 - C11
        },
        # k=2: M3
        {
            'A': [(1, 0)],         # A00
            'B': [(1, 1), (-1, 3)],# B01 - B11
            'C': [(1, 1), (1, 3)]  # C01 + C11
        },
        # k=3: M4
        {
            'A': [(1, 3)],         # A11
            'B': [(1, 2), (-1, 0)],# B10 - B00
            'C': [(1, 0), (1, 2)]  # C00 + C10
        },
        # k=4: M5
        {
            'A': [(1, 0), (1, 1)], # A00 + A01
            'B': [(1, 3)],         # B11
            'C': [(-1, 0), (1, 1)] # -C00 + C01
        },
        # k=5: M6
        {
            'A': [(1, 2), (-1, 0)],# A10 - A00
            'B': [(1, 0), (1, 1)], # B00 + B01
            'C': [(1, 3)]          # C11
        },
        # k=6: M7
        {
            'A': [(1, 1), (-1, 3)],# A01 - A11
            'B': [(1, 2), (1, 3)], # B10 + B11
            'C': [(1, 0)]          # C00
        }
    ]

    # Helper to map block index to global indices
    # Block 0: rows 0-1, cols 0-1
    # Block 1: rows 0-1, cols 2-3
    # Block 2: rows 2-3, cols 0-1
    # Block 3: rows 2-3, cols 2-3
    
    def get_global_indices(block_idx, dim=4):
        r_start = (block_idx // 2) * 2
        c_start = (block_idx % 2) * 2
        indices = []
        for r in range(2):
            for c in range(2):
                indices.append((r_start + r, c_start + c))
        return indices

    # Iterate over the two halves of the 8-column output
    for half in range(2):
        for k_idx, step in enumerate(steps):
            for m_s in range(7): # Inner Strassen index
                col = half * 49 + k_idx * 7 + m_s
                
                # A part
                for coeff_a, block_idx in step['A']:
                    for (r, c) in get_global_indices(block_idx):
                        global_a_idx = r * 4 + c
                        # Local index in 2x2 block
                        local_r = r % 2
                        local_c = c % 2
                        local_a_idx = local_r * 2 + local_c
                        U[global_a_idx, col] += coeff_a * U_2[local_a_idx, m_s]
                
                # B part
                for coeff_b, block_idx in step['B']:
                    for (r, c) in get_global_indices(block_idx):
                        global_b_idx = r * 8 + (c + half * 4)
                        local_r = r % 2
                        local_c = c % 2
                        local_b_idx = local_r * 2 + local_c
                        V[global_b_idx, col] += coeff_b * V_2[local_b_idx, m_s]
                
                # C part (W)
                for coeff_c, block_idx in step['C']:
                    for (r, c) in get_global_indices(block_idx):
                        global_w_idx = (c + half * 4) * 4 + r
                        local_r = r % 2
                        local_c = c % 2
                        local_w_idx = local_r * 2 + local_c
                        W[global_w_idx, col] += coeff_c * W_2[local_w_idx, m_s]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
