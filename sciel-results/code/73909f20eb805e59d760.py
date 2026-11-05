import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strassen decomposition for 6x6 using 2x2 blocks of 3x3
    # Outer rank 7, inner rank 27 (schoolbook for 3x3)
    # Total rank 189
    
    # Block definitions
    # A blocks: (0,0), (0,1), (1,0), (1,1)
    # B blocks: (0,0), (0,1), (1,0), (1,1)
    # C blocks: (0,0), (0,1), (1,0), (1,1)
    
    # Strassen steps: (L_blocks, R_blocks, C_targets_with_signs)
    # L_blocks: list of (block_row, block_col, coeff)
    # R_blocks: list of (block_row, block_col, coeff)
    # C_targets: list of (block_row, block_col, coeff)
    
    steps = [
        # M1
        {
            'L': [((0, 0), 1), ((1, 1), 1)],
            'R': [((0, 0), 1), ((1, 1), 1)],
            'C': [((0, 0), 1), ((1, 1), 1)]
        },
        # M2
        {
            'L': [((1, 0), 1), ((1, 1), 1)],
            'R': [((0, 0), 1)],
            'C': [((1, 0), 1), ((1, 1), -1)]
        },
        # M3
        {
            'L': [((0, 0), 1)],
            'R': [((0, 1), 1), ((1, 1), -1)],
            'C': [((0, 1), 1), ((1, 1), 1)]
        },
        # M4
        {
            'L': [((1, 1), 1)],
            'R': [((1, 0), 1), ((0, 0), -1)],
            'C': [((0, 0), 1), ((1, 0), 1)]
        },
        # M5
        {
            'L': [((0, 0), 1), ((0, 1), 1)],
            'R': [((1, 1), 1)],
            'C': [((0, 0), -1), ((0, 1), 1)]
        },
        # M6
        {
            'L': [((1, 0), 1), ((0, 0), -1)],
            'R': [((0, 0), 1), ((0, 1), 1)],
            'C': [((1, 1), 1)]
        },
        # M7
        {
            'L': [((0, 1), 1), ((1, 1), -1)],
            'R': [((1, 0), 1), ((1, 1), 1)],
            'C': [((0, 0), 1)]
        }
    ]
    
    rank = 0
    U_list = []
    V_list = []
    W_list = []
    
    block_size = 3
    dim = 6
    
    for step in steps:
        L_blocks = step['L']
        R_blocks = step['R']
        C_targets = step['C']
        
        # Iterate over local indices of the 3x3 blocks
        for i in range(block_size):
            for j in range(block_size):
                for k in range(block_size):
                    # For each pair of contributing A and B blocks
                    for (br_L, bc_L), coeff_L in L_blocks:
                        # Global index for A
                        a_row = br_L * block_size + i
                        a_col = bc_L * block_size + j
                        a_idx = a_row * dim + a_col
                        
                        for (br_R, bc_R), coeff_R in R_blocks:
                            # Global index for B
                            b_row = br_R * block_size + j
                            b_col = bc_R * block_size + k
                            b_idx = b_row * dim + b_col
                            
                            # Create U, V, W vectors
                            U = [0] * (dim * dim)
                            V = [0] * (dim * dim)
                            W = [0] * (dim * dim)
                            
                            U[a_idx] = coeff_L
                            V[b_idx] = coeff_R
                            
                            # Determine W entries
                            for (br_C, bc_C), coeff_C in C_targets:
                                c_row = br_C * block_size + i
                                c_col = bc_C * block_size + k
                                # Trace convention: W[k*n + i]
                                # Here n=p=6. c_col is k, c_row is i.
                                c_idx = c_col * dim + c_row
                                W[c_idx] += coeff_L * coeff_R * coeff_C
                            
                            U_list.append(U)
                            V_list.append(V)
                            W_list.append(W)
                            rank += 1
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
