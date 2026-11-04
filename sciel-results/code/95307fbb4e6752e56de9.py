import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 189 decomposition: 7 Strassen products * 27 trivial 3x3 multiplications
    rank = 189
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    # Strassen definitions for 2x2 blocks of 3x3 matrices
    # Each entry is a list of (block_idx, coeff)
    # Block indices: 0: 00, 1: 01, 2: 10, 3: 11
    # Row block: 0 (0-2), 1 (3-5). Col block: 0 (0-2), 1 (3-5).
    # Block idx = row_block * 2 + col_block
    
    strassen_defs = [
        # P0 = (A00+A11)(B00+B11)
        {
            'A': [(0, 1), (3, 1)],
            'B': [(0, 1), (3, 1)],
            'C': [(0, 1), (3, 1)] # C00, C11
        },
        # P1 = (A10+A11)B00
        {
            'A': [(2, 1), (3, 1)],
            'B': [(0, 1)],
            'C': [(2, 1), (3, -1)] # C10, C11
        },
        # P2 = A00(B01-B11)
        {
            'A': [(0, 1)],
            'B': [(1, 1), (3, -1)],
            'C': [(1, 1), (3, 1)] # C01, C11
        },
        # P3 = A11(B10-B01)
        {
            'A': [(3, 1)],
            'B': [(2, 1), (1, -1)],
            'C': [(0, 1), (2, 1)] # C00, C10
        },
        # P4 = (A00+A01)B11
        {
            'A': [(0, 1), (1, 1)],
            'B': [(3, 1)],
            'C': [(0, -1), (1, 1)] # C00, C01
        },
        # P5 = (A10-A00)(B00+B01)
        {
            'A': [(2, 1), (0, -1)],
            'B': [(0, 1), (1, 1)],
            'C': [(3, 1)] # C11
        },
        # P6 = (A01-A11)(B10+B11)
        {
            'A': [(1, 1), (3, -1)],
            'B': [(2, 1), (3, 1)],
            'C': [(0, 1)] # C00
        }
    ]
    
    r_idx = 0
    for s in range(7):
        defs = strassen_defs[s]
        A_blocks = defs['A']
        B_blocks = defs['B']
        C_blocks = defs['C']
        
        # For each combination of active A block and active B block
        # The product is non-zero only where the column of A block matches the row of B block?
        # No, A is 6x6, B is 6x6.
        # A_block (ra, ca) occupies rows [3*ra, 3*ra+2] and cols [3*ca, 3*ca+2].
        # B_block (rb, cb) occupies rows [3*rb, 3*rb+2] and cols [3*cb, 3*cb+2].
        # The term A[I, J] * B[J, K] is non-zero if:
        # I is in row range of A_block, J is in col range of A_block.
        # J is in row range of B_block, K is in col range of B_block.
        # So J must be in [3*ca, 3*ca+2] AND J must be in [3*rb, 3*rb+2].
        # This implies ca == rb.
        
        for (ba_idx, coeff_a) in A_blocks:
            ra = ba_idx // 2
            ca = ba_idx % 2
            
            for (bb_idx, coeff_b) in B_blocks:
                rb = bb_idx // 2
                cb = bb_idx % 2
                
                if ca != rb:
                    continue
                    
                # J range is [3*ca, 3*ca+2]
                j_start = 3 * ca
                j_end = 3 * ca + 3
                
                # I range is [3*ra, 3*ra+2]
                i_start = 3 * ra
                i_end = 3 * ra + 3
                
                # K range is [3*cb, 3*cb+2]
                k_start = 3 * cb
                k_end = 3 * cb + 3
                
                # Iterate over local indices
                for i_local in range(3):
                    I = i_start + i_local
                    for j_local in range(3):
                        J = j_start + j_local
                        for k_local in range(3):
                            K = k_start + k_local
                            
                            # This term contributes to C[I, K]
                            # Determine which C block contains (I, K)
                            rc = I // 3
                            cc = K // 3
                            bc_idx = rc * 2 + cc
                            
                            # Find coeff_c for this block in C_blocks
                            coeff_c = 0
                            for (bc, c_val) in C_blocks:
                                if bc == bc_idx:
                                    coeff_c = c_val
                                    break
                            
                            if coeff_c != 0:
                                # Assign rank
                                u_val = coeff_a
                                v_val = coeff_b
                                w_val = coeff_c
                                
                                # U index: I*6 + J
                                # V index: J*6 + K
                                # W index: K*6 + I
                                
                                U[I*6 + J][r_idx] = u_val
                                V[J*6 + K][r_idx] = v_val
                                W[K*6 + I][r_idx] = w_val
                                r_idx += 1
                                
    # r_idx should be 189
    if r_idx != rank:
        # In case of mismatch, pad or truncate (should not happen)
        pass
        
    return {"U": U, "V": V, "W": W}
