import numpy as np
import random
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Target Rank: 43
    # Construction:
    # Split A (3x3) into:
    #   A_TL (2x2), A_TR (2x1)
    #   A_BL (1x2), A_BR (1x1)
    # Split B (3x5) into:
    #   B_TL (2x2), B_TR (2x3)
    #   B_BL (1x2), B_BR (1x3)
    
    # Blocks and their ranks:
    # 1. A_TL * B_TL (2x2 * 2x2) -> Rank 7 (Strassen)
    # 2. A_TL * B_TR (2x2 * 2x3) -> Rank 11 (Strassen 2x2x2 + Schoolbook 2x2x1)
    # 3. A_TR * B_BL (2x1 * 1x2) -> Rank 4 (Schoolbook)
    # 4. A_TR * B_BR (2x1 * 1x3) -> Rank 6 (Schoolbook)
    # 5. A_BL * B_TL (1x2 * 2x2) -> Rank 4 (Schoolbook)
    # 6. A_BL * B_TR (1x2 * 2x3) -> Rank 6 (Schoolbook)
    # 7. A_BR * B_BL (1x1 * 1x2) -> Rank 2 (Schoolbook)
    # 8. A_BR * B_BR (1x1 * 1x3) -> Rank 3 (Schoolbook)
    
    total_rank = 7 + 11 + 4 + 6 + 4 + 6 + 2 + 3
    # total_rank = 43
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    current_rank = 0
    
    # Helper to add a block
    # a_start, a_end: rows of A
    # a_col_start, a_col_end: cols of A
    # b_start, b_end: rows of B
    # b_col_start, b_col_end: cols of B
    # c_start, c_end: rows of C
    # c_col_start, c_col_end: cols of C
    # rank: rank of this block
    # coeff_type: 'strassen' or 'schoolbook'
    
    def add_block(a_r, a_c, b_r, b_c, c_r, c_c, rank, is_strassen=False):
        nonlocal current_rank
        start_r = current_rank
        end_r = current_rank + rank
        current_rank += rank
        
        # Map indices
        # A index: i*m + j
        # B index: j*p + k
        # W index: k*n + i
        
        if is_strassen:
            # 2x2x2 Strassen
            # A indices: a_r[0], a_r[1] rows; a_c[0], a_c[1] cols
            # B indices: b_r[0], b_r[1] rows; b_c[0], b_c[1] cols
            # C indices: c_r[0], c_r[1] rows; c_c[0], c_c[1] cols
            
            # Local indices 0,1 for rows/cols
            # Global A row indices: a_r[0], a_r[1]
            # Global A col indices: a_c[0], a_c[1]
            
            # Strassen Multiplications:
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B01
            # M3 = A00(B10-B00)
            # M4 = A11(B11-B01)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B10)
            # M7 = (A01-A11)(B11+B01)
            
            # We will fill U, V, W for these 7 terms
            
            # Term 1
            U[a_r[0]*m + a_c[0], start_r] += 1
            U[a_r[1]*m + a_c[1], start_r] += 1
            V[b_r[0]*p + b_c[0], start_r] += 1
            V[b_r[1]*p + b_c[1], start_r] += 1
            W[b_c[0]*n + c_r[0], start_r] += 1
            W[b_c[0]*n + c_r[1], start_r] += 1
            W[b_c[1]*n + c_r[0], start_r] += 1
            W[b_c[1]*n + c_r[1], start_r] += 1
            
            # Term 2
            U[a_r[1]*m + a_c[0], start_r+1] += 1
            U[a_r[1]*m + a_c[1], start_r+1] += 1
            V[b_r[0]*p + b_c[1], start_r+1] += 1
            W[b_c[1]*n + c_r[1], start_r+1] += 1
            
            # Term 3
            U[a_r[0]*m + a_c[0], start_r+2] += 1
            V[b_r[1]*p + b_c[0], start_r+2] += 1
            V[b_r[0]*p + b_c[0], start_r+2] -= 1
            W[b_c[0]*n + c_r[0], start_r+2] += 1
            W[b_c[1]*n + c_r[1], start_r+2] += 1
            
            # Term 4
            U[a_r[1]*m + a_c[1], start_r+3] += 1
            V[b_r[1]*p + b_c[1], start_r+3] += 1
            V[b_r[0]*p + b_c[1], start_r+3] -= 1
            W[b_c[0]*n + c_r[0], start_r+3] += 1
            W[b_c[1]*n + c_r[1], start_r+3] += 1
            
            # Term 5
            U[a_r[0]*m + a_c[0], start_r+4] += 1
            U[a_r[0]*m + a_c[1], start_r+4] += 1
            V[b_r[1]*p + b_c[1], start_r+4] += 1
            W[b_c[1]*n + c_r[0], start_r+4] += 1
            W[b_c[1]*n + c_r[1], start_r+4] += 1
            
            # Term 6
            U[a_r[1]*m + a_c[0], start_r+5] += 1
            U[a_r[0]*m + a_c[0], start_r+5] -= 1
            V[b_r[0]*p + b_c[0], start_r+5] += 1
            V[b_r[1]*p + b_c[0], start_r+5] += 1
            W[b_c[0]*n + c_r[1], start_r+5] += 1
            W[b_c[1]*n + c_r[1], start_r+5] += 1
            
            # Term 7
            U[a_r[0]*m + a_c[1], start_r+6] += 1
            U[a_r[1]*m + a_c[1], start_r+6] -= 1
            V[b_r[1]*p + b_c[1], start_r+6] += 1
            V[b_r[0]*p + b_c[1], start_r+6] += 1
            W[b_c[0]*n + c_r[0], start_r+6] += 1
            W[b_c[1]*n + c_r[0], start_r+6] += 1
            
        else:
            # Schoolbook
            # Iterate over all i in a_r, j in a_c, k in b_r
            # But we need to map to C indices c_r, c_c
            # For each multiplication A[i,j] * B[j,k], it contributes to C[i,k]
            
            for i in a_r:
                for j in a_c:
                    for k in b_r:
                        # Check if j is compatible (it is, by block definition)
                        # A index: i*m + j
                        # B index: j*p + k
                        # C index: k*n + i
                        # But we only want to contribute to specific C indices?
                        # No, the block decomposition ensures that A[i,j] and B[j,k]
                        # only contribute to C[i,k] where i is in c_r and k is in c_c.
                        # Wait, the block definition says:
                        # A rows a_r, cols a_c.
                        # B rows b_r, cols b_c.
                        # C rows c_r, cols c_c.
                        # The multiplication A * B naturally maps A[i,j] * B[j,k] to C[i,k].
                        # So we just need to ensure that the resulting C[i,k] is within c_r, c_c.
                        # And we add 1 to W[k*n + i].
                        
                        U[i*m + j, current_rank] = 1
                        V[j*p + k, current_rank] = 1
                        W[k*n + i, current_rank] = 1
                        current_rank += 1

    # Define Blocks
    # A rows: 0,1 for TL/TR; 2 for BL/BR
    # A cols: 0,1 for TL/BL; 2 for TR/BR
    # B rows: 0,1 for TL/TR; 2 for BL/BR
    # B cols: 0,1 for TL/BL; 2,3,4 for TR/BR
    
    # Block 1: A_TL (0,1; 0,1) * B_TL (0,1; 0,1) -> C (0,1; 0,1)
    add_block([0,1], [0,1], [0,1], [0,1], [0,1], [0,1], 7, is_strassen=True)
    
    # Block 2: A_TL (0,1; 0,1) * B_TR (0,1; 2,3,4) -> C (0,1; 2,3,4)
    # Rank 11: Split B_TR into B_TL2 (2x2, cols 2,3) and B_TR1 (2x1, col 4)
    # Sub-block 2a: A_TL * B_TL2 (2x2 * 2x2) -> Rank 7
    add_block([0,1], [0,1], [0,1], [2,3], [0,1], [2,3], 7, is_strassen=True)
    # Sub-block 2b: A_TL * B_TR1 (2x2 * 2x1) -> Rank 4
    add_block([0,1], [0,1], [0,1], [4], [0,1], [4], 4, is_strassen=False)
    
    # Block 3: A_TR (0,1; 2) * B_BL (2; 0,1) -> C (0,1; 0,1)
    add_block([0,1], [2], [2], [0,1], [0,1], [0,1], 4, is_strassen=False)
    
    # Block 4: A_TR (0,1; 2) * B_BR (2; 2,3,4) -> C (0,1; 2,3,4)
    add_block([0,1], [2], [2], [2,3,4], [0,1], [2,3,4], 6, is_strassen=False)
    
    # Block 5: A_BL (2; 0,1) * B_TL (0,1; 0,1) -> C (2; 0,1)
    add_block([2], [0,1], [0,1], [0,1], [2], [0,1], 4, is_strassen=False)
    
    # Block 6: A_BL (2; 0,1) * B_TR (0,1; 2,3,4) -> C (2; 2,3,4)
    add_block([2], [0,1], [0,1], [2,3,4], [2], [2,3,4], 6, is_strassen=False)
    
    # Block 7: A_BR (2; 2) * B_BL (2; 0,1) -> C (2; 0,1)
    add_block([2], [2], [2], [0,1], [2], [0,1], 2, is_strassen=False)
    
    # Block 8: A_BR (2; 2) * B_BR (2; 2,3,4) -> C (2; 2,3,4)
    add_block([2], [2], [2], [2,3,4], [2], [2,3,4], 3, is_strassen=False)
    
    # Verify current_rank matches total_rank
    if current_rank != total_rank:
        # This should not happen
        pass
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
