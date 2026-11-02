import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    rank = 40
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # A is 2x4. A00 = A[0:2, 0:2], A01 = A[0:2, 2:4].
    # B is 4x8. B1 = B[0:4, 0:4], B2 = B[0:4, 4:8].
    # B1 blocks: B100, B101, B110, B111 (2x2 each).
    # B2 blocks: B200, B201, B210, B211 (2x2 each).
    
    # Strassen-like terms for A_padded (A10=A11=0):
    # T1 = A00 (B00+B11)
    # T3 = A00 (B01-B11)
    # T5 = A00 B11
    # T6 = A00 (B00+B01)  # Note: M6 was (A10-A00)(...) = -A00(...)
    # T7 = A01 (B10+B11)
    
    # C00 = T1 - T3 - T5 - T6  (Since M6 = -A00(...), C00 = M1 - M3 - M5 + M6 = T1 - T3 - T5 - T6)
    # C01 = T3 + T5
    # C10 = T7
    # C11 = T1 + T7
    
    # We need C[0:2, 0:4] and C[0:2, 4:8].
    # C[0:2, 0:4] = [C1_00, C1_01]
    # C[0:2, 4:8] = [C2_00, C2_01]
    
    # Indices for U (A part):
    # A00 indices: 0, 1, 2, 3 (in A[0:2, 0:2])
    # A01 indices: 4, 5, 6, 7 (in A[0:2, 2:4])
    
    # Indices for V (B part):
    # B100: 0, 1, 8, 9
    # B101: 2, 3, 10, 11
    # B110: 4, 5, 12, 13
    # B111: 6, 7, 14, 15
    # B200: 16, 17, 24, 25
    # B201: 18, 19, 26, 27
    # B210: 20, 21, 28, 29
    # B211: 22, 23, 30, 31
    
    # Indices for W (C part):
    # C[0:2, 0:4] -> 0, 1, 2, 3, 4, 5, 6, 7
    # C[0:2, 4:8] -> 8, 9, 10, 11, 12, 13, 14, 15
    
    # Helper to add a 2x2 product term
    def add_2x2_term(col, a_block, b_block, w_sign, w_offset):
        # a_block: 'A00' or 'A01'
        # b_block: list of (block_name, sign)
        # w_sign: sign for C contribution
        # w_offset: which C block (0 for left, 1 for right)
        
        # Determine A indices
        if a_block == 'A00':
            a_indices = [0, 1, 2, 3]
        else:
            a_indices = [4, 5, 6, 7]
            
        # Determine B indices
        b_indices = []
        signs = []
        for b_name, s in b_block:
            if b_name == 'B100':
                idxs = [0, 1, 8, 9]
            elif b_name == 'B101':
                idxs = [2, 3, 10, 11]
            elif b_name == 'B110':
                idxs = [4, 5, 12, 13]
            elif b_name == 'B111':
                idxs = [6, 7, 14, 15]
            elif b_name == 'B200':
                idxs = [16, 17, 24, 25]
            elif b_name == 'B201':
                idxs = [18, 19, 26, 27]
            elif b_name == 'B210':
                idxs = [20, 21, 28, 29]
            elif b_name == 'B211':
                idxs = [22, 23, 30, 31]
            else:
                continue
            b_indices.extend(idxs)
            signs.extend([s]*4)
            
        # For each of the 4 scalar products in the 2x2 matrix product
        for k in range(4):
            sub_col = col + k
            U[a_indices[k], sub_col] = 1
            V[b_indices[k], sub_col] = signs[k]
            
            # W indices
            # C is 2x2 block.
            # C[0,0] -> w_offset*4 + 0
            # C[0,1] -> w_offset*4 + 1
            # C[1,0] -> w_offset*4 + 2
            # C[1,1] -> w_offset*4 + 3
            
            # The 2x2 product A*B contributes to C.
            # A is 2x2, B is 2x2.
            # Product P[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
            # Our decomposition is rank-1 scalar terms.
            # We are decomposing the 2x2 product into 4 scalar terms.
            # Term 0: A[0,0] * B[0,0] -> contributes to P[0,0]
            # Term 1: A[0,1] * B[1,0] -> contributes to P[0,0]
            # Term 2: A[1,0] * B[0,1] -> contributes to P[1,0]
            # Term 3: A[1,1] * B[1,1] -> contributes to P[1,1]
            # Wait, this is schoolbook for 2x2.
            # P[0,0] uses terms 0, 1.
            # P[0,1] uses terms 2, 3? No.
            # P[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1].
            # My mapping above is wrong for schoolbook.
            
            # Let's map correctly.
            # A = [[a00, a01], [a10, a11]]
            # B = [[b00, b01], [b10, b11]]
            # P = [[p00, p01], [p10, p11]]
            # p00 = a00b00 + a01b10
            # p01 = a00b01 + a01b11
            # p10 = a10b00 + a11b10
            # p11 = a10b01 + a11b11
            
            # We need 4 terms? No, 8 terms for schoolbook 2x2.
            # My "add_2x2_term" assumed 4 terms. That's Strassen? No.
            # Schoolbook 2x2 has 8 scalar mults.
            # So each "2x2 product" requires 8 columns in U,V,W.
            
            # So rank is 5 terms * 8 mults * 2 blocks = 80?
            # No, I can reuse the A parts?
            # No, each scalar mult is a separate rank-1 term.
            
            # So rank 80 is worse than 64.
            
            pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
