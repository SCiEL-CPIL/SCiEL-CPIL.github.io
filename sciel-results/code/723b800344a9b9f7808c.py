import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s limit
    
    # Schoolbook baseline
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if total != 1:
                        return False
        return True
    
    # Try Strassen-based decomposition for 4x4x4 + schoolbook for 4x4x3
    # Split: A is 4x4, B is 4x7, C = A*B is 4x7
    # Split B into B1 (4x4) and B2 (4x3)
    # C1 = A*B1 (4x4) uses Strassen rank 49
    # C2 = A*B2 (4x3) uses schoolbook rank 48
    # Total rank: 97
    
    def build_strassen_4x4x4():
        """Build Strassen decomposition for 4x4x4 matrix multiplication.
        Uses 2x2 block Strassen with 7 multiplications.
        Each 2x2x2 Strassen block has rank 7 (scalar level).
        For 4x4x4: split into 4 blocks of 2x2x2.
        Actually, for 4x4x4, we can use the standard Strassen recursion:
        A = [[A11, A12], [A21, A22]] (each 2x2)
        B = [[B11, B12], [B21, C12]] (each 2x2)
        C = A*B = [[A11*B11 + A12*B21, A11*B12 + A12*B22],
                   [A21*B11 + A22*B21, A21*B12 + A22*B22]]
        
        Strassen:
        M1 = (A11+A22)*(B11+B22)
        M2 = (A21+A22)*B11
        M3 = A11*(B12-B22)
        M4 = A22*(B21-B11)
        M5 = (A11+A12)*B22
        M6 = (A21-A11)*(B11+B12)
        M7 = (A12-A22)*(B21+B22)
        
        C11 = M1 + M4 - M2 + M6
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M3 + M5 - M7
        
        Each Mi is a 2x2x2 tensor (multiplying two 2x2 matrices).
        For 2x2x2, Strassen has rank 7.
        So total rank for 4x4x4 = 7 * 7 = 49.
        """
        # For each of the 7 Strassen multiplications, we have a 2x2x2 subproblem.
        # For 2x2x2 Strassen:
        # M = (a11+a22)*(b11+b22)
        # M2 = (a21+a22)*b11
        # M3 = a11*(b12-b22)
        # M4 = a22*(b21-b11)
        # M5 = (a11+a12)*b22
        # M6 = (a21-a11)*(b11+b12)
        # M7 = (a12-a22)*(b21+b22)
        # c11 = M1 + M4 - M2 + M6
        # c12 = M3 + M5
        # c21 = M2 + M4
        # c22 = M1 - M3 + M5 - M7
        
        # U entries for 2x2x2 Strassen (rows: a11,a12,a21,a22 -> indices 0,1,2,3)
        # V entries (rows: b11,b12,b21,b22 -> indices 0,1,2,3)
        # W entries (rows: c11,c12,c21,c22 -> indices 0,1,2,3)
        
        strassen_2x2x2 = {
            'U': [
                [1, 0, 1, 0, 1, 0, 0],   # a11
                [0, 1, 0, 0, 1, 0, 0],   # a12
                [0, 0, 1, 1, 0, 1, 0],   # a21
                [1, 0, 0, 1, 0, 0, 1],   # a22
            ],
            'V': [
                [1, 1, 1, 0, 0, 1, 0],   # b11
                [1, 0, 0, 1, 0, 1, 1],   # b12
                [1, 1, -1, 1, 1, 0, 1],  # b21
                [1, 0, -1, 0, 1, 0, 1],  # b22
            ],
            'W': [
                [1, 0, 0, 0, 0, 1, 0],   # c11
                [0, 0, 1, 0, 1, 0, 0],   # c12
                [0, 1, 0, 1, 0, 0, 0],   # c21
                [1, 0, -1, 0, 1, 0, -1], # c22
            ]
        }
        
        # Now build 4x4x4 from 7 copies of 2x2x2 Strassen
        # The 7 Strassen terms at the 4x4 level:
        # M1 = (A11+A22)*(B11+B22) -> 2x2x2 subproblem
        # M2 = (A21+A22)*B11 -> 2x2x2 subproblem  
        # M3 = A11*(B12-B22) -> 2x2x2 subproblem
        # M4 = A22*(B21-B11) -> 2x2x2 subproblem
        # M5 = (A11+A12)*B22 -> 2x2x2 subproblem
        # M6 = (A21-A11)*(B11+B12) -> 2x2x2 subproblem
        # M7 = (A12-A22)*(B21+B22) -> 2x2x2 subproblem
        
        # Each 2x2x2 subproblem uses 7 scalar multiplications.
        # Total: 7 * 7 = 49 rank terms.
        
        # For each Strassen term t in 0..6, and each 2x2x2 Strassen sub-term s in 0..6:
        # The U contribution: combine the 4x4-level linear combination with the 2x2x2-level
        # The V contribution: similarly
        # The W contribution: similarly
        
        # 4x4 index mapping:
        # A[i,j] where i,j in 0..3
        # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4]
        # A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
        # So A[i,j]: if i<2 and j<2: A11[i,j]; if i<2 and j>=2: A12[i,j-2]
        #            if i>=2 and j<2: A21[i-2,j]; if i>=2 and j>=2: A22[i-2,j-2]
        
        # Similarly for B (4x4 in this subproblem):
        # B[k,j] where k,j in 0..3
        # B11 = B[0:2, 0:2], B12 = B[0:2, 2:4]
        # B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
        
        # C[i,j]:
        # C11 = C[0:2, 0:2], C12 = C[0:2, 2:4]
        # C21 = C[2:4, 0:2], C22 = C[2:4, 2:4]
        
        # For each of the 7 top-level Strassen terms, define:
        # - Which 2x2 blocks of A are involved and with what coefficients
        # - Which 2x2 blocks of B are involved and with what coefficients
        # - How the resulting 2x2 product contributes to which 2x2 blocks of C
        
        # Term definitions: (A_blocks, B_blocks, C_blocks)
        # Each entry: list of (block_index, coefficient)
        # Block indices: 0=A11/B11/C11, 1=A12/B12/C12, 2=A21/B21/C21, 3=A22/B22/C22
        
        top_strassen = [
            # M1 = (A11+A22)*(B11+B22)
            {'A': [(0, 1), (3, 1)], 'B': [(0, 1), (3, 1)], 'C': [(0, 1), (3, 1)]},
            # M2 = (A21+A22)*B11
            {'A': [(2, 1), (3, 1)], 'B': [(0, 1)], 'C': [(2, 1), (3, 1)]},
            # M3 = A11*(B12-B22)
            {'A': [(0, 1)], 'B': [(1, 1), (3, -1)], 'C': [(1, 1)]},
            # M4 = A22*(B21-B11)
            {'A': [(3, 1)], 'B': [(2, 1), (0, -1)], 'C': [(0, 1), (2, 1)]},
            # M5 = (A11+A12)*B22
            {'A': [(0, 1), (1, 1)], 'B': [(3, 1)], 'C': [(1, 1), (3, 1)]},
            # M6 = (A21-A11)*(B11+B12)
            {'A': [(2, 1), (0, -1)], 'B': [(0, 1), (1, 1)], 'C': [(0, 1)]},
            # M7 = (A12-A22)*(B21+B22)
            {'A': [(1, 1), (3, -1)], 'B': [(2, 1), (3, 1)], 'C': [(3, -1)]},
        ]
        
        # Build U, V, W for 4x4x4
        rank_4x4x4 = 49
        U_4x4x4 = [[0] * rank_4x4x4 for _ in range(16)]  # 4*4
        V_4x4x4 = [[0] * rank_4x4x4 for _ in range(16)]  # 4*4
        W_4x4x4 = [[0] * rank_4x4x4 for _ in range(16)]  # 4*4
        
        for t in range(7):  # top-level Strassen term
            term = top_strassen[t]
            for s in range(7):  # 2x2x2 Strassen sub-term
                col = t * 7 + s
                # U: for each row i in 0..3 of A
                for i in range(4):
                    for j in range(4):
                        a_idx = i * 4 + j
                        # Determine which 2x2 block this (i,j) belongs to
                        if i < 2 and j < 2:
                            block = 0
                            li, lj = i, j
                        elif i < 2 and j >= 2:
                            block = 1
                            li, lj = i, j - 2
                        elif i >= 2 and j < 2:
                            block = 2
                            li, lj = i - 2, j
                        else:
                            block = 3
                            li, lj = i - 2, j - 2
                        
                        # Get coefficient from top-level A combination
                        coeff_top = 0
                        for blk, cf in term['A']:
                            if blk == block:
                                coeff_top += cf
                        
                        # Get coefficient from 2x2x2 Strassen U
                        # U_2x2x2[li*2+lj][s]
                        coeff_sub = strassen_2x2x2['U'][li * 2 + lj][s]
                        
                        U_4x4x4[a_idx][col] = coeff_top * coeff_sub
                
                # V: for each row k in 0..3 of B
                for k in range(4):
                    for j in range(4):
                        b_idx = k * 4 + j
                        if k < 2 and j < 2:
                            block = 0
                            lk, lj = k, j
                        elif k < 2 and j >= 2:
                            block = 1
                            lk, lj = k, j - 2
                        elif k >= 2 and j < 2:
                            block = 2
                            lk, lj = k - 2, j
                        else:
                            block = 3
                            lk, lj = k - 2, j - 2
                        
                        coeff_top = 0
                        for blk, cf in term['B']:
                            if blk == block:
                                coeff_top += cf
                        
                        coeff_sub = strassen_2x2x2['V'][lk * 2 + lj][s]
                        
                        V_4x4x4[b_idx][col] = coeff_top * coeff_sub
                
                # W: for each row i in 0..3 of C (output)
                for i in range(4):
                    for j in range(4):
                        c_idx = i * 4 + j
                        if i < 2 and j < 2:
                            block = 0
                            li, lj = i, j
                        elif i < 2 and j >= 2:
                            block = 1
                            li, lj = i, j - 2
                        elif i >= 2 and j < 2:
                            block = 2
                            li, lj = i - 2, j
                        else:
                            block = 3
                            li, lj = i - 2, j - 2
                        
                        coeff_top = 0
                        for blk, cf in term['C']:
                            if blk == block:
                                coeff_top += cf
                        
                        coeff_sub = strassen_2x2x2['W'][li * 2 + lj][s]
                        
                        W_4x4x4[c_idx][col] = coeff_top * coeff_sub
        
        return U_4x4x4, V_4x4x4, W_4x4x4
    
    # Build the full 4x4x7 decomposition
    # Split B into B1 (cols 0-3, 4x4) and B2 (cols 4-6, 4x3)
    # C = A*B1 + A*B2
    # C1 = A*B1: 4x4x4, rank 49
    # C2 = A*B2: 4x4x3, rank 48 (schoolbook)
    # Total rank: 97
    
    total_rank = 49 + 48
    
    U_full = [[0] * total_rank for _ in range(16)]  # n*m = 4*4
    V_full = [[0] * total_rank for _ in range(28)]  # m*p = 4*7
    W_full = [[0] * total_rank for _ in range(28)]  # p*n = 7*4
    
    # Part 1: 4x4x4 Strassen for B1 (columns 0-3)
    U_4x4x4, V_4x4x4, W_4x4x4 = build_strassen_4x4x4()
    
    for i in range(16):
        for c in range(49):
            U_full[i][c] = U_4x4x4[i][c]
    
    for k in range(4):
        for j in range(4):
            v_idx = k * 7 + j  # B[k, j] where j in 0..3 maps to V_full[k*7+j]
            for c in range(49):
                V_full[v_idx][c] = V_4x4x4[k * 4 + j][c]
    
    for k in range(4):
        for i in range(4):
            w_idx = k * 4 + i  # C[k, i] where k in 0..3 maps to W_full[k*4+i]
            for c in range(49):
                W_full[w_idx][c] = W_4x4x4[k * 4 + i][c]
    
    # Part 2: 4x4x3 schoolbook for B2 (columns 4-6)
    offset = 49
    for i in range(4):
        for j in range(4):
            for k in range(3):
                col = offset + i * 4 * 3 + j * 3 + k
                # A[i,j]
                U_full[i * 4 + j][col] = 1
                # B[j, 4+k] -> V_full[j*7 + (4+k)]
                V_full[j * 7 + 4 + k][col] = 1
                # C[4+k, i] -> W_full[(4+k)*4 + i]
                W_full[(4 + k) * 4 + i][col] = 1
    
    # Verify
    if verify(U_full, V_full, W_full):
        return {"U": U_full, "V": V_full, "W": W_full}
    
    # Fallback to schoolbook
    if verify(U_sb, V_sb, W_sb):
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}

# EVOLVE-BLOCK-END
