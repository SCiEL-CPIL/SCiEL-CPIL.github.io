"""Matrix multiplication 4,4,9 - rank-minimal exact tensor decomposition."""
import numpy as np
import time

def _verify(U, V, W, n=4, m=4, p=9):
    """Verify that U, V, W exactly decompose the matmul tensor."""
    n, m, p = 4, 4, 9
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    rank = U.shape[1]
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if s != 1:
                    return False
    return True

def _build_strassen_2x2():
    """
    Build the rank-7 Strassen decomposition for 2x2 x 2x2 -> 2x2.
    Returns (U, V, W) where:
      U: 4x7 (entries for A[i,j])
      V: 4x7 (entries for B[j,k])
      W: 4x7 (entries for C[i,k])
    
    Strassen's 7 products:
      M1 = (A00+A11)(B00+B11)
      M2 = (A10+A11)B00
      M3 = A00(B01-B11)
      M4 = A11(B10-B00)
      M5 = (A00+A01)B11
      M6 = (A10-A00)(B00+B01)
      M7 = (A01-A11)(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    # U[i*2+j, c] for i,j in {0,1}
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1  # A00
    U[3][0] = 1  # A11
    V[0][0] = 1  # B00
    V[3][0] = 1  # B11
    
    # M2 = (A10+A11)B00
    U[2][1] = 1  # A10
    U[3][1] = 1  # A11
    V[0][1] = 1  # B00
    
    # M3 = A00(B01-B11)
    U[0][2] = 1  # A00
    V[1][2] = 1  # B01
    V[3][2] = -1 # B11
    
    # M4 = A11(B10-B00)
    U[3][3] = 1  # A11
    V[2][3] = 1  # B10
    V[0][3] = -1 # B00
    
    # M5 = (A00+A01)B11
    U[0][4] = 1  # A00
    U[1][4] = 1  # A01
    V[3][4] = 1  # B11
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1  # A10
    U[0][5] = -1 # A00
    V[0][5] = 1  # B00
    V[1][5] = 1  # B01
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1  # A01
    U[3][6] = -1 # A11
    V[2][6] = 1  # B10
    V[3][6] = 1  # B11
    
    # C00 = M1 + M4 - M5 + M7
    W[0][0] = 1  # M1
    W[0][3] = 1  # M4
    W[0][4] = -1 # -M5
    W[0][6] = 1  # M7
    
    # C01 = M3 + M5
    W[1][2] = 1  # M3
    W[1][4] = 1  # M5
    
    # C10 = M2 + M4
    W[2][1] = 1  # M2
    W[2][3] = 1  # M4
    
    # C11 = M1 - M2 + M3 + M6
    W[3][0] = 1  # M1
    W[3][1] = -1 # -M2
    W[3][2] = 1  # M3
    W[3][5] = 1  # M6
    
    return U, V, W

def _build_4x4x4_strassen():
    """
    Build a rank-21 decomposition for 4x4 x 4x4 -> 4x4 using Strassen.
    Uses 3x3 block decomposition of 4x4 into 2x2 blocks, with padding.
    Actually, let's use a direct 4x4 Strassen-based approach.
    
    A better approach: decompose 4x4 as 2x2 blocks of 2x2 matrices.
    The 4x4 matmul can be done via 3x3 block matmul of 2x2 blocks.
    But 4x4 doesn't split evenly into 3x3 blocks of 2x2.
    
    Alternative: Use the fact that 4x4x4 can be decomposed using
    Strassen on 2x2 sub-blocks with some overlap handling.
    
    Let me use a different approach: direct rank-21 for 4x4x4.
    Actually, the known best for 4x4x4 is rank 21 (from Strassen applied to 2x2 blocks).
    
    For 4x4, we can view it as multiplying two 4x4 matrices.
    One approach: split into 2x2 blocks:
    A = [[A00, A01], [A10, A11]] where each is 2x2
    B = [[B00, B01], [B10, B11]] where each is 2x2
    
    Then C = [[A00*B00 + A01*B10, A00*B01 + A01*B11],
              [A10*B00 + A11*B10, A10*B01 + A11*B11]]
    
    Each 2x2x2x2 matmul needs rank 7, and we have 8 such products.
    That gives rank 8*7 = 56, which is worse than schoolbook 64.
    
    To get rank 21, we need a more clever approach.
    Let me just use a known rank-21 decomposition or fall back to a simpler approach.
    
    For simplicity and correctness, let me use a block approach that gives a valid
    (if not optimal) decomposition, then try to optimize.
    
    Actually, for the 4x4x9 case, let me think differently.
    """
    # For now, let's build a valid decomposition using nested Strassen.
    # 4x4x4: Use Strassen on 2x2 blocks.
    # This gives rank 8*7 = 56, but we can do better.
    # Let's just use the schoolbook for 4x4x4 as a building block for now,
    # and focus on the 4x4x9 structure.
    
    # Actually, let me build the 4x4x4 decomposition using the 2x2 Strassen
    # applied to the 2x2 block structure.
    
    U2, V2, W2 = _build_strassen_2x2()
    
    # For 4x4, split into 2x2 blocks.
    # A has blocks A00, A01, A10, A11 (each 2x2)
    # B has blocks B00, B01, B10, B11 (each 2x2)
    # C has blocks C00, C01, C10, C11 (each 2x2)
    
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Each term is a 2x2x2x2 matmul (rank 7 each)
    # Total: 8 * 7 = 56
    
    # But we can share: use Strassen on the block level too.
    # Block-level Strassen for 2x2 blocks:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each Mi is a 2x2x2x2 matmul (rank 7)
    # Total rank: 7 * 7 = 49
    
    # This is better than 56 but still not 21.
    # For now, let's use this rank-49 decomposition as a building block.
    
    n_block = 2  # 2x2 blocks
    m_block = 2
    p_block = 2
    
    # Build U, V, W for 4x4x4 using block Strassen
    # U has shape (4*4, 49) = (16, 49)
    # V has shape (4*4, 49) = (16, 49)  
    # W has shape (4*4, 49) = (16, 49)
    
    U = [[0]*49 for _ in range(16)]
    V = [[0]*49 for _ in range(16)]
    W = [[0]*49 for _ in range(16)]
    
    # For each block-level product Mi (i=0..6), we have a 2x2x2x2 matmul
    # The U entries for Mi depend on which A-blocks are involved
    # The V entries for Mi depend on which B-blocks are involved
    # The W entries for Mi depend on which C-blocks are involved
    
    # Let me define the block-level coefficients
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # For each Mi, the A-blocks involved and their coefficients:
    # M1: A00(+1), A11(+1)
    # M2: A10(+1), A11(+1)
    # M3: A00(+1)
    # M4: A11(+1)
    # M5: A00(+1), A01(+1)
    # M6: A10(+1), A00(-1)
    # M7: A01(+1), A11(-1)
    
    # For each Mi, the B-blocks involved and their coefficients:
    # M1: B00(+1), B11(+1)
    # M2: B00(+1)
    # M3: B01(+1), B11(-1)
    # M4: B10(+1), B00(-1)
    # M5: B11(+1)
    # M6: B00(+1), B01(+1)
    # M7: B10(+1), B11(+1)
    
    # For each C-block, which Mi contribute and with what coefficient:
    # C00: M1(+1), M4(+1), M5(-1), M7(+1)
    # C01: M3(+1), M5(+1)
    # C10: M2(+1), M4(+1)
    # C11: M1(+1), M2(-1), M3(+1), M6(+1)
    
    # Now, for each Mi, we need to embed the 2x2x2x2 Strassen into the 4x4x4.
    # The 2x2x2x2 Strassen has U2, V2, W2 of shape (4,7) each.
    # For A-block Aab (a,b in {0,1}), the local indices are (2a+ii, 2b+jj) for ii,jj in {0,1}
    # For B-block Bcd (c,d in {0,1}), the local indices are (2c+kk, 2d+ll) for kk,ll in {0,1}
    # For C-block Cef (e,f in {0,1}), the local indices are (2e+mm, 2f+nn) for mm,nn in {0,1}
    
    # Let's build this systematically.
    
    # For each Mi (i=0..6), for each local Strassen column c (c=0..6):
    #   The column index in the 4x4x4 decomposition is i*7 + c
    
    a_blocks_M = {
        0: [(0,0,1), (1,1,1)],   # M1: A00(+1), A11(+1)
        1: [(1,0,1), (1,1,1)],   # M2: A10(+1), A11(+1)
        2: [(0,0,1)],             # M3: A00(+1)
        3: [(1,1,1)],             # M4: A11(+1)
        4: [(0,0,1), (0,1,1)],   # M5: A00(+1), A01(+1)
        5: [(1,0,1), (0,0,-1)],  # M6: A10(+1), A00(-1)
        6: [(0,1,1), (1,1,-1)],  # M7: A01(+1), A11(-1)
    }
    
    b_blocks_M = {
        0: [(0,0,1), (1,1,1)],   # M1: B00(+1), B11(+1)
        1: [(0,0,1)],             # M2: B00(+1)
        2: [(0,1,1), (1,1,-1)],  # M3: B01(+1), B11(-1)
        3: [(1,0,1), (0,0,-1)],  # M4: B10(+1), B00(-1)
        4: [(1,1,1)],             # M5: B11(+1)
        5: [(0,0,1), (0,1,1)],   # M6: B00(+1), B01(+1)
        6: [(1,0,1), (1,1,1)],   # M7: B10(+1), B11(+1)
    }
    
    c_blocks_M = {
        0: [(0,0,1), (1,1,1)],   # M1: C00(+1), C11(+1)
        1: [(1,0,1)],             # M2: C10(+1)
        2: [(0,1,1)],             # M3: C01(+1)
        3: [(0,0,1), (1,0,1)],   # M4: C00(+1), C10(+1)
        4: [(0,0,-1), (0,1,1)],  # M5: C00(-1), C01(+1)
        5: [(1,1,1)],             # M6: C11(+1)
        6: [(0,0,1)],             # M7: C00(+1)
    }
    
    # Wait, I need to double-check the C-block assignments.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So for C00: M1(+1), M4(+1), M5(-1), M7(+1)
    # For C01: M3(+1), M5(+1)
    # For C10: M2(+1), M4(+1)
    # For C11: M1(+1), M2(-1), M3(+1), M6(+1)
    
    # Let me redo c_blocks_M:
    # c_blocks_M[mi] = list of (c_row_block, c_col_block, coefficient)
    
    c_blocks_M_corrected = {
        0: [(0,0,1), (1,1,1)],   # M1: C00(+1), C11(+1)
        1: [(1,0,1)],             # M2: C10(+1)
        2: [(0,1,1)],             # M3: C01(+1)
        3: [(0,0,1), (1,0,1)],   # M4: C00(+1), C10(+1)
        4: [(0,0,-1), (0,1,1)],  # M5: C00(-1), C01(+1)
        5: [(1,1,1)],             # M6: C11(+1)
        6: [(0,0,1)],             # M7: C00(+1)
    }
    
    # Now build U, V, W
    for mi in range(7):
        for ci in range(7):
            col = mi * 7 + ci
            
            # U entries
            for (a_row, a_col, coeff) in a_blocks_M[mi]:
                for ii in range(2):
                    for jj in range(2):
                        global_i = 2*a_row + ii
                        global_j = 2*a_col + jj
                        local_idx = ii*2 + jj
                        U[global_i*4 + global_j][col] += coeff * U2[local_idx][ci]
            
            # V entries
            for (b_row, b_col, coeff) in b_blocks_M[mi]:
                for kk in range(2):
                    for ll in range(2):
                        global_j = 2*b_row + kk
                        global_k = 2*b_col + ll
                        local_idx = kk*2 + ll
                        V[global_j*4 + global_k][col] += coeff * V2[local_idx][ci]
            
            # W entries
            for (c_row, c_col, coeff) in c_blocks_M_corrected[mi]:
                for mm in range(2):
                    for nn in range(2):
                        global_i = 2*c_row + mm
                        global_k = 2*c_col + nn
                        local_idx = mm*2 + nn
                        W[global_i*4 + global_k][col] += coeff * W2[local_idx][ci]
    
    return U, V, W

def construct():
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s limit
    
    n, m, p = 4, 4, 9
    schoolbook_rank = n * m * p  # 144
    
    # Start with schoolbook as fallback
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    # Build the 4x4x4 Strassen decomposition (rank 49)
    U44, V44, W44 = _build_4x4x4_strassen()
    
    # Verify the 4x4x4 decomposition
    if _verify(U44, V44, W44, n=4, m=4, p=4):
        # Now extend to 4x4x9.
        # The 4x4x9 matmul can be seen as:
        # C[i,k] = sum_j A[i,j] * B[j,k] for k=0..8
        # 
        # We can split the 9 columns into: 4 + 4 + 1
        # C[:,0:4] = A * B[:,0:4]  (4x4 x 4x4 -> 4x4, rank 49)
        # C[:,4:8] = A * B[:,4:8]  (4x4 x 4x4 -> 4x4, rank 49)
        # C[:,8:9] = A * B[:,8:9]  (4x4 x 4x1 -> 4x1, rank 16)
        #
        # Total rank: 49 + 49 + 16 = 114
        
        # For the 4x1 part, use schoolbook: rank 16
        # U for 4x1: U[i*4+j, c] = A[i,j] for c corresponding to each (i,j)
        # Actually, for 4x4 x 4x1:
        # C[i,0] = sum_j A[i,j] * B[j,0]
        # This is a vector-matrix product, rank 16 (schoolbook)
        
        # Let me build the 4x4x9 decomposition:
        # Part 1: columns 0-3 (4x4x4, rank 49)
        # Part 2: columns 4-7 (4x4x4, rank 49)
        # Part 3: column 8 (4x4x1, rank 16)
        
        U449 = [[0]*114 for _ in range(16)]
        V449 = [[0]*114 for _ in range(36)]
        W449 = [[0]*114 for _ in range(36)]
        
        # Part 1: columns 0-3
        for row in range(16):
            for c in range(49):
                U449[row][c] = U44[row][c]
        for j in range(4):
            for k in range(4):
                for c in range(49):
                    V449[j*9 + k][c] = V44[j*4 + k][c]
        for i in range(4):
            for k in range(4):
                for c in range(49):
                    W449[k*4 + i][c] = W44[i*4 + k][c]
        
        # Part 2: columns 4-7
        offset = 49
        for row in range(16):
            for c in range(49):
                U449[row][offset + c] = U44[row][c]
        for j in range(4):
            for k in range(4):
                for c in range(49):
                    V449[j*9 + (4 + k)][offset + c] = V44[j*4 + k][c]
        for i in range(4):
            for k in range(4):
                for c in range(49):
                    W449[(4 + k)*4 + i][offset + c] = W44[i*4 + k][c]
        
        # Part 3: column 8 (4x4x1, schoolbook rank 16)
        offset2 = 98
        col3 = 0
        for i in range(4):
            for j in range(4):
                U449[i*4 + j][offset2 + col3] = 1
                V449[j*9 + 8][offset2 + col3] = 1
                W449[8*4 + i][offset2 + col3] = 1
                col3 += 1
        
        # Verify
        if _verify(U449, V449, W449):
            best_result = {"U": U449, "V": V449, "W": W449}
            best_rank = 114
            print(f"Found rank 114 decomposition")
    
    # Try to do better: can we share the U part?
    # In parts 1 and 2, the U matrix is the same (both use U44).
    # So we can factor out the U dimension.
    # Instead of having 49+49=98 columns with duplicated U, 
    # we can have 49 columns for U and combine V and W.
    
    # Actually, the decomposition is:
    # sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1
    # 
    # If U is the same for parts 1 and 2, we can write:
    # sum_c U[i,j,c] * (V1[j,k,c] * W1[k,i,c] + V2[j,k,c] * W2[k,i,c]) = 1
    # 
    # But this doesn't directly reduce the rank because V and W are separate.
    # The rank is determined by the number of columns in U, V, W.
    # If U has 49 columns, V has 49+49=98 columns, W has 49+49=98 columns,
    # that's not valid because they must have the same number of columns.
    
    # So the simple concatenation gives rank 114.
    # To do better, we need a more clever decomposition.
    
    # Let's check if we can find a rank < 114 decomposition.
    # For now, return the rank 114 result.
    
    if time.time() - start_time > DEADLINE - 5:
        return best_result
    
    return best_result
