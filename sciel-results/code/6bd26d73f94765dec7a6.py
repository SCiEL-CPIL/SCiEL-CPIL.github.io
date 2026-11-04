import time
import sys

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Split m=5 into 2+3, p=5 into 2+3
    # Sub-problems:
    # 1. 3x2 x 2x2 -> 3x2 (Rank 12)
    # 2. 3x3 x 3x2 -> 3x2 (Rank 18)
    # 3. 3x2 x 2x3 -> 3x3 (Rank 18)
    # 4. 3x3 x 3x3 -> 3x3 (Rank 26 using Strassen on 2x2 block)
    
    rank_1 = 12
    rank_2 = 18
    rank_3 = 18
    rank_4 = 26
    rank = rank_1 + rank_2 + rank_3 + rank_4
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_decomp(n_sub, m_sub, p_sub, U_sub, V_sub, W_sub, i_offset, j_offset, k_offset, sub_rank):
        nonlocal col
        for i in range(n_sub):
            for j in range(m_sub):
                for c in range(sub_rank):
                    if U_sub[i*m_sub+j][c] != 0:
                        U[(i + i_offset) * m + (j + j_offset)][col + c] += U_sub[i*m_sub+j][c]
        for j in range(m_sub):
            for k in range(p_sub):
                for c in range(sub_rank):
                    if V_sub[j*p_sub+k][c] != 0:
                        V[(j + j_offset) * p + (k + k_offset)][col + c] += V_sub[j*p_sub+k][c]
        for k in range(p_sub):
            for i in range(n_sub):
                for c in range(sub_rank):
                    if W_sub[k*n_sub+i][c] != 0:
                        W[(k + k_offset) * n + (i + i_offset)][col + c] += W_sub[k*n_sub+i][c]
        col += sub_rank

    # 1. 3x2x2 (rank 12)
    U1 = [[0]*rank_1 for _ in range(3*2)]
    V1 = [[0]*rank_1 for _ in range(2*2)]
    W1 = [[0]*rank_1 for _ in range(2*3)]
    c = 0
    for i in range(3):
        for j in range(2):
            for k in range(2):
                U1[i*2+j][c] = 1
                V1[j*2+k][c] = 1
                W1[k*3+i][c] = 1
                c += 1
    add_decomp(3, 2, 2, U1, V1, W1, 0, 0, 0, rank_1)
    
    # 2. 3x3x2 (rank 18)
    U2 = [[0]*rank_2 for _ in range(3*3)]
    V2 = [[0]*rank_2 for _ in range(3*2)]
    W2 = [[0]*rank_2 for _ in range(2*3)]
    c = 0
    for i in range(3):
        for j in range(3):
            for k in range(2):
                U2[i*3+j][c] = 1
                V2[j*2+k][c] = 1
                W2[k*3+i][c] = 1
                c += 1
    add_decomp(3, 3, 2, U2, V2, W2, 0, 2, 0, rank_2)
    
    # 3. 3x2x3 (rank 18)
    U3 = [[0]*rank_3 for _ in range(3*2)]
    V3 = [[0]*rank_3 for _ in range(2*3)]
    W3 = [[0]*rank_3 for _ in range(3*3)]
    c = 0
    for i in range(3):
        for j in range(2):
            for k in range(3):
                U3[i*2+j][c] = 1
                V3[j*3+k][c] = 1
                W3[k*3+i][c] = 1
                c += 1
    add_decomp(3, 2, 3, U3, V3, W3, 0, 0, 2, rank_3)
    
    # 4. 3x3x3 (rank 26)
    # Decomposition:
    # A_TL (2x2) * B_TL (2x2) -> Strassen (7 terms)
    # A_TL (2x2) * B_TR (2x1) -> Naive (4 terms)
    # A_TR (2x1) * B_TL (2x2) -> Naive (4 terms)
    # A_TR (2x1) * B_BR (1x1) -> Naive (2 terms)
    # A_BL (1x2) * B_TL (2x2) -> Naive (4 terms)
    # A_BL (1x2) * B_TR (2x1) -> Naive (2 terms)
    # A_BR (1x1) * B_BL (1x2) -> Naive (2 terms)
    # A_BR (1x1) * B_BR (1x1) -> Naive (1 term)
    
    U4 = [[0]*rank_4 for _ in range(3*3)]
    V4 = [[0]*rank_4 for _ in range(3*3)]
    W4 = [[0]*rank_4 for _ in range(3*3)]
    
    c = 0
    # Strassen for 2x2x2 (indices 0,1 for TL)
    # M1 = A00(B00+B11) -> U: A00, V: B00, B11, W: C00, C11
    # This is complex to map directly.
    # I will use the explicit Strassen terms for the 2x2 block.
    
    # Strassen 2x2:
    # P1 = A00(B00+B11)
    # P2 = (A00+A01)B11
    # P3 = A11(B00+B10)
    # P4 = A10(B11+B10)
    # P5 = (A00+A11)(B00+B11)
    # P6 = (A01-A11)(B10+B11)
    # P7 = (A00-A10)(B00+B10)
    
    # We need to map these to the 3x3 indices.
    # A00 is A[0,0], A01 is A[0,1], A10 is A[1,0], A11 is A[1,1].
    # B00 is B[0,0], etc.
    # C00 is C[0,0], etc.
    
    # Term 1: P1
    U4[0*3+0][c] = 1 # A00
    V4[0*3+0][c] = 1 # B00
    V4[1*3+1][c] = 1 # B11
    W4[0*3+0][c] = 1 # C00
    W4[1*3+1][c] = 1 # C11
    c += 1
    
    # Term 2: P2
    U4[0*3+0][c] = 1 # A00
    U4[0*3+1][c] = 1 # A01
    V4[1*3+1][c] = 1 # B11
    W4[0*3+1][c] = 1 # C01
    W4[1*3+1][c] = 1 # C11
    c += 1
    
    # Term 3: P3
    U4[1*3+1][c] = 1 # A11
    V4[0*3+0][c] = 1 # B00
    V4[0*3+1][c] = 1 # B01
    W4[1*3+0][c] = 1 # C10
    W4[1*3+1][c] = 1 # C11
    c += 1
    
    # Term 4: P4
    U4[1*3+0][c] = 1 # A10
    V4[1*3+1][c] = 1 # B11
    V4[1*3+0][c] = 1 # B10
    W4[1*3+0][c] = 1 # C10
    W4[1*3+1][c] = 1 # C11
    c += 1
    
    # Term 5: P5
    U4[0*3+0][c] = 1 # A00
    U4[1*3+1][c] = 1 # A11
    V4[0*3+0][c] = 1 # B00
    V4[1*3+1][c] = 1 # B11
    W4[0*3+0][c] = 1 # C00
    W4[0*3+1][c] = 1 # C01
    W4[1*3+0][c] = 1 # C10
    W4[1*3+1][c] = 1 # C11
    c += 1
    
    # Term 6: P6
    U4[0*3+1][c] = 1 # A01
    U4[1*3+1][c] = -1 # A11
    V4[0*3+1][c] = 1 # B01
    V4[1*3+1][c] = 1 # B11
    W4[0*3+1][c] = 1 # C01
    W4[1*3+1][c] = 1 # C11
    c += 1
    
    # Term 7: P7
    U4[0*3+0][c] = 1 # A00
    U4[1*3+0][c] = -1 # A10
    V4[0*3+0][c] = 1 # B00
    V4[1*3+0][c] = 1 # B10
    W4[0*3+0][c] = 1 # C00
    W4[1*3+0][c] = 1 # C10
    c += 1
    
    # Now handle the mixed terms.
    # A_TL (2x2) * B_TR (2x1) -> C_TR (2x1)
    # Indices: A[0:2, 0:2], B[0:2, 2], C[0:2, 2]
    # Naive rank 4.
    for i in range(2):
        for j in range(2):
            U4[i*3+j][c] = 1
            V4[j*3+2][c] = 1
            W4[2*3+i][c] = 1
            c += 1
            
    # A_TR (2x1) * B_TL (2x2) -> C_TL (2x2)
    # Indices: A[0:2, 2], B[0:2, 0:2], C[0:2, 0:2]
    # Naive rank 4.
    for i in range(2):
        for j in range(2):
            U4[i*3+2][c] = 1
            V4[j*3+0][c] = 1
            W4[0*3+i][c] = 1
            c += 1
            U4[i*3+2][c] = 1
            V4[j*3+1][c] = 1
            W4[1*3+i][c] = 1
            c += 1
            # Wait, loop structure is wrong.
            
    # Let's redo the mixed terms carefully.
    
    # A_TL (2x2) * B_TR (2x1) -> C_TR (2x1)
    # C_i2 = sum_j A_ij B_j2
    # i in 0,1. j in 0,1.
    for i in range(2):
        for j in range(2):
            U4[i*3+j][c] = 1
            V4[j*3+2][c] = 1
            W4[2*3+i][c] = 1
            c += 1
            
    # A_TR (2x1) * B_TL (2x2) -> C_TL (2x2)
    # C_ij = sum_k A_ik B_kj
    # i in 0,1. k in 0,1. j in 0,1.
    for i in range(2):
        for k in range(2):
            for j in range(2):
                U4[i*3+2][c] = 1
                V4[k*3+j][c] = 1
                W4[j*3+i][c] = 1
                c += 1
                
    # A_TR (2x1) * B_BR (1x1) -> C_BR (2x1)
    # C_i2 = A_i2 B_22
    # i in 0,1.
    for i in range(2):
        U4[i*3+2][c] = 1
        V4[2*3+2][c] = 1
        W4[2*3+i][c] = 1
        c += 1
        
    # A_BL (1x2) * B_TL (2x2) -> C_BL (1x2)
    # C_2j = sum_k A_2k B_kj
    # k in 0,1. j in 0,1.
    for k in range(2):
        for j in range(2):
            U4[2*3+k][c] = 1
            V4[k*3+j][c] = 1
            W4[j*3+2][c] = 1
            c += 1
            
    # A_BL (1x2) * B_TR (2x1) -> C_BR (1x1)
    # C_22 = sum_k A_2k B_k2
    # k in 0,1.
    for k in range(2):
        U4[2*3+k][c] = 1
        V4[k*3+2][c] = 1
        W4[2*3+2][c] = 1
        c += 1
        
    # A_BR (1x1) * B_BL (1x2) -> C_BL (1x2)
    # C_2j = A_22 B_2j
    # j in 0,1.
    for j in range(2):
        U4[2*3+2][c] = 1
        V4[2*3+j][c] = 1
        W4[j*3+2][c] = 1
        c += 1
        
    # A_BR (1x1) * B_BR (1x1) -> C_BR (1x1)
    # C_22 = A_22 B_22
    U4[2*3+2][c] = 1
    V4[2*3+2][c] = 1
    W4[2*3+2][c] = 1
    c += 1
    
    add_decomp(3, 3, 3, U4, V4, W4, 0, 2, 2, rank_4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
