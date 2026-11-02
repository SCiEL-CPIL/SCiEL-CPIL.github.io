"""Matrix multiplication 2x3 by 3x4 tensor decomposition."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 4
    # Rank 22 decomposition
    # Split A into A_L (2x2) and A_R (2x1)
    # Split B into B_T (2x4) and B_B (1x4)
    # C = A_L B_T + A_R B_B
    
    # A_L B_T is 2x2x4, split into two 2x2x2 blocks
    # Block 1: k in {0, 1}
    # Block 2: k in {2, 3}
    
    # Strassen decomposition for 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def add_strassen_block(a_idx, b_idx, c_idx, U, V, W, rank_offset):
        # a_idx: [a, b, c, d] indices in A
        # b_idx: [e, f, g, h] indices in B
        # c_idx: [C00, C01, C10, C11] indices in C
        
        # M1 = (a+d)(e+h)
        U[a_idx[0]][rank_offset] += 1
        U[a_idx[3]][rank_offset] += 1
        V[b_idx[0]][rank_offset] += 1
        V[b_idx[3]][rank_offset] += 1
        W[c_idx[0]][rank_offset] += 1
        W[c_idx[3]][rank_offset] += 1
        
        # M2 = (c+d)e
        U[a_idx[2]][rank_offset+1] += 1
        U[a_idx[3]][rank_offset+1] += 1
        V[b_idx[0]][rank_offset+1] += 1
        W[c_idx[2]][rank_offset+1] += 1
        
        # M3 = a(f-h)
        U[a_idx[0]][rank_offset+2] += 1
        V[b_idx[1]][rank_offset+2] += 1
        V[b_idx[3]][rank_offset+2] -= 1
        W[c_idx[1]][rank_offset+2] += 1
        
        # M4 = d(g-e)
        U[a_idx[3]][rank_offset+3] += 1
        V[b_idx[2]][rank_offset+3] += 1
        V[b_idx[0]][rank_offset+3] -= 1
        W[c_idx[0]][rank_offset+3] += 1
        W[c_idx[2]][rank_offset+3] += 1
        
        # M5 = (a+b)h
        U[a_idx[0]][rank_offset+4] += 1
        U[a_idx[1]][rank_offset+4] += 1
        V[b_idx[3]][rank_offset+4] += 1
        W[c_idx[0]][rank_offset+4] -= 1
        W[c_idx[1]][rank_offset+4] += 1
        
        # M6 = (c-a)(e+f)
        U[a_idx[2]][rank_offset+5] += 1
        U[a_idx[0]][rank_offset+5] -= 1
        V[b_idx[0]][rank_offset+5] += 1
        V[b_idx[1]][rank_offset+5] += 1
        W[c_idx[3]][rank_offset+5] += 1
        
        # M7 = (b-d)(g+h)
        U[a_idx[1]][rank_offset+6] += 1
        U[a_idx[3]][rank_offset+6] -= 1
        V[b_idx[2]][rank_offset+6] += 1
        V[b_idx[3]][rank_offset+6] += 1
        W[c_idx[0]][rank_offset+6] += 1
        W[c_idx[3]][rank_offset+6] -= 1
        
        return rank_offset + 7

    # Initialize U, V, W with rank 22
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # A indices for A_L (2x2 part of 2x3)
    # A is 2x3. Indices:
    # 0 1 2
    # 3 4 5
    # A_L is cols 0,1:
    # a=0, b=1
    # c=3, d=4
    a_L_idx = [0, 1, 3, 4]
    
    # B indices for B_T (2x4 part of 3x4)
    # B is 3x4. Indices:
    # 0 1 2 3
    # 4 5 6 7
    # 8 9 10 11
    # B_T is rows 0,1:
    # e=0, f=1, g=4, h=5 (for k=0,1)
    # e=2, f=3, g=6, h=7 (for k=2,3)
    
    # C indices
    # C is 2x4. Output index k*n + i.
    # k=0: i=0->0, i=1->1
    # k=1: i=0->2, i=1->3
    # k=2: i=0->4, i=1->5
    # k=3: i=0->6, i=1->7
    
    # Block 1: k=0,1
    # B_T1: e=0, f=1, g=4, h=5
    b_T1_idx = [0, 1, 4, 5]
    # C1: C00=0, C01=2, C10=1, C11=3
    c_1_idx = [0, 2, 1, 3]
    
    rank_offset = add_strassen_block(a_L_idx, b_T1_idx, c_1_idx, U, V, W, 0)
    
    # Block 2: k=2,3
    # B_T2: e=2, f=3, g=6, h=7
    b_T2_idx = [2, 3, 6, 7]
    # C2: C02=4, C03=6, C12=5, C13=7
    c_2_idx = [4, 6, 5, 7]
    
    rank_offset = add_strassen_block(a_L_idx, b_T2_idx, c_2_idx, U, V, W, rank_offset)
    
    # A_R B_B part
    # A_R is col 2:
    # a=2, b=5 (using a,b for A_R[0,0], A_R[1,0] effectively)
    # Actually A_R is 2x1.
    # A_R[0] = A[0,2] = index 2
    # A_R[1] = A[1,2] = index 5
    
    # B_B is row 2:
    # B_B[0] = B[2,0] = index 8
    # B_B[1] = B[2,1] = index 9
    # B_B[2] = B[2,2] = index 10
    # B_B[3] = B[2,3] = index 11
    
    # Terms:
    # C[0,0] (idx 0) = A[0,2] * B[2,0]
    # C[0,1] (idx 2) = A[0,2] * B[2,1]
    # C[0,2] (idx 4) = A[0,2] * B[2,2]
    # C[0,3] (idx 6) = A[0,2] * B[2,3]
    # C[1,0] (idx 1) = A[1,2] * B[2,0]
    # C[1,1] (idx 3) = A[1,2] * B[2,1]
    # C[1,2] (idx 5) = A[1,2] * B[2,2]
    # C[1,3] (idx 7) = A[1,2] * B[2,3]
    
    a_R_0 = 2
    a_R_1 = 5
    b_B_0 = 8
    b_B_1 = 9
    b_B_2 = 10
    b_B_3 = 11
    
    c_00 = 0
    c_01 = 2
    c_02 = 4
    c_03 = 6
    c_10 = 1
    c_11 = 3
    c_12 = 5
    c_13 = 7
    
    # Rank 14-21
    # 14: A[0,2] * B[2,0] -> C[0,0]
    U[a_R_0][14] = 1
    V[b_B_0][14] = 1
    W[c_00][14] = 1
    
    # 15: A[0,2] * B[2,1] -> C[0,1]
    U[a_R_0][15] = 1
    V[b_B_1][15] = 1
    W[c_01][15] = 1
    
    # 16: A[0,2] * B[2,2] -> C[0,2]
    U[a_R_0][16] = 1
    V[b_B_2][16] = 1
    W[c_02][16] = 1
    
    # 17: A[0,2] * B[2,3] -> C[0,3]
    U[a_R_0][17] = 1
    V[b_B_3][17] = 1
    W[c_03][17] = 1
    
    # 18: A[1,2] * B[2,0] -> C[1,0]
    U[a_R_1][18] = 1
    V[b_B_0][18] = 1
    W[c_10][18] = 1
    
    # 19: A[1,2] * B[2,1] -> C[1,1]
    U[a_R_1][19] = 1
    V[b_B_1][19] = 1
    W[c_11][19] = 1
    
    # 20: A[1,2] * B[2,2] -> C[1,2]
    U[a_R_1][20] = 1
    V[b_B_2][20] = 1
    W[c_12][20] = 1
    
    # 21: A[1,2] * B[2,3] -> C[1,3]
    U[a_R_1][21] = 1
    V[b_B_3][21] = 1
    W[c_13][21] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
