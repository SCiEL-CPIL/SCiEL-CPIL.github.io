import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    # We'll build the decomposition block by block.
    # C = A * B, A is n×m, B is m×p, C is n×p
    # 
    # Strategy: Split B into B_left (m×3) and B_right (m×4)
    # C_left = A * B_left (2×4 × 4×3 → 2×3)
    # C_right = A * B_right (2×4 × 4×4 → 2×4)
    #
    # For C_right: split A into A1|A2 (2×2 each), B_right into B1;B2 (2×4 each)
    #   C_right = A1*B1 + A2*B2
    #   Each Ai*Bi is 2×2 × 2×4 → 2×4, split into two 2×2×2 Strassen blocks
    #   Total: 4 * 7 = 28
    #
    # For C_left: split A into A1|A2, B_left into B1;B2 (2×3 each)
    #   C_left = A1*B1 + A2*B2
    #   Each Ai*Bi is 2×2 × 2×3 → 2×3, split into 2×2×2 (Strassen) + 2×2×1 (school)
    #   Total: 2 * (7 + 4) = 22
    #
    # Grand total: 50
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_block(U_block, V_block, W_block):
        """Add a block of matrices (each is a list of columns)."""
        U_cols.append(U_block)
        V_cols.append(V_block)
        W_cols.append(W_block)
    
    def strassen_222():
        """
        Strassen decomposition for 2×2 × 2×2 → 2×2.
        Returns U (4×7), V (4×7), W (4×7) where:
        U[i*2+j, c] for i,j in 0..1 (A element)
        V[j*2+k, c] for j,k in 0..1 (B element)  
        W[k*2+i, c] for k,i in 0..1 (C element, trace convention W[k*n+i])
        """
        # Strassen's 7 multiplications for 2x2x2
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # C = [[ae+cg, af+dh], [ce+dg, cf+gh]]
        # 
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        # 
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        U = [[0]*7 for _ in range(4)]  # 2x2 A elements: a=0, b=1, c=2, d=3
        V = [[0]*7 for _ in range(4)]  # 2x2 B elements: e=0, f=1, g=2, h=3
        W = [[0]*7 for _ in range(4)]  # 2x2 C elements: C11=0, C12=1, C21=2, C22=3
        # W index: W[k*2+i], so C11: k=0,i=0 -> idx 0; C12: k=0,i=1 -> idx 1
        # C21: k=1,i=0 -> idx 2; C22: k=1,i=1 -> idx 3
        
        # M1 = (a+d)(e+h): U has a=1,d=1; V has e=1,h=1; W contributes to C11(+1), C22(+1)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (c+d)e: U has c=1,d=1; V has e=1; W contributes to C21(+1), C22(-1)
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1
        
        # M3 = a(f-h): U has a=1; V has f=1,h=-1; W contributes to C12(+1), C22(+1)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # M4 = d(g-e): U has d=1; V has g=1,e=-1; W contributes to C11(+1), C21(+1)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # M5 = (a+b)h: U has a=1,b=1; V has h=1; W contributes to C11(-1), C12(+1)
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # M6 = (c-a)(e+f): U has c=1,a=-1; V has e=1,f=1; W contributes to C22(+1)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # M7 = (b-d)(g+h): U has b=1,d=-1; V has g=1,h=1; W contributes to C11(+1)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W
    
    def schoolbook_221():
        """
        Schoolbook decomposition for 2×2 × 2×1 → 2×1.
        A is 2×2, B is 2×1 (column vector), C is 2×1.
        U (4×4), V (2×4), W (2×4)
        U[i*2+j, c] for A element (i,j in 0..1)
        V[j*1+k, c] for B element (j in 0..1, k in 0..0)
        W[k*2+i, c] for C element (k in 0..0, i in 0..1)
        """
        U = [[0]*4 for _ in range(4)]  # A: 2x2
        V = [[0]*4 for _ in range(2)]  # B: 2x1
        W = [[0]*4 for _ in range(2)]  # C: 2x1, W[k*2+i]
        
        # C[0] = A[0][0]*B[0] + A[0][1]*B[1]
        # C[1] = A[1][0]*B[0] + A[1][1]*B[1]
        
        # Term 1: A[0][0]*B[0] -> C[0]
        U[0*2+0][0] = 1
        V[0*1+0][0] = 1
        W[0*2+0][0] = 1
        
        # Term 2: A[0][1]*B[1] -> C[0]
        U[0*2+1][1] = 1
        V[1*1+0][1] = 1
        W[0*2+0][1] = 1
        
        # Term 3: A[1][0]*B[0] -> C[1]
        U[1*2+0][2] = 1
        V[0*1+0][2] = 1
        W[0*2+1][2] = 1
        
        # Term 4: A[1][1]*B[1] -> C[1]
        U[1*2+1][3] = 1
        V[1*1+0][3] = 1
        W[0*2+1][3] = 1
        
        return U, V, W
    
    # Build C_right: A * B_right where B_right is 4×4
    # A = [A1 | A2] (each 2×2), B_right = [B1; B2] (each 2×4)
    # C_right = A1*B1 + A2*B2
    # Each Ai*Bi is 2×2 × 2×4 → 2×4, split into two 2×2×2 blocks
    
    # A1*B1: A1 is rows 0-1, cols 0-1 of A; B1 is rows 0-1, cols 0-3 of B_right (cols 3-6 of B)
    # A2*B2: A2 is rows 0-1, cols 2-3 of A; B2 is rows 2-3, cols 0-3 of B_right (cols 3-6 of B)
    
    # For each 2×2×2 Strassen block, we need to map local indices to global
    # 
    # Block 1: A1*B1_1 (A1 is 2×2, B1_1 is 2×2, result is 2×2)
    #   A1 global: rows 0-1, cols 0-1 → U index: i*4+j for i in 0..1, j in 0..1
    #   B1_1 global: rows 0-1, cols 3-4 (since B_right starts at col 3) → V index: j*7+k for j in 0..1, k in 3..4
    #   Result: C_right[0:2, 0:2] → W index: k*2+i for k in 0..1, i in 0..1
    
    S1 = strassen_222()
    # Map S1 U (local 2×2 A: i*2+j) to global U (i*4+j)
    U1 = [[0]*7 for _ in range(8)]  # global U is 8×rank
    for i in range(2):
        for j in range(2):
            for c in range(7):
                U1[i*4+j][c] = S1[0][i*2+j][c]
    
    V1 = [[0]*7 for _ in range(14)]  # global V is 14×rank
    for j in range(2):
        for k in range(2):
            for c in range(7):
                V1[j*7+(3+k)][c] = S1[1][j*2+k][c]
    
    W1 = [[0]*7 for _ in range(14)]  # global W is 14×rank
    for k in range(2):
        for i in range(2):
            for c in range(7):
                W1[k*2+i][c] = S1[2][k*2+i][c]
    
    add_block(U1, V1, W1)
    
    # Block 2: A1*B1_2 (A1 is 2×2, B1_2 is 2×2, result is 2×2)
    #   B1_2 global: rows 0-1, cols 5-6 → V index: j*7+k for j in 0..1, k in 5..6
    #   Result: C_right[0:2, 2:4] → W index: k*2+i for k in 2..3, i in 0..1
    
    S2 = strassen_222()
    U2 = [[0]*7 for _ in range(8)]
    for i in range(2):
        for j in range(2):
            for c in range(7):
                U2[i*4+j][c] = S2[0][i*2+j][c]
    
    V2 = [[0]*7 for _ in range(14)]
    for j in range(2):
        for k in range(2):
            for c in range(7):
                V2[j*7+(5+k)][c] = S2[1][j*2+k][c]
    
    W2 = [[0]*7 for _ in range(14)]
    for k in range(2):
        for i in range(2):
            for c in range(7):
                W2[(2+k)*2+i][c] = S2[2][k*2+i][c]
    
    add_block(U2, V2, W2)
    
    # Block 3: A2*B2_1 (A2 is 2×2, B2_1 is 2×2, result is 2×2)
    #   A2 global: rows 0-1, cols 2-3 → U index: i*4+j for i in 0..1, j in 2..3
    #   B2_1 global: rows 2-3, cols 3-4 → V index: j*7+k for j in 2..3, k in 3..4
    #   Result: C_right[0:2, 0:2] → W index: k*2+i for k in 0..1, i in 0..1
    
    S3 = strassen_222()
    U3 = [[0]*7 for _ in range(8)]
    for i in range(2):
        for j in range(2):
            for c in range(7):
                U3[i*4+(2+j)][c] = S3[0][i*2+j][c]
    
    V3 = [[0]*7 for _ in range(14)]
    for j in range(2):
        for k in range(2):
            for c in range(7):
                V3[(2+j)*7+(3+k)][c] = S3[1][j*2+k][c]
    
    W3 = [[0]*7 for _ in range(14)]
    for k in range(2):
        for i in range(2):
            for c in range(7):
                W3[k*2+i][c] = S3[2][k*2+i][c]
    
    add_block(U3, V3, W3)
    
    # Block 4: A2*B2_2 (A2 is 2×2, B2_2 is 2×2, result is 2×2)
    #   B2_2 global: rows 2-3, cols 5-6 → V index: j*7+k for j in 2..3, k in 5..6
    #   Result: C_right[0:2, 2:4] → W index: k*2+i for k in 2..3, i in 0..1
    
    S4 = strassen_222()
    U4 = [[0]*7 for _ in range(8)]
    for i in range(2):
        for j in range(2):
            for c in range(7):
                U4[i*4+(2+j)][c] = S4[0][i*2+j][c]
    
    V4 = [[0]*7 for _ in range(14)]
    for j in range(2):
        for k in range(2):
            for c in range(7):
                V4[(2+j)*7+(5+k)][c] = S4[1][j*2+k][c]
    
    W4 = [[0]*7 for _ in range(14)]
    for k in range(2):
        for i in range(2):
            for c in range(7):
                W4[(2+k)*2+i][c] = S4[2][k*2+i][c]
    
    add_block(U4, V4, W4)
    
    # Build C_left: A * B_left where B_left is 4×3
    # A = [A1 | A2] (each 2×2), B_left = [B1; B2] (each 2×3)
    # C_left = A1*B1 + A2*B2
    # Each Ai*Bi is 2×2 × 2×3 → 2×3, split into 2×2×2 (Strassen) + 2×2×1 (school)
    
    # Block 5: A1*B1_1 (Strassen 2×2×2)
    #   A1 global: rows 0-1, cols 0-1 → U index: i*4+j for i in 0..1, j in 0..1
    #   B1_1 global: rows 0-1, cols 0-1 → V index: j*7+k for j in 0..1, k in 0..1
    #   Result: C_left[0:2, 0:2] → W index: k*2+i for k in 0..1, i in 0..1
    
    S5 = strassen_222()
    U5 = [[0]*7 for _ in range(8)]
    for i in range(2):
        for j in range(2):
            for c in range(7):
                U5[i*4+j][c] = S5[0][i*2+j][c]
    
    V5 = [[0]*7 for _ in range(14)]
    for j in range(2):
        for k in range(2):
            for c in range(7):
                V5[j*7+k][c] = S5[1][j*2+k][c]
    
    W5 = [[0]*7 for _ in range(14)]
    for k in range(2):
        for i in range(2):
            for c in range(7):
                W5[k*2+i][c] = S5[2][k*2+i][c]
    
    add_block(U5, V5, W5)
    
    # Block 6: A1*B1_2 (schoolbook 2×2×1)
    #   A1 global: rows 0-1, cols 0-1 → U index: i*4+j for i in 0..1, j in 0..1
    #   B1_2 global: rows 0-1, col 2 → V index: j*7+k for j in 0..1, k in 2
    #   Result: C_left[0:2, 2:3] → W index: k*2+i for k in 2, i in 0..1
    
    SB6 = schoolbook_221()
    U6 = [[0]*4 for _ in range(8)]
    for i in range(2):
        for j in range(2):
            for c in range(4):
                U6[i*4+j][c] = SB6[0][i*2+j][c]
    
    V6 = [[0]*4 for _ in range(14)]
    for j in range(2):
        for c in range(4):
            V6[j*7+2][c] = SB6[1][j*1+0][c]
    
    W6 = [[0]*4 for _ in range(14)]
    for i in range(2):
        for c in range(4):
            W6[2*2+i][c] = SB6[2][0*2+i][c]
    
    add_block(U6, V6, W6)
    
    # Block 7: A2*B2_1 (Strassen 2×2×2)
    #   A2 global: rows 0-1, cols 2-3 → U index: i*4+j for i in 0..1, j in 2..3
    #   B2_1 global: rows 2-3, cols 0-1 → V index: j*7+k for j in 2..3, k in 0..1
    #   Result: C_left[0:2, 0:2] → W index: k*2+i for k in 0..1, i in 0..1
    
    S7 = strassen_222()
    U7 = [[0]*7 for _ in range(8)]
    for i in range(2):
        for j in range(2):
            for c in range(7):
                U7[i*4+(2+j)][c] = S7[0][i*2+j][c]
    
    V7 = [[0]*7 for _ in range(14)]
    for j in range(2):
        for k in range(2):
            for c in range(7):
                V7[(2+j)*7+k][c] = S7[1][j*2+k][c]
    
    W7 = [[0]*7 for _ in range(14)]
    for k in range(2):
        for i in range(2):
            for c in range(7):
                W7[k*2+i][c] = S7[2][k*2+i][c]
    
    add_block(U7, V7, W7)
    
    # Block 8: A2*B2_2 (schoolbook 2×2×1)
    #   A2 global: rows 0-1, cols 2-3 → U index: i*4+j for i in 0..1, j in 2..3
    #   B2_2 global: rows 2-3, col 2 → V index: j*7+k for j in 2..3, k in 2
    #   Result: C_left[0:2, 2:3] → W index: k*2+i for k in 2, i in 0..1
    
    SB8 = schoolbook_221()
    U8 = [[0]*4 for _ in range(8)]
    for i in range(2):
        for j in range(2):
            for c in range(4):
                U8[i*4+(2+j)][c] = SB8[0][i*2+j][c]
    
    V8 = [[0]*4 for _ in range(14)]
    for j in range(2):
        for c in range(4):
            V8[(2+j)*7+2][c] = SB8[1][j*1+0][c]
    
    W8 = [[0]*4 for _ in range(14)]
    for i in range(2):
        for c in range(4):
            W8[2*2+i][c] = SB8[2][0*2+i][c]
    
    add_block(U8, V8, W8)
    
    # Now concatenate all blocks
    total_rank = sum(len(block[0]) for block in U_cols)
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for u_block, v_block, w_block in zip(U_cols, V_cols, W_cols):
        block_rank = len(u_block[0])
        for r in range(n * m):
            for c in range(block_rank):
                U[r][col_offset + c] = u_block[r][c]
        for r in range(m * p):
            for c in range(block_rank):
                V[r][col_offset + c] = v_block[r][c]
        for r in range(p * n):
            for c in range(block_rank):
                W[r][col_offset + c] = w_block[r][c]
        col_offset += block_rank
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
