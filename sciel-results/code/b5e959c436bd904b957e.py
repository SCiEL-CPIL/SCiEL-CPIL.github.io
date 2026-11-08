"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We decompose the 3x5x6 multiplication into blocks.
    # A (3x5) = [A1 (3x2), A2 (3x3)]
    # B (5x6) = [B1 (2x6); B2 (3x6)]
    # C = A1 B1 + A2 B2
    
    # B1 (2x6) = [B11 (2x3), B12 (2x3)]
    # B2 (3x6) = [B21 (3x3), B22 (3x3)]
    
    # Term 1: A1 (3x2) * B1 (2x6) = A1 B11 + A1 B12
    # Two multiplications of type 3x2x3.
    
    # Term 2: A2 (3x3) * B2 (3x6) = A2 B21 + A2 B22
    # Two multiplications of type 3x3x3.
    
    # We will construct the U, V, W matrices for each block and combine them.
    
    def build_3x2x3_decomp(offset):
        # Returns U, V, W for 3x2x3 with rank 17
        # A (3x2), B (2x3), C (3x3)
        # Indices:
        # A: i in 0..2, j in 0..1 -> U[i*2+j, col]
        # B: j in 0..1, k in 0..2 -> V[j*3+k, col]
        # C: k in 0..2, i in 0..2 -> W[k*3+i, col]
        
        U = [[0]*17 for _ in range(3*2)]
        V = [[0]*17 for _ in range(2*3)]
        W = [[0]*17 for _ in range(3*3)]
        
        # 7 Strassen terms for top-left 2x2x2
        # A_sub = [[a,b],[c,d]] (rows 0,1; cols 0,1)
        # B_sub = [[g,h],[j,k]] (rows 0,1; cols 0,1)
        # Note: B is 2x3, so B[0,0]=g, B[0,1]=h, B[1,0]=j, B[1,1]=k
        
        # m1 = (a+d)(g+k)
        U[0][offset+0] = 1; U[3][offset+0] = 1
        V[0][offset+0] = 1; V[4][offset+0] = 1
        W[0][offset+0] = 1; W[3][offset+0] = 1; W[1][offset+0] = -1; W[4][offset+0] = -1
        
        # m2 = (c+d)g
        U[3][offset+1] = 1; U[4][offset+1] = 1
        V[0][offset+1] = 1
        W[0][offset+1] = 1; W[3][offset+1] = -1
        
        # m3 = a(h-j)
        U[0][offset+2] = 1
        V[1][offset+2] = 1; V[3][offset+2] = -1
        W[0][offset+2] = 1; W[1][offset+2] = -1
        
        # m4 = (a+b)k
        U[0][offset+3] = 1; U[1][offset+3] = 1
        V[4][offset+3] = 1
        W[0][offset+3] = -1; W[1][offset+3] = 1
        
        # m5 = (c+d)j
        U[3][offset+4] = 1; U[4][offset+4] = 1
        V[3][offset+4] = 1
        W[0][offset+4] = 1; W[3][offset+4] = -1
        
        # m6 = (a-b)(g+j)
        U[0][offset+5] = 1; U[1][offset+5] = -1
        V[0][offset+5] = 1; V[3][offset+5] = 1
        W[0][offset+5] = 1; W[1][offset+5] = -1; W[3][offset+5] = -1; W[4][offset+5] = 1
        
        # m7 = (b-d)(h+k)
        U[1][offset+6] = 1; U[4][offset+6] = -1
        V[1][offset+6] = 1; V[4][offset+6] = 1
        W[0][offset+6] = 1; W[1][offset+6] = -1; W[3][offset+6] = 1; W[4][offset+6] = -1
        
        # Remaining 10 terms
        # C13 = ai + bl (i=B[0,2], l=B[1,2])
        # p1 = ai
        U[0][offset+7] = 1
        V[2][offset+7] = 1
        W[2][offset+7] = 1
        
        # p2 = bl
        U[1][offset+8] = 1
        V[5][offset+8] = 1
        W[2][offset+8] = 1
        
        # C23 = ci + dl
        # p3 = ci
        U[3][offset+9] = 1
        V[2][offset+9] = 1
        W[2][offset+9] = 1
        
        # p4 = dl
        U[4][offset+10] = 1
        V[5][offset+10] = 1
        W[2][offset+10] = 1
        
        # C31 = eg + fj (e=A[2,0], f=A[2,1])
        # q1 = eg
        U[6][offset+11] = 1
        V[0][offset+11] = 1
        W[0][offset+11] = 1
        
        # q2 = fj
        U[7][offset+12] = 1
        V[3][offset+12] = 1
        W[0][offset+12] = 1
        
        # C32 = eh + fk
        # q3 = eh
        U[6][offset+13] = 1
        V[1][offset+13] = 1
        W[1][offset+13] = 1
        
        # q4 = fk
        U[7][offset+14] = 1
        V[4][offset+14] = 1
        W[1][offset+14] = 1
        
        # C33 = ei + fl
        # r1 = ei
        U[6][offset+15] = 1
        V[2][offset+15] = 1
        W[2][offset+15] = 1
        
        # r2 = fl
        U[7][offset+16] = 1
        V[5][offset+16] = 1
        W[2][offset+16] = 1
        
        return U, V, W

    def build_3x3x3_decomp(offset):
        # Returns U, V, W for 3x3x3 with rank 26
        # A (3x3), B (3x3), C (3x3)
        # Split into 2x2 and 1x1 blocks
        
        U = [[0]*26 for _ in range(3*3)]
        V = [[0]*26 for _ in range(3*3)]
        W = [[0]*26 for _ in range(3*3)]
        
        # 7 Strassen terms for top-left 2x2x2
        # A_sub = [[a,b],[c,d]] (rows 0,1; cols 0,1)
        # B_sub = [[e,f],[g,h]] (rows 0,1; cols 0,1)
        
        # m1 = (a+d)(e+h)
        U[0][offset+0] = 1; U[4][offset+0] = 1
        V[0][offset+0] = 1; V[3][offset+0] = 1
        W[0][offset+0] = 1; W[3][offset+0] = 1; W[1][offset+0] = -1; W[4][offset+0] = -1
        
        # m2 = (c+d)e
        U[3][offset+1] = 1; U[4][offset+1] = 1
        V[0][offset+1] = 1
        W[0][offset+1] = 1; W[3][offset+1] = -1
        
        # m3 = a(f-g)
        U[0][offset+2] = 1
        V[1][offset+2] = 1; V[3][offset+2] = -1
        W[0][offset+2] = 1; W[1][offset+2] = -1
        
        # m4 = (a+b)h
        U[0][offset+3] = 1; U[1][offset+3] = 1
        V[4][offset+3] = 1
        W[0][offset+3] = -1; W[1][offset+3] = 1
        
        # m5 = (c+d)g
        U[3][offset+4] = 1; U[4][offset+4] = 1
        V[3][offset+4] = 1
        W[0][offset+4] = 1; W[3][offset+4] = -1
        
        # m6 = (a-b)(e+g)
        U[0][offset+5] = 1; U[1][offset+5] = -1
        V[0][offset+5] = 1; V[3][offset+5] = 1
        W[0][offset+5] = 1; W[1][offset+5] = -1; W[3][offset+5] = -1; W[4][offset+5] = 1
        
        # m7 = (b-d)(f+h)
        U[1][offset+6] = 1; U[4][offset+6] = -1
        V[1][offset+6] = 1; V[4][offset+6] = 1
        W[0][offset+6] = 1; W[1][offset+6] = -1; W[3][offset+6] = 1; W[4][offset+6] = -1
        
        # Remaining 19 terms
        # C13 = A11*f13 + A12*f23 + A13*f33 (A11=a, A12=b, A13=A[0,2])
        # B13 = f13=B[0,2], f23=B[1,2], f33=B[2,2]
        # Actually, let's just do schoolbook for the remaining parts to be safe and simple, 
        # but optimized where possible.
        # The 26 rank construction is complex. 
        # Let's use the 26 rank logic derived:
        # M1 (7) done.
        # M2: A_TR (2x1) * B_BL (1x2) -> Rank 4
        # A_TR = [[A[0,2]],[A[1,2]]]
        # B_BL = [[B[2,0], B[2,1]]]
        # Terms:
        # a02*b20, a02*b21, a12*b20, a12*b21
        
        U[2][offset+7] = 1; V[6][offset+7] = 1; W[0][offset+7] = 1
        U[2][offset+8] = 1; V[7][offset+8] = 1; W[1][offset+8] = 1
        U[5][offset+9] = 1; V[6][offset+9] = 1; W[3][offset+9] = 1
        U[5][offset+10] = 1; V[7][offset+10] = 1; W[4][offset+10] = 1
        
        # M3: A_TL (2x2) * B_TR (2x1) -> Rank 4
        # A_TL = [[a,b],[c,d]]
        # B_TR = [[B[0,2]],[B[1,2]]]
        # Terms: a*b02, b*b02, c*b02, d*b02? No, matrix mult.
        # C_TR = A_TL B_TR + A_TR B_BR
        # We need A_TL B_TR.
        # (a,b; c,d) * (b02; b12)
        # Row 0: a*b02 + b*b12
        # Row 1: c*b02 + d*b12
        
        U[0][offset+11] = 1; V[2][offset+11] = 1; W[2][offset+11] = 1 # a*b02
        U[1][offset+12] = 1; V[5][offset+12] = 1; W[2][offset+12] = 1 # b*b12
        U[3][offset+13] = 1; V[2][offset+13] = 1; W[5][offset+13] = 1 # c*b02
        U[4][offset+14] = 1; V[5][offset+14] = 1; W[5][offset+14] = 1 # d*b12
        
        # M4: A_TR (2x1) * B_BR (1x1) -> Rank 2
        # A_TR = [a02, a12]
        # B_BR = b22
        # Terms: a02*b22, a12*b22
        
        U[2][offset+15] = 1; V[8][offset+15] = 1; W[2][offset+15] = 1
        U[5][offset+16] = 1; V[8][offset+16] = 1; W[5][offset+16] = 1
        
        # M5: A_BL (1x2) * B_TL (2x2) -> Rank 4
        # A_BL = [a20, a21]
        # B_TL = [[e,f],[g,h]]
        # Terms: a20*e, a20*f, a21*g, a21*h
        
        U[6][offset+17] = 1; V[0][offset+17] = 1; W[6][offset+17] = 1
        U[6][offset+18] = 1; V[1][offset+18] = 1; W[7][offset+18] = 1
        U[7][offset+19] = 1; V[3][offset+19] = 1; W[6][offset+19] = 1
        U[7][offset+20] = 1; V[4][offset+20] = 1; W[7][offset+20] = 1
        
        # M6: A_BR (1x1) * B_BL (1x2) -> Rank 2
        # A_BR = a22
        # B_BL = [b20, b21]
        # Terms: a22*b20, a22*b21
        
        U[8][offset+21] = 1; V[6][offset+21] = 1; W[6][offset+21] = 1
        U[8][offset+22] = 1; V[7][offset+22] = 1; W[7][offset+22] = 1
        
        # M7: A_BL (1x2) * B_TR (2x1) -> Rank 2
        # A_BL = [a20, a21]
        # B_TR = [b02, b12]
        # Terms: a20*b02, a21*b12
        
        U[6][offset+23] = 1; V[2][offset+23] = 1; W[8][offset+23] = 1
        U[7][offset+24] = 1; V[5][offset+24] = 1; W[8][offset+24] = 1
        
        # M8: A_BR (1x1) * B_BR (1x1) -> Rank 1
        # A_BR = a22
        # B_BR = b22
        # Terms: a22*b22
        
        U[8][offset+25] = 1; V[8][offset+25] = 1; W[8][offset+25] = 1
        
        return U, V, W

    # Combine blocks
    # Block 1: 3x2x3 (A1, B11) -> offset 0
    U1, V1, W1 = build_3x2x3_decomp(0)
    # Block 2: 3x2x3 (A1, B12) -> offset 17
    U2, V2, W2 = build_3x2x3_decomp(17)
    # Block 3: 3x3x3 (A2, B21) -> offset 34
    U3, V3, W3 = build_3x3x3_decomp(34)
    # Block 4: 3x3x3 (A2, B22) -> offset 60
    U4, V4, W4 = build_3x3x3_decomp(60)
    
    total_rank = 60 + 26 # 34 + 52 = 86
    
    U = [[0]*total_rank for _ in range(3*5)]
    V = [[0]*total_rank for _ in range(5*6)]
    W = [[0]*total_rank for _ in range(6*3)]
    
    # Map indices
    # A1 is 3x2 (rows 0-2, cols 0-1 of A)
    # A2 is 3x3 (rows 0-2, cols 2-4 of A)
    # B11 is 2x3 (rows 0-1, cols 0-2 of B)
    # B12 is 2x3 (rows 0-1, cols 3-5 of B)
    # B21 is 3x3 (rows 2-4, cols 0-2 of B)
    # B22 is 3x3 (rows 2-4, cols 3-5 of B)
    
    # U mapping:
    # Block 1,2: A1 (3x2). U1,U2 have shape 6x17.
    # A1[i,j] corresponds to U[i*2+j, col].
    # In full U, A1[i,j] is at [i*5+j, col].
    for i in range(3):
        for j in range(2):
            for c in range(17):
                U[i*5+j][c] += U1[i*2+j][c]
                U[i*5+j][17+c] += U2[i*2+j][c]
    
    # Block 3,4: A2 (3x3). U3,U4 have shape 9x26.
    # A2[i,j] corresponds to U3[i*3+j, col].
    # In full U, A2[i,j] is at [i*5+(j+2), col].
    for i in range(3):
        for j in range(3):
            for c in range(26):
                U[i*5+j+2][34+c] += U3[i*3+j][c]
                U[i*5+j+2][60+c] += U4[i*3+j][c]
    
    # V mapping:
    # Block 1: B11 (2x3). V1 shape 6x17.
    # B11[j,k] corresponds to V1[j*3+k, col].
    # In full V, B11[j,k] is at [j*6+k, col].
    for j in range(2):
        for k in range(3):
            for c in range(17):
                V[j*6+k][c] += V1[j*3+k][c]
                
    # Block 2: B12 (2x3). V2 shape 6x17.
    # B12[j,k] corresponds to V2[j*3+k, col].
    # In full V, B12[j,k] is at [j*6+(k+3), col].
    for j in range(2):
        for k in range(3):
            for c in range(17):
                V[j*6+k+3][17+c] += V2[j*3+k][c]
                
    # Block 3: B21 (3x3). V3 shape 9x26.
    # B21[j,k] corresponds to V3[j*3+k, col].
    # In full V, B21[j,k] is at [(j+2)*6+k, col].
    for j in range(3):
        for k in range(3):
            for c in range(26):
                V[(j+2)*6+k][34+c] += V3[j*3+k][c]
                
    # Block 4: B22 (3x3). V4 shape 9x26.
    # B22[j,k] corresponds to V4[j*3+k, col].
    # In full V, B22[j,k] is at [(j+2)*6+(k+3), col].
    for j in range(3):
        for k in range(3):
            for c in range(26):
                V[(j+2)*6+k+3][60+c] += V4[j*3+k][c]
                
    # W mapping:
    # All blocks produce 3x3 C blocks.
    # W1..W4 shape 9x17/26.
    # W[k*3+i, col].
    # Full W is 6x3. W[k*3+i, col].
    # The C blocks are added to the same C.
    for k in range(3):
        for i in range(3):
            for c in range(17):
                W[k*3+i][c] += W1[k*3+i][c]
                W[k*3+i][17+c] += W2[k*3+i][c]
            for c in range(26):
                W[k*3+i][34+c] += W3[k*3+i][c]
                W[k*3+i][60+c] += W4[k*3+i][c]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
