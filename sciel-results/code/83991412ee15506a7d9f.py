import time

DIMENSIONS = [4, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin
    
    # 2x2 Strassen decomposition (Rank 7)
    # U2, V2, W2 are 4x7 matrices
    # Indices: 0:11, 1:12, 2:21, 3:22 (0-indexed: 0:00, 1:01, 2:10, 3:11)
    # W index: k*n+i -> 0:00, 1:01, 2:10, 3:11
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1 # C00, C11
    
    # M2 = (A10+A11)B00
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1 # C10
    
    # M3 = A00(B01-B11)
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1; W2[3][2] = 1 # C01, C11
    
    # M4 = A11(B10-B00)
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[2][3] = 1 # C00, C10
    
    # M5 = (A00+A01)B11
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[1][4] = 1 # C00(-1), C01(1)
    
    # M6 = (A10-A00)(B00+B01)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1 # C11
    
    # M7 = (A01-A11)(B10+B11)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1 # C00
    
    # 4x4 Strassen (Rank 49)
    # Outer steps s=0..6
    # For each s, we have Xs, Ys (2x2 matrices of 2x2 blocks)
    # We need coefficients c_s[pq][IJ] for Xs and d_s[pq][JK] for Ys
    
    # Xs coefficients (for A blocks)
    # Xs is 2x2 matrix of blocks.
    # Xs[p][q] = sum c[p][q][I][J] A[I][J]
    
    # s=0: M1 = (A00+A11)(B00+B11)
    # X0[0][0] = A00+A11? No.
    # Strassen on 2x2 blocks:
    # A = [[A00, A01], [A10, A11]]
    # M1 = (A00+A11)(B00+B11)
    # This corresponds to product of (A00+A11) and (B00+B11).
    # But (A00+A11) is a 2x2 matrix.
    # So X0 is a 2x2 matrix where X0[0][0] = A00+A11, X0[0][1]=0, X0[1][0]=0, X0[1][1]=0?
    # No. The Strassen algorithm for 4x4 treats the 4x4 as 2x2 blocks.
    # The 7 products are:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each Ms is a product of two 2x2 matrices (which are sums of 2x2 blocks).
    # So for each s, we define Xs and Ys as 2x2 matrices of 2x2 blocks.
    # Wait, Xs is NOT a 2x2 matrix of blocks in the sense of the Strassen formula for the 4x4 case.
    # The formula is for the 4x4 matrix.
    # The "entries" of the 4x4 matrix are the 2x2 blocks.
    # So Xs is a 2x2 matrix where each entry is a 2x2 matrix (a block of A).
    
    # Let's define the block coefficients for Xs and Ys for each s.
    # Xs[p][q] is a 2x2 matrix.
    # We represent Xs[p][q] by its 4 scalar coefficients relative to A blocks.
    # Actually, Xs[p][q] is a linear combination of A00, A01, A10, A11.
    
    # s=0: M1 = (A00+A11)(B00+B11)
    # X0 = [[A00+A11, 0], [0, 0]]? No.
    # The product is (A00+A11) * (B00+B11).
    # This is a product of two 2x2 matrices.
    # Let P = A00+A11, Q = B00+B11.
    # M1 = P Q.
    # P is a 2x2 matrix. Q is a 2x2 matrix.
    # So X0 is a 2x2 matrix? No, P is the "A" part.
    # In the 2x2 Strassen decomposition of P Q, P plays the role of A, Q plays the role of B.
    # So for s=0, X0 = P, Y0 = Q.
    # P = A00+A11.
    # P is a 2x2 matrix.
    # P[0][0] = A00[0][0] + A11[0][0]
    # P[0][1] = A00[0][1] + A11[0][1]
    # P[1][0] = A00[1][0] + A11[1][0]
    # P[1][1] = A00[1][1] + A11[1][1]
    
    # So X0 is a 2x2 matrix of 2x2 blocks?
    # No, P is a 2x2 matrix of scalars (if we consider blocks as scalars).
    # But here the "scalars" are 2x2 blocks.
    # So P is a 2x2 matrix where each entry is a 2x2 block.
    # P[0][0] = A00 + A11 (block sum).
    # P[0][1] = 0.
    # P[1][0] = 0.
    # P[1][1] = 0.
    # Wait, A00 and A11 are 2x2 blocks.
    # So P is a 2x2 matrix of 2x2 blocks.
    # P[0][0] = A00 + A11.
    # P[0][1] = 0.
    # P[1][0] = 0.
    # P[1][1] = 0.
    
    # This is correct.
    
    # Let's define Xs and Ys for all 7 steps.
    # Xs is 2x2 matrix of 2x2 blocks.
    # We store coefficients for each block.
    # Xs[p][q][I][J] is the coefficient of block A[I][J] in Xs[p][q].
    
    X = [[[[0]*4 for _ in range(2)] for _ in range(2)] for _ in range(7)]
    Y = [[[[0]*4 for _ in range(2)] for _ in range(2)] for _ in range(7)]
    
    # s=0: M1 = (A00+A11)(B00+B11)
    # X0[0][0] = A00+A11
    X[0][0][0][0] = 1; X[0][0][0][1] = 0; X[0][0][1][0] = 0; X[0][0][1][1] = 1
    # Y0[0][0] = B00+B11
    Y[0][0][0][0] = 1; Y[0][0][0][1] = 0; Y[0][0][1][0] = 0; Y[0][0][1][1] = 1
    
    # s=1: M2 = (A10+A11)B00
    # X1[0][0] = A10+A11
    X[1][0][0][0] = 0; X[1][0][0][1] = 0; X[1][0][1][0] = 1; X[1][0][1][1] = 1
    # Y1[0][0] = B00
    Y[1][0][0][0] = 1
    
    # s=2: M3 = A00(B01-B11)
    # X2[0][0] = A00
    X[2][0][0][0] = 1
    # Y2[0][0] = B01-B11
    Y[2][0][0][1] = 1; Y[2][0][1][1] = -1
    
    # s=3: M4 = A11(B10-B00)
    # X3[0][0] = A11
    X[3][0][1][1] = 1
    # Y3[0][0] = B10-B00
    Y[3][0][1][0] = 1; Y[3][0][0][0] = -1
    
    # s=4: M5 = (A00+A01)B11
    # X4[0][0] = A00+A01
    X[4][0][0][0] = 1; X[4][0][0][1] = 1
    # Y4[0][0] = B11
    Y[4][0][1][1] = 1
    
    # s=5: M6 = (A10-A00)(B00+B01)
    # X5[0][0] = A10-A00
    X[5][0][1][0] = 1; X[5][0][0][0] = -1
    # Y5[0][0] = B00+B01
    Y[5][0][0][0] = 1; Y[5][0][0][1] = 1
    
    # s=6: M7 = (A01-A11)(B10+B11)
    # X6[0][0] = A01-A11
    X[6][0][0][1] = 1; X[6][0][1][1] = -1
    # Y6[0][0] = B10+B11
    Y[6][0][1][0] = 1; Y[6][0][1][1] = 1

    # Construct U, V, W (16x49)
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    for s in range(7):
        for r in range(7):
            col = s * 7 + r
            # U
            for i in range(4):
                for j in range(4):
                    idx = i * 4 + j
                    # i = 2I+a, j = 2J+b
                    I = i // 2; a = i % 2
                    J = j // 2; b = j % 2
                    coeff = 0
                    for p in range(2):
                        for q in range(2):
                            e = p * 2 + q
                            # U2[e][r] is coefficient of Xs[p][q] in U2
                            coeff += U2[e][r] * X[s][p][q][I][J]
                    U[idx][col] = coeff
            
            # V
            for i in range(4):
                for j in range(4):
                    idx = i * 4 + j
                    I = i // 2; a = i % 2
                    J = j // 2; b = j % 2
                    coeff = 0
                    for p in range(2):
                        for q in range(2):
                            e = p * 2 + q
                            coeff += V2[e][r] * Y[s][p][q][I][J]
                    V[idx][col] = coeff
            
            # W
            # W index is k*n+i -> k*4+i
            # W2 index is k*2+i
            # W2[k2][r] contributes to W[k][i]
            # k2 = 2K+c, i2 = 2I+d ?
            # W2 row index is k*2+i.
            # k is row of C (2x2 block), i is col of C (2x2 block).
            # So W2[e][r] where e = k*2+i.
            # This contributes to C[k][i] which is a 2x2 block.
            # C[k][i] = sum_r W2[k*2+i][r] * (Zs[k][i])
            # Zs[k][i] is a 2x2 block of C.
            # Zs = Xs Ys.
            # We don't have Zs explicitly, but the Strassen decomposition handles it.
            # The W part of the 2x2 Strassen decomposition gives the coefficients for C.
            # W2[e][r] is the coefficient of C_block[k][i] in the term.
            # So for each k, i in 0..1:
            #   e = k*2 + i
            #   coeff = W2[e][r]
            #   This coeff applies to the block C[k][i].
            #   C[k][i] is a 2x2 matrix.
            #   So W[2k+c][2i+d] gets coeff for c,d in 0..1.
            #   Wait, W index is k*4+i.
            #   Here k is global row, i is global col.
            #   Global row = 2K + c. Global col = 2I + d.
            #   So W[2K+c][2I+d] += W2[K*2+I][r]
            
            for K in range(2):
                for I in range(2):
                    e = K * 2 + I
                    coeff = W2[e][r]
                    if coeff != 0:
                        for c in range(2):
                            for d in range(2):
                                w_idx = (2*K + c) * 4 + (2*I + d)
                                W[w_idx][col] = coeff

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print(res["U"][0])
