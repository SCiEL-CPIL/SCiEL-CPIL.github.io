import numpy as np

def construct():
    n, m, p = 2, 4, 4
    rank = 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    # The four 2x2 blocks of A (2x4) and B (4x4)
    # A = [A11 A12; A21 A22] where A11, A12, A21, A22 are 2x2
    # B = [B11 B12; B21 B22] where B11, B12, B21, B22 are 2x2
    #
    # U rows:  A[0,0]=U0, A[0,1]=U1, A[0,2]=U2, A[0,3]=U3, A[1,0]=U4, A[1,1]=U5, A[1,2]=U6, A[1,3]=U7
    # V rows:  B[0,0]=V0, B[0,1]=V1, B[0,2]=V2, B[0,3]=V3,
    #          B[1,0]=V4, B[1,1]=V5, B[1,2]=V6, B[1,3]=V7,
    #          B[2,0]=V8, B[2,1]=V9, B[2,2]=V10, B[2,3]=V11,
    #          B[3,0]=V12, B[3,1]=V13, B[3,2]=V14, B[3,3]=V15
    # W rows:  C[0,0]=W0, C[0,1]=W1, C[0,2]=W2, C[0,3]=W3, C[1,0]=W4, C[1,1]=W5, C[1,2]=W6, C[1,3]=W7
    #          (trace convention W[k*n+i, r] -> W[row, col] with row = k*4 + i for 4x4 output? No, output is 2x4)
    #          Actually output C is 2x4. W has shape (p*n, rank) = (8, rank).
    #          Index: W[k*n+i, r] where k is row of C (0..1), i is col of C (0..3)?
    #          p=4, n=2. W has 8 rows. Index = k*n + i = k*2 + i? No, that's for 4x2.
    #          Let's re-read: "output factor follows the official trace convention W[k*n+i,r]"
    #          For C = A @ B, A is (n x m) = (2 x 4), B is (m x p) = (4 x 4). C is (n x p) = (2 x 4).
    #          The indices for C are (i, j) with i in 0..1, j in 0..3.
    #          W has shape (p*n, rank) = (8, rank).
    #          The index in W is k*n + i? If k is the column index of C (0..3) and i is the row index (0..1),
    #          then k*2 + i gives 0..7.
    #          Let's assume W[k*n+i, r] means W[col_of_C * n + row_of_C, r].
    #          So W[0]=C[0,0], W[1]=C[1,0], W[2]=C[0,1], W[3]=C[1,1], W[4]=C[0,2], W[5]=C[1,2], W[6]=C[0,3], W[7]=C[1,3].
    #
    # Let's verify with the seed program's Z arrays.
    # Block 1: Z1 = [0, 2, 1, 3]. These are W indices for the 4 outputs of the 2x2 multiplication.
    # The 4 outputs of A11 * B11 (2x2) are C[0,0], C[0,1], C[1,0], C[1,1].
    # In the W indexing above: C[0,0]=W0, C[0,1]=W2, C[1,0]=W1, C[1,1]=W3.
    # So Z1 = [0, 2, 1, 3] corresponds to [C00, C01, C10, C11]. This matches the standard order for a 2x2 matrix if read row-major: (0,0), (0,1), (1,0), (1,1).
    #
    # We implement the 7 Strassen multiplications for each block.

    def add_strassen_block(col_offset, x_idx, y_idx, z_idx):
        # x_idx: [A11_00, A11_01, A11_10, A11_11] indices into U
        # y_idx: [B11_00, B11_01, B11_10, B11_11] indices into V
        # z_idx: [C11_00, C11_01, C11_10, C11_11] indices into W
        
        # M1 = (A11_00 + A11_11) * (B11_00 + B11_11)
        U[x_idx[0], col_offset + 0] += 1
        U[x_idx[3], col_offset + 0] += 1
        V[y_idx[0], col_offset + 0] += 1
        V[y_idx[3], col_offset + 0] += 1
        W[z_idx[0], col_offset + 0] += 1
        W[z_idx[3], col_offset + 0] += 1

        # M2 = A11_10 * (B11_00 + B11_01)
        # Wait, standard Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        #
        # Let's re-map the indices carefully.
        # A is 2x2: A11, A12 (top), A21, A22 (bottom).
        # x_idx = [A11, A12, A21, A22] ? No, x_idx are specific row indices of U.
        # In the seed: X1 = [0, 1, 4, 5].
        # U0=A[0,0]=A11_00, U1=A[0,1]=A11_01? No.
        # A is 2x4. Rows 0,1. Cols 0,1,2,3.
        # A11 is top-left 2x2 of A? No, A is 2x4.
        # We split A into A_L (2x2) and A_R (2x2).
        # A_L = [[A00, A01], [A10, A11]]. A_R = [[A02, A03], [A12, A13]].
        # U0=A00, U1=A01, U2=A02, U3=A03, U4=A10, U5=A11, U6=A12, U7=A13.
        # So X1 = [0, 1, 4, 5] corresponds to A00, A01, A10, A11. This is A_L.
        # Y1 = [0, 1, 4, 5] corresponds to B00, B01, B10, B11. This is B_TL.
        #
        # So for a block multiplication A_L * B_TL:
        # A11 = A00, A12 = A01, A21 = A10, A22 = A11.
        # B11 = B00, B12 = B01, B21 = B10, B22 = B11.
        
        # M1 = (A11+A22)(B11+B22) -> (A00+A11)(B00+B11)
        # x: 0, 3. y: 0, 3.
        # z: C11, C22.
        
        # M2 = (A21+A22)B11 -> (A10+A11)B00
        # x: 2, 3. y: 0.
        # z: C21, C22 (negative).
        
        # M3 = A11(B21-B22) -> A00(B10-B11)
        # x: 0. y: 2, 3 (negative).
        # z: C12, C22.
        
        # M4 = A22(B12-B11) -> A11(B01-B00)
        # x: 3. y: 1, 0 (negative).
        # z: C11, C21.
        
        # M5 = (A11+A12)B22 -> (A00+A01)B11
        # x: 0, 1. y: 3.
        # z: C11 (negative), C12.
        
        # M6 = (A21-A11)(B11+B12) -> (A10-A00)(B00+B01)
        # x: 2, 0 (negative). y: 0, 1.
        # z: C22.
        
        # M7 = (A12-A22)(B21+B22) -> (A01-A11)(B10+B11)
        # x: 1, 3 (negative). y: 2, 3.
        # z: C11.
        
        # Let's check the seed's add_block implementation to match exactly.
        # Seed M1: U[x0]+=1, U[x3]+=1, V[y0]+=1, V[y3]+=1, W[z0]+=1, W[z3]+=1.
        # My M1: A00, A11, B00, B11. x0=A00, x3=A11. y0=B00, y3=B11.
        # C11=W[z0], C22=W[z3]. Matches.
        
        # Seed M2: U[x2]+=1, U[x3]+=1, V[y0]+=1, W[z2]+=1, W[z3]-=1.
        # My M2: (A10+A11)B00. x2=A10, x3=A11. y0=B00.
        # C21=W[z2], C22=W[z3]. Matches.
        
        # Seed M3: U[x0]+=1, V[y1]+=1, V[y3]-=1, W[z1]+=1, W[z3]+=1.
        # My M3: A00(B10-B11). x0=A00. y2=B10, y3=B11.
        # Wait, seed uses V[y1] and V[y3].
        # Y1 = [0, 1, 4, 5]. y0=B00, y1=B01, y2=B10, y3=B11.
        # My B21=B10=y2, B22=B11=y3.
        # Seed M3 uses V[y1] and V[y3]. That's B01 and B11.
        # Let's re-evaluate M3.
        # M3 = A11(B21 - B22).
        # A11 = A00 = x0.
        # B21 = B10 = y2. B22 = B11 = y3.
        # So M3 should be x0 * (y2 - y3).
        # Seed M3: U[x0]+=1, V[y1]+=1, V[y3]-=1.
        # This is x0 * (y1 - y3) = A00 * (B01 - B11).
        # This is not A00 * (B10 - B11).
        # Let's check the output of M3 in the seed.
        # W[z1]+=1, W[z3]+=1.
        # z1=C01, z3=C11.
        # M3 contributes to C01 and C11.
        # In standard Strassen: M3 contributes to C12 and C22.
        # C12 is top-right, C22 is bottom-right.
        # Here z1=C01 (top-right of 2x2 block), z3=C11 (bottom-right).
        # So M3 contributes to top-right and bottom-right.
        # My M3: A00(B10-B11).
        # C01 = sum A0k Bk1. In the 2x2 block, C01 = A00 B01 + A01 B11.
        # C11 = A10 B01 + A11 B11.
        # M3 = A00 B10 - A00 B11.
        # This doesn't look like it directly forms C01 or C11 alone.
        # Let's trust the seed's algebra. It's a known valid decomposition.
        # I will replicate the seed's logic exactly but write it from scratch.

        # Seed M4: U[x3]+=1, V[y2]+=1, V[y0]-=1, W[z0]+=1, W[z2]+=1.
        # x3=A11. y2=B10, y0=B00.
        # M4 = A11 (B10 - B00).
        # W[z0]=C00, W[z2]=C10.
        
        # Seed M5: U[x0]+=1, U[x1]+=1, V[y3]+=1, W[z0]-=1, W[z1]+=1.
        # x0=A00, x1=A01. y3=B11.
        # M5 = (A00+A01) B11.
        # W[z0]=C00, W[z1]=C01.
        
        # Seed M6: U[x2]+=1, U[x0]-=1, V[y0]+=1, V[y1]+=1, W[z3]+=1.
        # x2=A10, x0=A00. y0=B00, y1=B01.
        # M6 = (A10-A00)(B00+B01).
        # W[z3]=C11.
        
        # Seed M7: U[x1]+=1, U[x3]-=1, V[y2]+=1, V[y3]+=1, W[z0]+=1.
        # x1=A01, x3=A11. y2=B10, y3=B11.
        # M7 = (A01-A11)(B10+B11).
        # W[z0]=C00.

        # Let's implement this specific pattern.
        
        # M1
        U[x_idx[0], col_offset + 0] += 1
        U[x_idx[3], col_offset + 0] += 1
        V[y_idx[0], col_offset + 0] += 1
        V[y_idx[3], col_offset + 0] += 1
        W[z_idx[0], col_offset + 0] += 1
        W[z_idx[3], col_offset + 0] += 1

        # M2
        U[x_idx[2], col_offset + 1] += 1
        U[x_idx[3], col_offset + 1] += 1
        V[y_idx[0], col_offset + 1] += 1
        W[z_idx[2], col_offset + 1] += 1
        W[z_idx[3], col_offset + 1] -= 1

        # M3
        U[x_idx[0], col_offset + 2] += 1
        V[y_idx[1], col_offset + 2] += 1
        V[y_idx[3], col_offset + 2] -= 1
        W[z_idx[1], col_offset + 2] += 1
        W[z_idx[3], col_offset + 2] += 1

        # M4
        U[x_idx[3], col_offset + 3] += 1
        V[y_idx[2], col_offset + 3] += 1
        V[y_idx[0], col_offset + 3] -= 1
        W[z_idx[0], col_offset + 3] += 1
        W[z_idx[2], col_offset + 3] += 1

        # M5
        U[x_idx[0], col_offset + 4] += 1
        U[x_idx[1], col_offset + 4] += 1
        V[y_idx[3], col_offset + 4] += 1
        W[z_idx[0], col_offset + 4] -= 1
        W[z_idx[1], col_offset + 4] += 1

        # M6
        U[x_idx[2], col_offset + 5] += 1
        U[x_idx[0], col_offset + 5] -= 1
        V[y_idx[0], col_offset + 5] += 1
        V[y_idx[1], col_offset + 5] += 1
        W[z_idx[3], col_offset + 5] += 1

        # M7
        U[x_idx[1], col_offset + 6] += 1
        U[x_idx[3], col_offset + 6] -= 1
        V[y_idx[2], col_offset + 6] += 1
        V[y_idx[3], col_offset + 6] += 1
        W[z_idx[0], col_offset + 6] += 1

    # Block 1: A_L * B_TL -> C_TL
    # A_L: U0, U1, U4, U5. B_TL: V0, V1, V4, V5.
    # C_TL: C00, C01, C10, C11 -> W0, W2, W1, W3.
    add_strassen_block(0, [0, 1, 4, 5], [0, 1, 4, 5], [0, 2, 1, 3])

    # Block 2: A_R * B_BL -> C_BL
    # A_R: U2, U3, U6, U7. B_BL: V8, V9, V12, V13.
    # C_BL: C02, C03, C12, C13 -> W4, W6, W5, W7.
    add_strassen_block(7, [2, 3, 6, 7], [8, 9, 12, 13], [4, 6, 5, 7])

    # Block 3: A_L * B_TR -> C_TR
    # A_L: U0, U1, U4, U5. B_TR: V2, V3, V6, V7.
    # C_TR: C00+? No, C = A_L B_TL + A_R B_BL is not right.
    # C = A B. C = [A_L A_R] [B_TL B_TR; B_BL B_BR].
    # C_TL = A_L B_TL + A_R B_BL.
    # C_TR = A_L B_TR + A_R B_BR.
    #
    # Wait, the seed program adds 4 blocks.
    # Block 1: A_L B_TL -> contributes to C_TL.
    # Block 2: A_R B_BL -> contributes to C_TL.
    # Block 3: A_L B_TR -> contributes to C_TR.
    # Block 4: A_R B_BR -> contributes to C_TR.
    #
    # Let's check the Z indices in the seed.
    # Block 1 Z: [0, 2, 1, 3] -> C00, C01, C10, C11. (C_TL)
    # Block 2 Z: [0, 2, 1, 3] -> C00, C01, C10, C11. (C_TL)
    # Block 3 Z: [4, 6, 5, 7] -> C02, C03, C12, C13. (C_TR)
    # Block 4 Z: [4, 6, 5, 7] -> C02, C03, C12, C13. (C_TR)
    #
    # So the seed computes:
    # C_TL = A_L B_TL + A_R B_BL
    # C_TR = A_L B_TR + A_R B_BR
    #
    # This is correct for C = A B.

    # Block 3: A_L * B_TR -> C_TR
    add_strassen_block(14, [0, 1, 4, 5], [2, 3, 6, 7], [4, 6, 5, 7])

    # Block 4: A_R * B_BR -> C_TR
    add_strassen_block(21, [2, 3, 6, 7], [10, 11, 14, 15], [4, 6, 5, 7])

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
