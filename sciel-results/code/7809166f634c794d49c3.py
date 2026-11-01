"""Rewritten program for 2x2 by 2x14 matrix multiplication tensor decomposition.
Uses Strassen's algorithm on 7 independent 2x2 blocks to achieve rank 49.
"""

def construct():
    n, m, p = 2, 2, 14
    rank = 49  # 7 blocks * 7 Strassen terms
    
    # Strassen's 7 linear forms for 2x2 matrix multiplication:
    # P1 = (A00 + A11)(B00 + B11)
    # P2 = (A10 + A11)B00
    # P3 = A00(B01 - B11)
    # P4 = A11(B10 - B00)
    # P5 = (A00 + A01)B11
    # P6 = (A10 - A00)(B00 + B01)
    # P7 = (A01 - A11)(B10 + B11)
    #
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Coefficients for A (rows of U) for each Strassen term s=0..6
    # A indices: A00=0, A01=1, A10=2, A11=3
    A_coeffs = [
        # s=0: P1 = (A00 + A11)(B00 + B11)
        [1, 0, 0, 1],
        # s=1: P2 = (A10 + A11)B00
        [0, 0, 1, 1],
        # s=2: P3 = A00(B01 - B11)
        [1, 0, 0, 0],
        # s=3: P4 = A11(B10 - B00)
        [0, 0, 0, 1],
        # s=4: P5 = (A00 + A01)B11
        [1, 1, 0, 0],
        # s=5: P6 = (A10 - A00)(B00 + B01)
        [-1, 0, 1, 0],
        # s=6: P7 = (A01 - A11)(B10 + B11)
        [0, 1, 0, -1],
    ]
    
    # Coefficients for B (rows of V) for each Strassen term s=0..6
    # B indices within a 2x2 block: B00=0, B01=1, B10=2, B11=3
    B_coeffs = [
        # s=0: P1 = (A00 + A11)(B00 + B11)
        [1, 0, 0, 1],
        # s=1: P2 = (A10 + A11)B00
        [1, 0, 0, 0],
        # s=2: P3 = A00(B01 - B11)
        [0, 1, 0, -1],
        # s=3: P4 = A11(B10 - B00)
        [-1, 0, 1, 0],
        # s=4: P5 = (A00 + A01)B11
        [0, 0, 0, 1],
        # s=5: P6 = (A10 - A00)(B00 + B01)
        [1, 1, 0, 0],
        # s=6: P7 = (A01 - A11)(B10 + B11)
        [0, 0, 1, 1],
    ]
    
    # Coefficients for C (rows of W) for each Strassen term s=0..6
    # C indices within a 2x2 block: C00=0, C01=1, C10=2, C11=3
    C_coeffs = [
        # s=0: P1
        [1, 0, 0, -1],  # C00 += 1, C01 += 0, C10 += 0, C11 -= 1... wait let me recheck
    ]
    
    # Let me recompute C coefficients carefully:
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    #
    # So for each s, the contribution to each C:
    C_coeffs = [
        # s=0 (P1): C00+=1, C01+=0, C10+=0, C11+=1
        [1, 0, 0, 1],
        # s=1 (P2): C00+=0, C01+=0, C10+=1, C11+=-1
        [0, 0, 1, -1],
        # s=2 (P3): C00+=0, C01+=1, C10+=0, C11+=1
        [0, 1, 0, 1],
        # s=3 (P4): C00+=1, C01+=0, C10+=1, C11+=0
        [1, 0, 1, 0],
        # s=4 (P5): C00+=-1, C01+=1, C10+=0, C11+=0
        [-1, 1, 0, 0],
        # s=5 (P6): C00+=0, C01+=0, C10+=0, C11+=1
        [0, 0, 0, 1],
        # s=6 (P7): C00+=1, C01+=0, C10+=0, C11+=0
        [1, 0, 0, 0],
    ]
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 49
    V = [[0] * rank for _ in range(m * p)]  # 28 x 49
    W = [[0] * rank for _ in range(p * n)]  # 28 x 49
    
    col = 0
    for b in range(7):  # 7 blocks
        for s in range(7):  # 7 Strassen terms
            # U: same A coefficients for all blocks
            for i in range(4):
                U[i][col] = A_coeffs[s][i]
            
            # V: B coefficients mapped to global indices
            # Block b uses B columns 2b and 2b+1
            # B global index: row j (0 or 1), col k (0..13) -> index j*p + k = 14*j + k
            for r in range(4):
                coeff = B_coeffs[s][r]
                if coeff != 0:
                    # r=0: B00 -> B[0][2b] -> index 14*0 + 2b = 2b
                    # r=1: B01 -> B[0][2b+1] -> index 14*0 + (2b+1) = 2b+1
                    # r=2: B10 -> B[1][2b] -> index 14*1 + 2b = 14 + 2b
                    # r=3: B11 -> B[1][2b+1] -> index 14*1 + (2b+1) = 15 + 2b
                    if r == 0:
                        V[2 * b][col] += coeff
                    elif r == 1:
                        V[2 * b + 1][col] += coeff
                    elif r == 2:
                        V[14 + 2 * b][col] += coeff
                    elif r == 3:
                        V[15 + 2 * b][col] += coeff
            
            # W: C coefficients mapped to global indices
            # C global index: W[k*n+i] = W[2*k + i]
            # Block b outputs: C[0][2b], C[0][2b+1], C[1][2b], C[1][2b+1]
            # C[0][2b] -> k=2b, i=0 -> W[2*2b+0] = W[4b]
            # C[0][2b+1] -> k=2b+1, i=0 -> W[2*(2b+1)+0] = W[4b+2]
            # C[1][2b] -> k=2b, i=1 -> W[2*2b+1] = W[4b+1]
            # C[1][2b+1] -> k=2b+1, i=1 -> W[2*(2b+1)+1] = W[4b+3]
            for r in range(4):
                coeff = C_coeffs[s][r]
                if coeff != 0:
                    if r == 0:  # C00 of block -> C[0][2b]
                        W[4 * b][col] += coeff
                    elif r == 1:  # C01 of block -> C[0][2b+1]
                        W[4 * b + 2][col] += coeff
                    elif r == 2:  # C10 of block -> C[1][2b]
                        W[4 * b + 1][col] += coeff
                    elif r == 3:  # C11 of block -> C[1][2b+1]
                        W[4 * b + 3][col] += coeff
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
