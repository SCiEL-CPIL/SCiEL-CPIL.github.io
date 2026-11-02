"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use the tensor product of Strassen's M2 (rank 7) and Schoolbook M3 (rank 27)
    # This gives a rank of 7 * 27 = 189 for M6.
    
    # Define Strassen M2 decomposition
    # U2[i, j, a], V2[j, k, a], W2[i, k, a]
    # a in 0..6
    U2 = [[0]*7 for _ in range(4)] # 2x2 matrix, flattened
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # Term 0: M1 = (A00+A11)(B00+B11)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # Term 1: M2 = (A10+A11)B00
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1; W2[3][1] = 1 # C10 = M2+M4, C11 = M1-M2+... 
    # Wait, C10 gets M2, C11 gets -M2.
    W2[2][1] = 1
    W2[3][1] = -1
    
    # Term 2: M3 = A00(B10-B11)
    U2[0][2] = 1
    V2[2][2] = 1; V2[3][2] = -1
    W2[1][2] = 1 # C01 = M3+M5
    # C11 gets M3? No, C11 = M1 - M2 + M3 + M6.
    W2[3][2] = 1
    
    # Term 3: M4 = A11(B01-B00)
    U2[3][3] = 1
    V2[1][3] = 1; V2[0][3] = -1
    W2[0][3] = 1 # C00 = M1+M4...
    W2[2][3] = 1 # C10 = M2+M4
    
    # Term 4: M5 = (A00+A01)B11
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1 # C00 = ... -M5
    W2[1][4] = 1 # C01 = M3+M5
    
    # Term 5: M6 = (A10-A00)(B00+B01)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1 # C11 = ... +M6
    
    # Term 6: M7 = (A01-A11)(B10+B11)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1 # C00 = ... +M7
    
    # Define Schoolbook M3 decomposition
    # U3[i, j, b], V3[j, k, b], W3[i, k, b]
    # b in 0..26
    U3 = [[0]*27 for _ in range(9)]
    V3 = [[0]*27 for _ in range(9)]
    W3 = [[0]*27 for _ in range(9)]
    for i in range(3):
        for j in range(3):
            b = i * 3 + j
            U3[i * 3 + j][b] = 1
            V3[j * 3 + 0][b] = 0 # Placeholder, will fill below
            # Actually, V3 index is j*p+k. Here p=3.
            # So V3[j*3+k][b] = 1 if b corresponds to j,k.
            # In schoolbook, term b uses A_ij and B_jk.
            # So for a fixed b, i,j are fixed for U, and j,k are fixed for V.
            # But b iterates over all i,j,k? No, schoolbook has 27 terms, one for each (i,j,k).
            # Wait, schoolbook M3 has rank 27.
            # Term (i,j,k): U[i,j]=1, V[j,k]=1, W[i,k]=1.
            # So b = i*9 + j*3 + k?
            # Let's map b to (i,j,k).
            # b = i*9 + j*3 + k.
            pass
            
    # Re-do Schoolbook M3
    U3 = [[0]*27 for _ in range(9)]
    V3 = [[0]*27 for _ in range(9)]
    W3 = [[0]*27 for _ in range(9)]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                b = i * 9 + j * 3 + k
                U3[i * 3 + j][b] = 1
                V3[j * 3 + k][b] = 1
                W3[i * 3 + k][b] = 1

    # Combine M2 and M3
    rank = 7 * 27
    U = [[0] * rank for _ in range(36)] # 6x6 = 36
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    for a in range(7):
        for b in range(27):
            c = a * 27 + b
            # Update U
            for i2 in range(2):
                for j2 in range(2):
                    u2_val = U2[i2 * 2 + j2][a]
                    if u2_val != 0:
                        for i3 in range(3):
                            for j3 in range(3):
                                u3_val = U3[i3 * 3 + j3][b]
                                if u3_val != 0:
                                    i = i2 * 3 + i3
                                    j = j2 * 3 + j3
                                    U[i * 6 + j][c] += u2_val * u3_val
            # Update V
            for j2 in range(2):
                for k2 in range(2):
                    v2_val = V2[j2 * 2 + k2][a]
                    if v2_val != 0:
                        for j3 in range(3):
                            for k3 in range(3):
                                v3_val = V3[j3 * 3 + k3][b]
                                if v3_val != 0:
                                    j = j2 * 3 + j3
                                    k = k2 * 3 + k3
                                    V[j * 6 + k][c] += v2_val * v3_val
            # Update W
            for i2 in range(2):
                for k2 in range(2):
                    w2_val = W2[i2 * 2 + k2][a]
                    if w2_val != 0:
                        for i3 in range(3):
                            for k3 in range(3):
                                w3_val = W3[i3 * 3 + k3][b]
                                if w3_val != 0:
                                    i = i2 * 3 + i3
                                    k = k2 * 3 + k3
                                    W[k * 6 + i][c] += w2_val * w3_val
                                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
