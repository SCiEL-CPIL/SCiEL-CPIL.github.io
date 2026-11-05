"""Tensor decomposition for 6x6 matrix multiplication."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-189 decomposition for 6x6 matrix multiplication.
    
    Strategy: Kronecker product of Strassen 2x2 (rank 7) and schoolbook 3x3 (rank 27).
    6 = 2 * 3, so we decompose the 6x6 multiplication into a 2x2 block structure
    where each block is 3x3, using Strassen for the outer 2x2 and schoolbook for
    the inner 3x3.
    
    The decomposition satisfies:
      (A * B)[i, k] = sum_{j,c} U[i*6+j, c] * V[j*6+k, c] * W[k*6+i, c]
    
    Using Kronecker product with 6 = 2*3:
      i = i2*3 + i3, j = j2*3 + j3, k = k2*3 + k3
      c = c2*27 + c3
      
      U6[i*6+j, c] = U2[i2*2+j2, c2] * U3[i3*3+j3, c3]
      V6[j*6+k, c] = V2[j2*2+k2, c2] * V3[j3*3+k3, c3]
      W6[k*6+i, c] = W2[k2*2+i2, c2] * W3[k3*3+i3, c3]
    """
    
    # --- Strassen 2x2 decomposition (rank 7) ---
    # U2[i*2+j, c], V2[j*2+k, c], W2[k*2+i, c]
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # P1 = (A00+A11)*(B00+B11)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # P2 = (A10+A11)*B00
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[1][1] = 1; W2[3][1] = -1
    
    # P3 = A00*(B01-B11)
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[2][2] = 1; W2[3][2] = 1
    
    # P4 = A11*(B10-B00)
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[1][3] = 1
    
    # P5 = (A00+A01)*B11
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[2][4] = 1
    
    # P6 = (A10-A00)*(B00+B01)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # P7 = (A01-A11)*(B10+B11)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1
    
    # --- Schoolbook 3x3 decomposition (rank 27) ---
    # U3[i*3+j, c], V3[j*3+k, c], W3[k*3+i, c]
    
    U3 = [[0]*27 for _ in range(9)]
    V3 = [[0]*27 for _ in range(9)]
    W3 = [[0]*27 for _ in range(9)]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i*3+j][col] = 1
                V3[j*3+k][col] = 1
                W3[k*3+i][col] = 1
                col += 1
    
    # --- Build 6x6 decomposition via Kronecker product ---
    rank = 7 * 27  # 189
    U6 = [[0]*rank for _ in range(36)]
    V6 = [[0]*rank for _ in range(36)]
    W6 = [[0]*rank for _ in range(36)]
    
    for c2 in range(7):
        for c3 in range(27):
            c = c2 * 27 + c3
            
            # Build U6
            for i2 in range(2):
                for j2 in range(2):
                    u2_val = U2[i2*2+j2][c2]
                    if u2_val == 0:
                        continue
                    for i3 in range(3):
                        for j3 in range(3):
                            u3_val = U3[i3*3+j3][c3]
                            if u3_val == 0:
                                continue
                            i = i2 * 3 + i3
                            j = j2 * 3 + j3
                            U6[i*6+j][c] = u2_val * u3_val
            
            # Build V6
            for j2 in range(2):
                for k2 in range(2):
                    v2_val = V2[j2*2+k2][c2]
                    if v2_val == 0:
                        continue
                    for j3 in range(3):
                        for k3 in range(3):
                            v3_val = V3[j3*3+k3][c3]
                            if v3_val == 0:
                                continue
                            j = j2 * 3 + j3
                            k = k2 * 3 + k3
                            V6[j*6+k][c] = v2_val * v3_val
            
            # Build W6
            for k2 in range(2):
                for i2 in range(2):
                    w2_val = W2[k2*2+i2][c2]
                    if w2_val == 0:
                        continue
                    for k3 in range(3):
                        for i3 in range(3):
                            w3_val = W3[k3*3+i3][c3]
                            if w3_val == 0:
                                continue
                            k = k2 * 3 + k3
                            i = i2 * 3 + i3
                            W6[k*6+i][c] = w2_val * w3_val
    
    return {"U": U6, "V": V6, "W": W6}
# EVOLVE-BLOCK-END
