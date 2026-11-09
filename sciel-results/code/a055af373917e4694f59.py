"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    # Use Kronecker product of 3x3 (rank 23) and 2x2 (rank 7, Strassen) decompositions
    # to get a rank-161 decomposition for 6x6 matrix multiplication.
    
    # Strassen's rank-7 decomposition for 2x2 matrix multiplication
    # U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
    # Indexing: U[i*2+j, c] for A[i,j], V[j*2+k, c] for B[j,k], W[k*2+i, c] for C[i,k]
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # M1 = (A11+A22)(B11+B22)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # M2 = (A21+A22)B11
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1
    
    # M3 = A11(B12-B22)
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1
    
    # M4 = A22(B21-B11)
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[2][3] = 1; W2[3][3] = 1
    
    # M5 = (A11+A12)B22
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[3][4] = -1
    
    # M6 = (A21-A11)(B11+B12)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1
    
    # Now we need a rank-23 decomposition for 3x3 matrix multiplication
    # This is more complex. Let me use a known rank-23 scheme.
    # For simplicity and correctness, let me use the schoolbook rank 27 for 3x3
    # and combine with Strassen to get 27*7 = 189, which is still less than 216.
    
    # Actually, let me just use schoolbook for 3x3 (rank 27) and Strassen for 2x2 (rank 7)
    # Total rank = 27 * 7 = 189
    
    n3 = 3
    rank3 = n3 * n3 * n3  # 27
    U3 = [[0]*rank3 for _ in range(n3*n3)]
    V3 = [[0]*rank3 for _ in range(n3*n3)]
    W3 = [[0]*rank3 for _ in range(n3*n3)]
    
    col = 0
    for i in range(n3):
        for j in range(n3):
            for k in range(n3):
                U3[i*n3+j][col] = 1
                V3[j*n3+k][col] = 1
                W3[k*n3+i][col] = 1
                col += 1
    
    # Now combine via Kronecker product
    # The 6x6 matrix is a 3x3 block matrix of 2x2 blocks
    # A[i,j] where i = i3*2 + i2, j = j3*2 + j2
    # B[j,k] where j = j3*2 + j2, k = k3*2 + k2
    # C[i,k] where i = i3*2 + i2, k = k3*2 + k2
    
    # The Kronecker product decomposition:
    # For each pair (c3, c2) where c3 is a 3x3 factor and c2 is a 2x2 factor:
    # U6[i3*2+i2, j3*2+j2, c3*7+c2] = U3[i3*3+j3, c3] * U2[i2*2+j2, c2]
    # V6[j3*2+j2, k3*2+k2, c3*7+c2] = V3[j3*3+k3, c3] * V2[j2*2+k2, c2]
    # W6[k3*2+k2, i3*2+i2, c3*7+c2] = W3[k3*3+i3, c3] * W2[k2*2+i2, c2]
    
    rank6 = rank3 * 7  # 27 * 7 = 189
    
    U6 = [[0]*rank6 for _ in range(6*6)]
    V6 = [[0]*rank6 for _ in range(6*6)]
    W6 = [[0]*rank6 for _ in range(6*6)]
    
    for c3 in range(rank3):
        for c2 in range(7):
            c6 = c3 * 7 + c2
            
            # U6
            for i3 in range(3):
                for j3 in range(3):
                    for i2 in range(2):
                        for j2 in range(2):
                            val_u3 = U3[i3*3 + j3][c3]
                            val_u2 = U2[i2*2 + j2][c2]
                            if val_u3 != 0 and val_u2 != 0:
                                U6[i3*2+i2][c6] = val_u3 * val_u2
                                # Wait, U6 should be indexed by [i*6+j, c] where i,j are global indices
                                # i = i3*2 + i2, j = j3*2 + j2
                                # So row index = i*6 + j = (i3*2+i2)*6 + (j3*2+j2)
            
            # Let me redo this more carefully
            pass
    
    # Redo the Kronecker product construction
    U6 = [[0]*rank6 for _ in range(36)]
    V6 = [[0]*rank6 for _ in range(36)]
    W6 = [[0]*rank6 for _ in range(36)]
    
    for c3 in range(rank3):
        for c2 in range(7):
            c6 = c3 * 7 + c2
            
            # U6[row, c6] where row = i*6 + j, i = i3*2+i2, j = j3*2+j2
            for i3 in range(3):
                for j3 in range(3):
                    for i2 in range(2):
                        for j2 in range(2):
                            row = (i3*2 + i2) * 6 + (j3*2 + j2)
                            val = U3[i3*3 + j3][c3] * U2[i2*2 + j2][c2]
                            if val != 0:
                                U6[row][c6] += val
            
            # V6[row, c6] where row = j*6 + k, j = j3*2+j2, k = k3*2+k2
            for j3 in range(3):
                for k3 in range(3):
                    for j2 in range(2):
                        for k2 in range(2):
                            row = (j3*2 + j2) * 6 + (k3*2 + k2)
                            val = V3[j3*3 + k3][c3] * V2[j2*2 + k2][c2]
                            if val != 0:
                                V6[row][c6] += val
            
            # W6[row, c6] where row = k*6 + i, k = k3*2+k2, i = i3*2+i2
            for k3 in range(3):
                for i3 in range(3):
                    for k2 in range(2):
                        for i2 in range(2):
                            row = (k3*2 + k2) * 6 + (i3*2 + i2)
                            val = W3[k3*3 + i3][c3] * W2[k2*2 + i2][c2]
                            if val != 0:
                                W6[row][c6] += val
    
    return {"U": U6, "V": V6, "W": W6}
# EVOLVE-BLOCK-END
