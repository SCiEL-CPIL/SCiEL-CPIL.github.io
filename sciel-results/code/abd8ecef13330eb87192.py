"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use Kronecker product of Strassen (2x2, Rank 7) and Schoolbook (3x3, Rank 27)
    # Total Rank: 7 * 27 = 189
    
    # 1. Construct Strassen 2x2 decomposition (Rank 7)
    # U2, V2, W2 are 4x7
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # Indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    # M0: (A00+A11)(B00+B11) -> C00+C11
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # M1: (A10+A11)B00 -> C10-C11
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1; W2[3][1] = -1
    
    # M2: A00(B01-B11) -> C01+C11
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1; W2[3][2] = 1
    
    # M3: A11(B10-B00) -> C00+C10
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[2][3] = 1
    
    # M4: (A00+A01)B11 -> -C00+C01
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[1][4] = 1
    
    # M5: (A10-A00)(B00+B01) -> C11
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # M6: (A01-A11)(B10+B11) -> C00
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1

    # 2. Construct Schoolbook 3x3 decomposition (Rank 27)
    # U3, V3, W3 are 9x27
    U3 = [[0]*27 for _ in range(9)]
    V3 = [[0]*27 for _ in range(9)]
    W3 = [[0]*27 for _ in range(9)]
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                r = i*9 + j*3 + k
                # A index: i*3+j
                # B index: j*3+k
                # C index: k*3+i
                U3[i*3 + j][r] = 1
                V3[j*3 + k][r] = 1
                W3[k*3 + i][r] = 1

    # 3. Construct 6x6 decomposition via Kronecker Product
    rank = 7 * 27
    U = [[0]*rank for _ in range(36)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    # Iterate over all combinations of indices
    for i2 in range(2):
        for j2 in range(2):
            for k2 in range(2):
                for i3 in range(3):
                    for j3 in range(3):
                        for k3 in range(3):
                            # Calculate 6x6 indices
                            i = 2*i2 + i3
                            j = 2*j2 + j3
                            k = 2*k2 + k3
                            
                            u_idx = i*6 + j
                            v_idx = j*6 + k
                            w_idx = k*6 + i
                            
                            # Calculate rank index
                            r = 7*k2 + k3
                            
                            # Get values from smaller decompositions
                            u_val = U2[i2][k2] * U3[i3][k3]
                            v_val = V2[j2][k2] * V3[j3][k3]
                            w_val = W2[k2][k2] * W3[k3][k3] # Wait, W2 index is k2?
                            # W2 is 4x7. Row index is k (output index).
                            # In W2, row k2 corresponds to output index k2.
                            # So W2[k2][k2] is correct for the "diagonal" part?
                            # No, W2 is a matrix.
                            # W2[row, col]. Row is output index. Col is rank index.
                            # So W2[k2, k2_rank] ?
                            # The rank index for the 2x2 part is k2? No.
                            # The rank index is the column index in U2, V2, W2.
                            # Let's call the rank index 'r2' and 'r3'.
                            # r = 7*r2 + r3.
                            
                            # I used k2 as the rank index in the loop above?
                            # No, k2 is an index of the 2x2 matrix (0 or 1).
                            # The rank index for Strassen is 0..6.
                            # I need to iterate over the rank indices of the sub-decompositions.
                            
                            # Let's restart the Kronecker loop structure.
                            pass

    # Correct Kronecker Construction
    for r2 in range(7):
        for r3 in range(27):
            r = 7*r2 + r3
            for i2 in range(2):
                for j2 in range(2):
                    for k2 in range(2):
                        # Get 2x2 values
                        u2_val = U2[i2][r2]
                        v2_val = V2[j2][r2]
                        w2_val = W2[k2][r2]
                        
                        if u2_val == 0 or v2_val == 0 or w2_val == 0:
                            continue
                            
                        for i3 in range(3):
                            for j3 in range(3):
                                for k3 in range(3):
                                    # Get 3x3 values
                                    u3_val = U3[i3][r3]
                                    v3_val = V3[j3][r3]
                                    w3_val = W3[k3][r3]
                                    
                                    if u3_val == 0 or v3_val == 0 or w3_val == 0:
                                        continue
                                    
                                    # Calculate 6x6 indices
                                    i = 2*i2 + i3
                                    j = 2*j2 + j3
                                    k = 2*k2 + k3
                                    
                                    u_idx = i*6 + j
                                    v_idx = j*6 + k
                                    w_idx = k*6 + i
                                    
                                    U[u_idx][r] = u2_val * u3_val
                                    V[v_idx][r] = v2_val * v3_val
                                    W[w_idx][r] = w2_val * w3_val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
