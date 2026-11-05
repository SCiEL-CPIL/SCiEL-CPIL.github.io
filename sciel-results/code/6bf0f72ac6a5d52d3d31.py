import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Using tensor product of Strassen (2x2, rank 7) and Schoolbook (3x3, rank 27)
    # Total rank = 7 * 27 = 189
    
    # Strassen 2x2 factors
    # Rows correspond to A[i,j], B[j,k], C[i,k] flattened as i*2+j
    U2 = [
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ]
    V2 = [
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 0, 1, 0, 1, 0],  # B01
        [0, 0, 1, 0, 0, 0, 1],  # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ]
    W2 = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    
    # Schoolbook 3x3 factors
    # Rows correspond to A[i,j], B[j,k], C[k,i] flattened
    # Note: W convention is W[k*n+i, r], so for 3x3 it is k*3+i
    U3 = [[0]*27 for _ in range(9)]
    V3 = [[0]*27 for _ in range(9)]
    W3 = [[0]*27 for _ in range(9)]
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                c = i*9 + j*3 + k
                U3[i*3 + j][c] = 1
                V3[j*3 + k][c] = 1
                W3[k*3 + i][c] = 1
                
    rank = 7 * 27
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Tensor product construction
    # Indices:
    # I, J: Block indices for A (0,1)
    # i, j: Intra-block indices for A (0,1,2)
    # Global A row: I*3 + i, Col: J*3 + j
    # U index: (I*3+i)*6 + (J*3+j)
    
    # J, K: Block indices for B (0,1)
    # j, k: Intra-block indices for B (0,1,2)
    # Global B row: J*3 + j, Col: K*3 + k
    # V index: (J*3+j)*6 + (K*3+k)
    
    # I, K: Block indices for C (0,1)
    # i, k: Intra-block indices for C (0,1,2)
    # Global C row: I*3 + i, Col: K*3 + k
    # W index: (K*3+k)*6 + (I*3+i) -- Wait, W convention is W[k*n+i, r]
    # So row index is k_global * 6 + i_global
    # k_global = K*3 + k
    # i_global = I*3 + i
    # W index: (K*3+k)*6 + (I*3+i)
    
    for s in range(7):
        for t in range(27):
            c = s*27 + t
            
            # Fill U
            for I in range(2):
                for J in range(2):
                    u2_val = U2[I*2 + J][s]
                    if u2_val == 0: continue
                    for i in range(3):
                        for j in range(3):
                            u3_val = U3[i*3 + j][t]
                            if u3_val == 0: continue
                            idx = (I*3 + i)*6 + (J*3 + j)
                            U[idx][c] = u2_val * u3_val
                            
            # Fill V
            for J in range(2):
                for K in range(2):
                    v2_val = V2[J*2 + K][s]
                    if v2_val == 0: continue
                    for j in range(3):
                        for k in range(3):
                            v3_val = V3[j*3 + k][t]
                            if v3_val == 0: continue
                            idx = (J*3 + j)*6 + (K*3 + k)
                            V[idx][c] = v2_val * v3_val
                            
            # Fill W
            for I in range(2):
                for K in range(2):
                    w2_val = W2[I*2 + K][s]
                    if w2_val == 0: continue
                    for i in range(3):
                        for k in range(3):
                            w3_val = W3[k*3 + i][t]
                            if w3_val == 0: continue
                            idx = (K*3 + k)*6 + (I*3 + i)
                            W[idx][c] = w2_val * w3_val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
