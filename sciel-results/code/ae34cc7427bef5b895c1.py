import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    # Decompose M_{4,4,6} as M_{2,2,3} (x) M_{2,2,2}
    # Rank of M_{2,2,2} is 7 (Strassen)
    # Rank of M_{2,2,3} is 12 (Schoolbook)
    # Total Rank = 7 * 12 = 84
    
    # Strassen decomposition for M_{2,2,2}
    # U, V, W for M_{2,2,2}
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = [[c11, c12], [c21, c22]]
    
    # Strassen 7 multiplications:
    # m1 = (a11 + a22)(b11 + b22)
    # m2 = (a21 + a22)b11
    # m3 = a11(b12 - b22)
    # m4 = a22(b21 - b11)
    # m5 = (a11 + a12)b22
    # m6 = (a21 - a11)(b11 + b12)
    # m7 = (a12 - a22)(b21 + b22)
    # c11 = m1 + m4 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m6
    
    # We need U, V, W matrices for M_{2,2,2}
    # U[i, r], V[j, r], W[k, r] where i,j in 0..1, k in 0..1
    # Indexing: U[a_idx], V[b_idx], W[c_idx]
    # a_idx = i*2 + j
    # b_idx = j*2 + k
    # c_idx = i*2 + k
    
    strassen_U = [
        [1, 0, 1, 0, 1, 0, 0], # a11
        [0, 0, 0, 0, 1, 1, -1], # a12
        [0, 1, 0, 0, 0, -1, 0], # a21
        [1, 1, 0, 1, 0, 0, -1]  # a22
    ]
    strassen_V = [
        [1, 1, 0, -1, 0, 1, 1], # b11
        [0, 0, 1, 0, 0, 1, 0], # b12
        [1, 0, -1, 0, 1, 0, 1], # b21
        [1, 0, 0, 1, 1, 0, 1]  # b22
    ]
    strassen_W = [
        [1, 0, 0, 0, -1, 0, 1], # c11
        [0, 0, 1, 0, 1, 0, 0], # c12
        [0, 1, 0, 1, 0, 0, 0], # c21
        [1, -1, 0, 0, 0, 1, 0]  # c22
    ]
    
    # Schoolbook decomposition for M_{2,2,3}
    # Rank 12
    # A (2x2), B (2x3), C (2x3)
    # a_idx = i*2 + j (0..3)
    # b_idx = j*3 + k (0..5)
    # c_idx = i*3 + k (0..5)
    
    sb_U = []
    sb_V = []
    sb_W = []
    for i in range(2):
        for j in range(2):
            for k in range(3):
                col = [0]*12
                col[i*2+j] = 1
                sb_U.append(col)
                col = [0]*12
                col[j*3+k] = 1
                sb_V.append(col)
                col = [0]*12
                col[i*3+k] = 1
                sb_W.append(col)
                
    # Wait, the loop above appends 12 columns to U, V, W.
    # U should be 4x12, V 6x12, W 6x12.
    # The loop creates 12 vectors of length 4, 6, 6.
    # So U is 12x4? No, U is (n*m, rank) = (4, 12).
    # So I need to transpose or construct correctly.
    
    sb_U = [[0]*12 for _ in range(4)]
    sb_V = [[0]*12 for _ in range(6)]
    sb_W = [[0]*12 for _ in range(6)]
    
    col_idx = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                sb_U[i*2+j][col_idx] = 1
                sb_V[j*3+k][col_idx] = 1
                sb_W[i*3+k][col_idx] = 1
                col_idx += 1
                
    # Now combine via Kronecker Product
    # M_{4,4,6} = M_{2,2,3} (x) M_{2,2,2}
    # U_total[i4, r] = U_223[i2, r1] * U_222[i2', r2]
    # where i4 = i2 * 2 + i2'
    # r = r1 * 7 + r2
    
    rank_total = 12 * 7
    U_total = [[0]*rank_total for _ in range(4)]
    V_total = [[0]*rank_total for _ in range(4)] # Wait, V is m*p = 4*6 = 24?
    # M_{4,4,6}: A is 4x4, B is 4x6.
    # U shape: (4*4, rank) = (16, rank)
    # V shape: (4*6, rank) = (24, rank)
    # W shape: (6*4, rank) = (24, rank)
    
    # My sb_V is 6x12 (2x3 matrix).
    # My strassen_V is 4x7 (2x2 matrix).
    # Combined V index: (j2, j2') for middle dim?
    # Middle dim of M_{2,2,3} is 2. Middle dim of M_{2,2,2} is 2.
    # Combined middle dim is 4.
    # B index in M_{2,2,3}: (j, k) where j in 0..1, k in 0..2.
    # B index in M_{2,2,2}: (j', k') where j' in 0..1, k' in 0..1.
    # Combined B index: (j, j', k, k').
    # Middle index (input to A): (j, j'). Size 4.
    # Output index (output of B): (k, k'). Size 6.
    # So V_total index is (j, j', k, k').
    # j, j' in 0..1. k in 0..2, k' in 0..1.
    # V_total[ (j*2+j')*6 + (k*2+k') ]?
    # Let's map the indices carefully.
    
    # U_total index: (i, i') where i, i' in 0..1.
    # U_total[ i*2 + i' ]
    
    # V_total index: (j, j', k, k')
    # j, j' in 0..1. k in 0..2. k' in 0..1.
    # V_total[ (j*2 + j')*6 + (k*2 + k') ]
    
    # W_total index: (i, i', k, k')
    # W_total[ (k*2 + k')*4 + (i*2 + i') ]?
    # W is p*n = 6*4 = 24.
    # W index in M_{2,2,3}: (i, k) where i in 0..1, k in 0..2.
    # W index in M_{2,2,2}: (i', k') where i' in 0..1, k' in 0..1.
    # Combined W index: (k, k', i, i').
    # W_total[ (k*2 + k')*4 + (i*2 + i') ]
    
    U_total = [[0]*rank_total for _ in range(16)]
    V_total = [[0]*rank_total for _ in range(24)]
    W_total = [[0]*rank_total for _ in range(24)]
    
    for r1 in range(12):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i in range(2):
                for i2 in range(2):
                    u_val = sb_U[i*2 + 0][r1] * strassen_U[i2][r2] # Wait, sb_U index is i*2+j.
                    # sb_U is 4x12. Index is i*2+j.
                    # strassen_U is 4x7. Index is i2*2+j2.
                    # Combined i4 = i*2 + i2?
                    # A index in M_{2,2,3}: (i, j).
                    # A index in M_{2,2,2}: (i2, j2).
                    # Combined A index: (i, i2, j, j2).
                    # U_total index: (i*2+i2)*4 + (j*2+j2)?
                    # U shape 16.
                    # Let's iterate over all indices.
                    
    # This manual construction is error-prone. I will use a simpler loop.
    
    U_total = [[0]*rank_total for _ in range(16)]
    V_total = [[0]*rank_total for _ in range(24)]
    W_total = [[0]*rank_total for _ in range(24)]
    
    for r1 in range(12):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i in range(2):
                for j in range(2):
                    for k in range(3):
                        # sb parts
                        u1 = sb_U[i*2+j][r1]
                        v1 = sb_V[j*3+k][r1]
                        w1 = sb_W[i*3+k][r1]
                        
                        for i2 in range(2):
                            for j2 in range(2):
                                for k2 in range(2):
                                    # strassen parts
                                    u2 = strassen_U[i2*2+j2][r2]
                                    v2 = strassen_V[j2*2+k2][r2]
                                    w2 = strassen_W[i2*2+k2][r2]
                                    
                                    # Combined indices
                                    u_idx = (i*2 + i2)*4 + (j*2 + j2)
                                    v_idx = (j*2 + j2)*6 + (k*2 + k2)
                                    w_idx = (k*2 + k2)*4 + (i*2 + i2)
                                    
                                    U_total[u_idx][r] = u1 * u2
                                    V_total[v_idx][r] = v1 * v2
                                    W_total[w_idx][r] = w1 * w2

    return {"U": U_total, "V": V_total, "W": W_total}
