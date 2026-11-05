"""Matrix multiplication 4x4x5 with rank-65 decomposition."""
DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = 4, 4, 5
    rank = 65
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * rank for _ in range(m * p)]  # 20 x 65
    W = [[0] * rank for _ in range(p * n)]  # 20 x 65
    
    # Part 1: 4x4x4 decomposition using recursive Strassen (rank 49)
    # Base case: 2x2x2 Strassen (rank 7)
    u2 = [[0]*7 for _ in range(4)]
    v2 = [[0]*7 for _ in range(4)]
    w2 = [[0]*7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    u2[0][0] = 1; u2[3][0] = 1
    v2[0][0] = 1; v2[3][0] = 1
    w2[0][0] = 1; w2[3][0] = 1
    
    # M2 = (A10+A11)B00
    u2[2][1] = 1; u2[3][1] = 1
    v2[0][1] = 1
    w2[2][1] = 1
    
    # M3 = A00(B01-B11)
    u2[0][2] = 1
    v2[1][2] = 1; v2[3][2] = -1
    w2[1][2] = 1
    
    # M4 = A11(B10-B00)
    u2[3][3] = 1
    v2[2][3] = 1; v2[0][3] = -1
    w2[2][3] = 1; w2[0][3] = 1
    
    # M5 = (A00+A01)B11
    u2[0][4] = 1; u2[1][4] = 1
    v2[3][4] = 1
    w2[0][4] = -1; w2[1][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    u2[2][5] = 1; u2[0][5] = -1
    v2[0][5] = 1; v2[1][5] = 1
    w2[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    u2[1][6] = 1; u2[3][6] = -1
    v2[2][6] = 1; v2[3][6] = 1
    w2[0][6] = 1

    # Construct 4x4x4 via Kronecker product
    # Indices:
    # A: I = 2*i + i', J = 2*j + j'
    # B: J = 2*j + j', K = 2*k + k'
    # C: K = 2*k + k', I = 2*i + i'
    
    for s1 in range(7):
        for s2 in range(7):
            col = s1 * 7 + s2
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        for ip in range(2):
                            for jp in range(2):
                                for kp in range(2):
                                    # Global indices
                                    I = 2*i + ip
                                    J = 2*j + jp
                                    K = 2*k + kp
                                    
                                    # Local indices for outer (i, j, k)
                                    # Local indices for inner (ip, jp, kp)
                                    
                                    u_val = u2[i*2+j][s1] * u2[ip*2+jp][s2]
                                    v_val = v2[j*2+k][s1] * v2[jp*2+kp][s2]
                                    w_val = w2[k*2+i][s1] * w2[kp*2+ip][s2]
                                    
                                    if u_val != 0 and v_val != 0 and w_val != 0:
                                        U[I*4+J][col] = u_val
                                        V[J*5+K][col] = v_val
                                        W[K*4+I][col] = w_val
                                    else:
                                        # Even if one is 0, the product is 0, so we can skip or set 0
                                        # But we must ensure we don't miss non-zero contributions
                                        # The loop covers all, so if any factor is 0, the term is 0.
                                        pass
                                        
    # Part 2: 4x4x1 decomposition for the 5th column of B (K=4)
    # Rank 16 schoolbook
    col_offset = 49
    col = col_offset
    for i in range(4):
        for j in range(4):
            I = i
            J = j
            K = 4
            U[I*4+J][col] = 1
            V[J*5+K][col] = 1
            W[K*4+I][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
