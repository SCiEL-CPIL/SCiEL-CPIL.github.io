DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define linear forms for A and B blocks
    # A00, A01, A10, A11
    # B00, B01, B10, B11
    
    def make_lf(i, j, coeff=1):
        return {(i, j): coeff}
        
    def add_lf(lf1, lf2):
        res = lf1.copy()
        for k, v in lf2.items():
            res[k] = res.get(k, 0) + v
        return res
        
    def sub_lf(lf1, lf2):
        res = lf1.copy()
        for k, v in lf2.items():
            res[k] = res.get(k, 0) - v
        return res
        
    def make_2x2(lf00, lf01, lf10, lf11):
        return [[lf00, lf01], [lf10, lf11]]
        
    def add_2x2(M1, M2):
        return make_2x2(
            add_lf(M1[0][0], M2[0][0]),
            add_lf(M1[0][1], M2[0][1]),
            add_lf(M1[1][0], M2[1][0]),
            add_lf(M1[1][1], M2[1][1])
        )
        
    def sub_2x2(M1, M2):
        return make_2x2(
            sub_lf(M1[0][0], M2[0][0]),
            sub_lf(M1[0][1], M2[0][1]),
            sub_lf(M1[1][0], M2[1][0]),
            sub_lf(M1[1][1], M2[1][1])
        )
        
    # A blocks
    A00 = make_2x2(make_lf(0,0), make_lf(0,1), make_lf(1,0), make_lf(1,1))
    A01 = make_2x2(make_lf(0,2), make_lf(0,3), make_lf(1,2), make_lf(1,3))
    A10 = make_2x2(make_lf(2,0), make_lf(2,1), make_lf(3,0), make_lf(3,1))
    A11 = make_2x2(make_lf(2,2), make_lf(2,3), make_lf(3,2), make_lf(3,3))
    
    # B blocks
    B00 = make_2x2(make_lf(0,0), make_lf(0,1), make_lf(1,0), make_lf(1,1))
    B01 = make_2x2(make_lf(0,2), make_lf(0,3), make_lf(1,2), make_lf(1,3))
    B10 = make_2x2(make_lf(2,0), make_lf(2,1), make_lf(3,0), make_lf(3,1))
    B11 = make_2x2(make_lf(2,2), make_lf(2,3), make_lf(3,2), make_lf(3,3))
    
    # 4x4 Strassen products P1..P7
    # Each Pk is a pair (Mk, Nk) of 2x2 matrices of linear forms
    P1 = (add_2x2(A00, A11), add_2x2(B00, B11))
    P2 = (add_2x2(A10, A11), B00)
    P3 = (A00, sub_2x2(B01, B11))
    P4 = (A11, sub_2x2(B10, B00))
    P5 = (add_2x2(A00, A01), B11)
    P6 = (sub_2x2(A10, A00), add_2x2(B00, B01))
    P7 = (sub_2x2(A01, A11), add_2x2(B10, B11))
    
    # 2x2 Strassen products for each Pk
    # Q1 = (M00 + M11)(N00 + N11)
    # Q2 = (M10 + M11)N00
    # Q3 = M00(N01 - N11)
    # Q4 = M11(N10 - N00)
    # Q5 = (M00 + M01)N11
    # Q6 = (M10 - M00)(N00 + N01)
    # Q7 = (M01 - M11)(N10 + N11)
    
    def get_2x2_products(M, N):
        Q1 = (add_lf(M[0][0], M[1][1]), add_lf(N[0][0], N[1][1]))
        Q2 = (add_lf(M[1][0], M[1][1]), N[0][0])
        Q3 = (M[0][0], sub_lf(N[0][1], N[1][1]))
        Q4 = (M[1][1], sub_lf(N[1][0], N[0][0]))
        Q5 = (add_lf(M[0][0], M[0][1]), N[1][1])
        Q6 = (sub_lf(M[1][0], M[0][0]), add_lf(N[0][0], N[0][1]))
        Q7 = (sub_lf(M[0][1], M[1][1]), add_lf(N[1][0], N[1][1]))
        return [Q1, Q2, Q3, Q4, Q5, Q6, Q7]
        
    # Coefficients for Pk[i,j] in terms of Qk,l
    # P[0,0] = Q1 + Q4 - Q5 + Q7
    # P[0,1] = Q3 + Q5
    # P[1,0] = Q2 + Q4
    # P[1,1] = Q1 - Q2 + Q3 + Q6
    
    # Coefficients for C[i,j] in terms of Pk
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # We will iterate over k=1..7, l=1..7
    # For each (k,l), we have a product Qk,l = (U_lf, V_lf)
    # We need to add U_lf to U column r, V_lf to V column r
    # And determine the contribution to C'
    
    # Contribution of Qk,l to C[i,j]
    # First, contribution to Pk[i',j']
    # Then contribution to C[i,j]
    
    # Let's precompute the contribution of each (k,l) to C[i,j]
    # contrib[k][l][i][j] = coefficient
    
    contrib = {}
    for k in range(7):
        contrib[k] = {}
        for l in range(7):
            contrib[k][l] = [[0]*4 for _ in range(4)]
            
    # Pk[i',j'] coefficients for Qk,l
    # l=0 (Q1): P[0,0]+=1, P[1,1]+=1
    # l=1 (Q2): P[1,0]+=1, P[1,1]-=1
    # l=2 (Q3): P[0,1]+=1, P[1,1]+=1
    # l=3 (Q4): P[0,0]+=1, P[1,0]+=1
    # l=4 (Q5): P[0,0]-=1, P[0,1]+=1
    # l=5 (Q6): P[1,1]+=1
    # l=6 (Q7): P[0,0]+=1
    
    pk_contrib = {
        0: [(0,0,1), (1,1,1)],
        1: [(1,0,1), (1,1,-1)],
        2: [(0,1,1), (1,1,1)],
        3: [(0,0,1), (1,0,1)],
        4: [(0,0,-1), (0,1,1)],
        5: [(1,1,1)],
        6: [(0,0,1)]
    }
    
    # C[i,j] coefficients for Pk
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # C[0,0] = P1[0,0] + P4[0,0] - P5[0,0] + P7[0,0]
    # C[0,1] = P1[0,1] + P4[0,1] - P5[0,1] + P7[0,1]
    # C[0,2] = P3[0,0] + P5[0,0]
    # C[0,3] = P3[0,1] + P5[0,1]
    # C[1,0] = P1[1,0] + P4[1,0] - P5[1,0] + P7[1,0]
    # C[1,1] = P1[1,1] + P4[1,1] - P5[1,1] + P7[1,1]
    # C[1,2] = P3[1,0] + P5[1,0]
    # C[1,3] = P3[1,1] + P5[1,1]
    # C[2,0] = P2[0,0] + P4[0,0]
    # C[2,1] = P2[0,1] + P4[0,1]
    # C[2,2] = P1[0,0] - P2[0,0] + P3[0,0] + P6[0,0]
    # C[2,3] = P1[0,1] - P2[0,1] + P3[0,1] + P6[0,1]
    # C[3,0] = P2[1,0] + P4[1,0]
    # C[3,1] = P2[1,1] + P4[1,1]
    # C[3,2] = P1[1,0] - P2[1,0] + P3[1,0] + P6[1,0]
    # C[3,3] = P1[1,1] - P2[1,1] + P3[1,1] + P6[1,1]
    
    # We can build a dictionary c_contrib[i][j] = list of (k, i', j', coeff)
    # Then for each (k,l), we add pk_contrib[l] to c_contrib[i][j]
    
    c_contrib = {}
    for i in range(4):
        c_contrib[i] = {}
        for j in range(4):
            c_contrib[i][j] = []
            
    # C00
    c_contrib[0][0] += [(0,0,0,1), (3,0,0,1), (4,0,0,-1), (6,0,0,1)]
    c_contrib[0][1] += [(0,0,1,1), (3,0,1,1), (4,0,1,-1), (6,0,1,1)]
    c_contrib[1][0] += [(0,1,0,1), (3,1,0,1), (4,1,0,-1), (6,1,0,1)]
    c_contrib[1][1] += [(0,1,1,1), (3,1,1,1), (4,1,1,-1), (6,1,1,1)]
    # C01
    c_contrib[0][2] += [(2,0,0,1), (4,0,0,1)]
    c_contrib[0][3] += [(2,0,1,1), (4,0,1,1)]
    c_contrib[1][2] += [(2,1,0,1), (4,1,0,1)]
    c_contrib[1][3] += [(2,1,1,1), (4,1,1,1)]
    # C10
    c_contrib[2][0] += [(1,0,0,1), (3,0,0,1)]
    c_contrib[2][1] += [(1,0,1,1), (3,0,1,1)]
    c_contrib[3][0] += [(1,1,0,1), (3,1,0,1)]
    c_contrib[3][1] += [(1,1,1,1), (3,1,1,1)]
    # C11
    c_contrib[2][2] += [(0,0,0,1), (1,0,0,-1), (2,0,0,1), (5,0,0,1)]
    c_contrib[2][3] += [(0,0,1,1), (1,0,1,-1), (2,0,1,1), (5,0,1,1)]
    c_contrib[3][2] += [(0,1,0,1), (1,1,0,-1), (2,1,0,1), (5,1,0,1)]
    c_contrib[3][3] += [(0,1,1,1), (1,1,1,-1), (2,1,1,1), (5,1,1,1)]
    
    # Now compute contrib[k][l][i][j]
    for i in range(4):
        for j in range(4):
            for (k, i_prime, j_prime, coeff) in c_contrib[i][j]:
                for (i2, j2, coeff2) in pk_contrib[l]: # Wait, l is not fixed here
                    pass # This logic is wrong. We need to iterate over l.
                    
    # Let's do it differently.
    # For each k, l:
    #   For each (i_prime, j_prime, coeff2) in pk_contrib[l]:
    #     For each (k2, i2, j2, coeff3) in c_contrib[i][j]:
    #       If k2 == k:
    #         contrib[k][l][i][j] += coeff2 * coeff3
    # But c_contrib[i][j] is a list of (k, i_prime, j_prime, coeff).
    # So for a fixed (k,l), we look at all (i,j) and add to contrib[k][l][i][j].
    
    for k in range(7):
        for l in range(7):
            for (i_prime, j_prime, coeff2) in pk_contrib[l]:
                for i in range(4):
                    for j in range(4):
                        for (k2, i2, j2, coeff3) in c_contrib[i][j]:
                            if k2 == k and i2 == i_prime and j2 == j_prime:
                                contrib[k][l][i][j] += coeff2 * coeff3
                                
    # Now fill U, V, W
    col = 0
    for k in range(7):
        M, N = [P1, P2, P3, P4, P5, P6, P7][k]
        products = get_2x2_products(M, N)
        for l in range(7):
            U_lf, V_lf = products[l]
            # Fill U
            for (i, j), coeff in U_lf.items():
                U[i*m+j][col] += coeff
            # Fill V
            for (j, k_idx), coeff in V_lf.items():
                V[j*p+k_idx][col] += coeff
            # Fill W
            # C'[k_idx, i] = C[i, k_idx]
            # contrib[k][l][i][j] is coeff for C[i,j]
            # So W[j, i] += contrib[k][l][i][j]
            for i in range(4):
                for j in range(4):
                    coeff = contrib[k][l][i][j]
                    if coeff != 0:
                        W[j*n+i][col] += coeff
            col += 1
            
    return {"U": U, "V": V, "W": W}
