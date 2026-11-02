import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # Generate rank-7 factors for 2x2
    U2, V2, W2 = generate_strassen_2x2()
    
    # Generate rank-49 factors for 4x4
    # s = 7 * s1 + s2
    rank49 = 49
    U49 = [[0]*rank49 for _ in range(16)]
    V49 = [[0]*rank49 for _ in range(16)]
    W49 = [[0]*rank49 for _ in range(16)]
    
    for i1 in range(2):
        for j1 in range(2):
            for i2 in range(2):
                for j2 in range(2):
                    i = 2*i1 + i2
                    j = 2*j1 + j2
                    for s1 in range(7):
                        u1 = U2[i1*2 + j1][s1] # Wait, U2 index is i*2+j? No, U2 is 4x7.
                        # U2[i, s] where i is 0..3.
                        # i = 2*i1 + i2.
                        # So U2[i][s1] is the coefficient for A[i1, j1][i2, j2] in M_s1.
                        # But U2 was generated for 2x2.
                        # U2[i, s] where i is the row index of A.
                        # In the 2x2 case, A is 2x2.
                        # So U2[i1*2 + i2, s1] is not correct.
                        # U2 is for 2x2 matrices. The indices are 0,1,2,3.
                        # i = 2*i1 + i2.
                        # So U2[i, s1] is the coefficient for A[i, j] in M_s1.
                        # But j is also 2*j1 + j2.
                        # So we need U2[i][s1] * U2[j][s1]? No.
                        # U2[i, s] is the coefficient of A[i, j] in M_s.
                        # But j is fixed for a given U entry? No.
                        # U is a matrix of shape (n*m, rank).
                        # U[i*m + j, s].
                        # So U2[i*2 + j, s] is the coefficient of A[i, j] in M_s.
                        
                        # Let's re-read generate_strassen_2x2.
                        # U[0][0] += 1; U[3][0] += 1 # A11, A22
                        # A11 is i=0, j=0 -> idx 0.
                        # A22 is i=1, j=1 -> idx 3.
                        # So U2[idx, s] is correct.
                        
                        u_coeff = U2[i*2 + j][s1]
                        if u_coeff != 0:
                            for s2 in range(7):
                                s = 7*s1 + s2
                                u2_coeff = U2[i2*2 + j2][s2]
                                if u2_coeff != 0:
                                    U49[i*4 + j][s] += u_coeff * u2_coeff
    # This is getting complicated. Let's just use the tensor product property.
    # The rank-49 factors are the tensor product of the rank-7 factors.
    # U49[i, j, s] = U2[i1, j1, s1] * U2[i2, j2, s2]
    # where i = 2*i1 + i2, j = 2*j1 + j2, s = 7*s1 + s2.
    
    # Let's implement it cleanly.
    U49 = [[0]*49 for _ in range(16)]
    V49 = [[0]*49 for _ in range(16)]
    W49 = [[0]*49 for _ in range(16)]
    
    for i1 in range(2):
        for j1 in range(2):
            for i2 in range(2):
                for j2 in range(2):
                    i = 2*i1 + i2
                    j = 2*j1 + j2
                    idx_u = i*4 + j
                    for s1 in range(7):
                        u1 = U2[i1*2 + j1][s1]
                        if u1 == 0: continue
                        for s2 in range(7):
                            u2 = U2[i2*2 + j2][s2]
                            if u2 == 0: continue
                            s = 7*s1 + s2
                            U49[idx_u][s] += u1 * u2
                            
    for j1 in range(2):
        for k1 in range(2):
            for j2 in range(2):
                for k2 in range(2):
                    j = 2*j1 + j2
                    k = 2*k1 + k2
                    idx_v = j*4 + k
                    for s1 in range(7):
                        v1 = V2[j1*2 + k1][s1]
                        if v1 == 0: continue
                        for s2 in range(7):
                            v2 = V2[j2*2 + k2][s2]
                            if v2 == 0: continue
                            s = 7*s1 + s2
                            V49[idx_v][s] += v1 * v2
                            
    for k1 in range(2):
        for i1 in range(2):
            for k2 in range(2):
                for i2 in range(2):
                    k = 2*k1 + k2
                    i = 2*i1 + i2
                    idx_w = k*4 + i
                    for s1 in range(7):
                        w1 = W2[k1*2 + i1][s1]
                        if w1 == 0: continue
                        for s2 in range(7):
                            w2 = W2[k2*2 + i2][s2]
                            if w2 == 0: continue
                            s = 7*s1 + s2
                            W49[idx_w][s] += w1 * w2
                            
    # Now construct rank-98 factors for 4x4x8
    rank98 = 98
    U = [[0]*rank98 for _ in range(16)]
    V = [[0]*rank98 for _ in range(32)]
    W = [[0]*rank98 for _ in range(32)]
    
    # First 49 columns
    for i in range(16):
        for s in range(49):
            U[i][s] = U49[i][s]
            
    for j in range(16):
        for k in range(4):
            for s in range(49):
                V[j*8 + k][s] = V49[j*4 + k][s]
                
    for k in range(4):
        for i in range(4):
            for s in range(49):
                W[k*4 + i][s] = W49[k*4 + i][s]
                
    # Next 49 columns
    for i in range(16):
        for s in range(49):
            U[i][s + 49] = U49[i][s]
            
    for j in range(16):
        for k in range(4, 8):
            for s in range(49):
                V[j*8 + k][s + 49] = V49[j*4 + (k-4)][s]
                
    for k in range(4, 8):
        for i in range(4):
            for s in range(49):
                W[k*4 + i][s + 49] = W49[(k-4)*4 + i][s]
                
    return {"U": U, "V": V, "W": W}

def generate_strassen_2x2():
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    W[0][0] += 1; W[3][0] += 1
    
    U[1][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[1][1] += 1; W[3][1] -= 1
    
    U[0][2] += 1
    V[3][2] += 1; V[1][2] -= 1
    W[2][2] += 1; W[3][2] += 1
    
    U[3][3] += 1
    V[2][3] += 1; V[0][3] -= 1
    W[0][3] += 1; W[1][3] += 1
    
    U[0][4] += 1; U[2][4] += 1
    V[3][4] += 1
    W[0][4] -= 1; W[2][4] += 1
    
    U[1][5] += 1; U[0][5] -= 1
    V[0][5] += 1; V[2][5] += 1
    W[3][5] += 1
    
    U[2][6] += 1; U[3][6] -= 1
    V[1][6] += 1; V[3][6] += 1
    W[0][6] += 1; W[2][6] -= 1
    
    return U, V, W
