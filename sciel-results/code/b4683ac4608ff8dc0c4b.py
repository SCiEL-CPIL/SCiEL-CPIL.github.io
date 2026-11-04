import numpy as np

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # We aim for rank 56
    # If it fails, return 60
    # For now, return 60 to be safe, as implementing 56 correctly is complex
    # But the directive asks for a stronger algorithm.
    # I will implement the 56-rank decomposition.
    
    U = []
    V = []
    W = []
    
    def add_term(u, v, w):
        U.append(u)
        V.append(v)
        W.append(w)
        
    def init_vec(dim):
        return [0] * dim
        
    # Strassen for 2x2x2
    # A is 2x2, B is 2x2, C is 2x2
    # Indices for A: (i, j) -> i in {0,1}, j in {0,1}
    # Indices for B: (j, k) -> j in {0,1}, k in {0,1}
    # Indices for C: (k, i) -> k in {0,1}, i in {0,1}
    # Offsets: i0, j0 for A; j0, k0 for B; k0, i0 for C
    
    def add_strassen(i0, j0, k0):
        # M1 = (a+d)(e+i)
        # a=A[i0,j0], d=A[i0+1,j0+1]
        # e=B[j0,k0], i=B[j0+1,k0+1]
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Term 1: (a+d)(e+i)
        u1 = init_vec(n*m)
        v1 = init_vec(m*p)
        w1 = init_vec(p*n)
        
        # M1
        u1[i0*m+j0] = 1
        u1[(i0+1)*m+(j0+1)] = 1
        v1[j0*p+k0] = 1
        v1[(j0+1)*p+(k0+1)] = 1
        w1[k0*n+i0] = 1 # C11
        w1[(k0+1)*n+(i0+1)] = 1 # C22
        # Wait, C22 gets M1.
        # C11 gets M1.
        # So w1 has 1 at C11 and C22?
        # No, w1 is a vector. It contributes to multiple C entries.
        # The term is rank 1.
        # U is vector, V is vector, W is vector.
        # The product is U_k V_l W_m.
        # So W can have multiple non-zero entries.
        
        # Let's re-verify the structure.
        # T = sum U_r V_r W_r
        # U_r is in R^{nm}, V_r in R^{mp}, W_r in R^{pn}
        # So for each r, we define the vectors.
        
        # M1 = (A[i0,j0] + A[i0+1,j0+1]) * (B[j0,k0] + B[j0+1,k0+1])
        # Contributes to C[i0,k0] and C[i0+1,k0+1]
        
        u1[i0*m+j0] = 1
        u1[(i0+1)*m+(j0+1)] = 1
        v1[j0*p+k0] = 1
        v1[(j0+1)*p+(k0+1)] = 1
        w1[k0*n+i0] = 1
        w1[(k0+1)*n+(i0+1)] = 1
        add_term(u1, v1, w1)
        
        # M2 = (c+d)e = (A[i0+1,j0] + A[i0+1,j0+1]) * B[j0,k0]
        # Contributes to C[i0+1,k0] (C21) and C[i0+1,k0+1] (C22)
        u2 = init_vec(n*m)
        v2 = init_vec(m*p)
        w2 = init_vec(p*n)
        u2[(i0+1)*m+j0] = 1
        u2[(i0+1)*m+(j0+1)] = 1
        v2[j0*p+k0] = 1
        w2[k0*n+(i0+1)] = 1 # C21
        w2[(k0+1)*n+(i0+1)] = -1 # C22 (M1 - M2)
        add_term(u2, v2, w2)
        
        # M3 = a(f-i) = A[i0,j0] * (B[j0+1,k0] - B[j0+1,k0+1])
        # Contributes to C[i0,k0+1] (C12) and C[i0+1,k0+1] (C22)
        u3 = init_vec(n*m)
        v3 = init_vec(m*p)
        w3 = init_vec(p*n)
        u3[i0*m+j0] = 1
        v3[(j0+1)*p+k0] = 1
        v3[(j0+1)*p+(k0+1)] = -1
        w3[(k0+1)*n+i0] = 1 # C12
        w3[(k0+1)*n+(i0+1)] = 1 # C22
        add_term(u3, v3, w3)
        
        # M4 = d(g-h) = A[i0+1,j0+1] * (B[j0,k0+1] - B[j0+1,k0+1])
        # Contributes to C[i0,k0] (C11) and C[i0+1,k0] (C21)
        u4 = init_vec(n*m)
        v4 = init_vec(m*p)
        w4 = init_vec(p*n)
        u4[(i0+1)*m+(j0+1)] = 1
        v4[j0*p+(k0+1)] = 1
        v4[(j0+1)*p+(k0+1)] = -1
        w4[k0*n+i0] = 1 # C11
        w4[k0*n+(i0+1)] = 1 # C21
        add_term(u4, v4, w4)
        
        # M5 = (a+b)h = (A[i0,j0] + A[i0,j0+1]) * B[j0+1,k0]
        # Contributes to C[i0,k0] (C11) and C[i0,k0+1] (C12)
        u5 = init_vec(n*m)
        v5 = init_vec(m*p)
        w5 = init_vec(p*n)
        u5[i0*m+j0] = 1
        u5[i0*m+(j0+1)] = 1
        v5[(j0+1)*p+k0] = 1
        w5[k0*n+i0] = -1 # C11
        w5[(k0+1)*n+i0] = 1 # C12
        add_term(u5, v5, w5)
        
        # M6 = (c-b)(e+f) = (A[i0+1,j0] - A[i0,j0+1]) * (B[j0,k0] + B[j0,k0+1])
        # Contributes to C[i0+1,k0+1] (C22)
        u6 = init_vec(n*m)
        v6 = init_vec(m*p)
        w6 = init_vec(p*n)
        u6[(i0+1)*m+j0] = 1
        u6[i0*m+(j0+1)] = -1
        v6[j0*p+k0] = 1
        v6[j0*p+(k0+1)] = 1
        w6[(k0+1)*n+(i0+1)] = 1 # C22
        add_term(u6, v6, w6)
        
        # M7 = (a-d)(i+j) = (A[i0,j0] - A[i0+1,j0+1]) * (B[j0+1,k0] + B[j0+1,k0+1])
        # Contributes to C[i0,k0] (C11)
        u7 = init_vec(n*m)
        v7 = init_vec(m*p)
        w7 = init_vec(p*n)
        u7[i0*m+j0] = 1
        u7[(i0+1)*m+(j0+1)] = -1
        v7[(j0+1)*p+k0] = 1
        v7[(j0+1)*p+(k0+1)] = 1
        w7[k0*n+i0] = 1 # C11
        add_term(u7, v7, w7)
        
    # Schoolbook for 2x2x1
    # A is 2x2, B is 2x1, C is 2x1
    def add_221(i0, j0, k0):
        # C1 = a e + b h
        # C2 = c e + d h
        # 4 terms
        for idx in range(4):
            u = init_vec(n*m)
            v = init_vec(m*p)
            w = init_vec(p*n)
            if idx == 0:
                u[i0*m+j0] = 1
                v[j0*p+k0] = 1
                w[k0*n+i0] = 1
            elif idx == 1:
                u[i0*m+(j0+1)] = 1
                v[(j0+1)*p+k0] = 1
                w[k0*n+i0] = 1
            elif idx == 2:
                u[(i0+1)*m+j0] = 1
                v[j0*p+k0] = 1
                w[k0*n+(i0+1)] = 1
            elif idx == 3:
                u[(i0+1)*m+(j0+1)] = 1
                v[(j0+1)*p+k0] = 1
                w[k0*n+(i0+1)] = 1
            add_term(u, v, w)

    # Schoolbook for 1x2x2
    # A is 1x2, B is 2x2, C is 1x2
    def add_122(i0, j0, k0):
        # C1 = a e + b h
        # C2 = a f + b i
        for idx in range(4):
            u = init_vec(n*m)
            v = init_vec(m*p)
            w = init_vec(p*n)
            if idx == 0:
                u[i0*m+j0] = 1
                v[j0*p+k0] = 1
                w[k0*n+i0] = 1
            elif idx == 1:
                u[i0*m+(j0+1)] = 1
                v[(j0+1)*p+k0] = 1
                w[k0*n+i0] = 1
            elif idx == 2:
                u[i0*m+j0] = 1
                v[j0*p+(k0+1)] = 1
                w[(k0+1)*n+i0] = 1
            elif idx == 3:
                u[i0*m+(j0+1)] = 1
                v[(j0+1)*p+(k0+1)] = 1
                w[(k0+1)*n+i0] = 1
            add_term(u, v, w)

    # Apply decomposition
    # A_top (2x4), B_left (4x2) -> C_top_left (2x2)
    # Split A_top into A1 (2x2), A2 (2x2)
    # Split B_left into B1 (2x2), B2 (2x2)
    add_strassen(0, 0, 0) # A1 B1
    add_strassen(0, 2, 0) # A2 B2 (j0=2)
    
    # A_top (2x4), B_right (4x3) -> C_top_right (2x3)
    # Split B_right into B3 (4x2), B4 (4x1)
    # B3 is k in {2,3}, B4 is k=4
    # A_top B3 (2x4x2)
    add_strassen(0, 0, 2) # A1 B3_part1 (j0=0, k0=2)
    add_strassen(0, 2, 2) # A2 B3_part2 (j0=2, k0=2)
    # A_top B4 (2x4x1)
    add_221(0, 0, 4) # A1 B4_part1
    add_221(0, 2, 4) # A2 B4_part2
    
    # A_bot (1x4), B_left (4x2) -> C_bot_left (1x2)
    add_122(2, 0, 0) # A_bot B1
    add_122(2, 2, 0) # A_bot B2
    
    # A_bot (1x4), B_right (4x3) -> C_bot_right (1x3)
    # Split B_right into B3 (4x2), B4 (4x1)
    add_122(2, 0, 2) # A_bot B3_part1
    add_122(2, 2, 2) # A_bot B3_part2
    # A_bot B4 (1x4x1)
    # 1x2x1 twice
    for j in [0, 2]:
        u = init_vec(n*m)
        v = init_vec(m*p)
        w = init_vec(p*n)
        u[2*m+j] = 1
        v[j*p+4] = 1
        w[4*n+2] = 1
        add_term(u, v, w)
        u = init_vec(n*m)
        v = init_vec(m*p)
        w = init_vec(p*n)
        u[2*m+(j+1)] = 1
        v[(j+1)*p+4] = 1
        w[4*n+2] = 1
        add_term(u, v, w)

    # Convert to matrices
    rank = len(U)
    U_mat = np.zeros((n*m, rank))
    V_mat = np.zeros((m*p, rank))
    W_mat = np.zeros((p*n, rank))
    
    for r in range(rank):
        for i in range(n*m):
            U_mat[i, r] = U[r][i]
        for i in range(m*p):
            V_mat[i, r] = V[r][i]
        for i in range(p*n):
            W_mat[i, r] = W[r][i]
            
    return {"U": U_mat, "V": V_mat, "W": W_mat}

