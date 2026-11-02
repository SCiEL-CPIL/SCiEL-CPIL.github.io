def construct():
    n, m, p = 3, 4, 4
    # We'll build U, V, W as lists
    rank = 44
    U = [[0]*rank for _ in range(n*m)]  # 12 x 44
    V = [[0]*rank for _ in range(m*p)]  # 16 x 44
    W = [[0]*rank for _ in range(p*n)]  # 12 x 44
    
    col = 0
    for b_r in range(2):
        for b_c in range(2):
            # Strassen part: 7 terms, rows i=0,1
            # m1 = (a+d)(e+h)
            # a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
            # e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
            # j_local 0,1 ; k_local 0,1
            # j_global = b_r*2 + j_local
            # k_global = b_c*2 + k_local
            
            # m1: A-part: A[0,0]=1, A[1,1]=1; B-part: B[0,0]=1, B[1,1]=1
            U[0*m + (b_r*2+0)][col] = 1
            U[1*m + (b_r*2+1)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+0)][col] = 1
            V[(b_r*2+1)*p + (b_c*2+1)][col] = 1
            W[(b_c*2+0)*n + 0][col] = 1   # C[0,0] += m1
            W[(b_c*2+1)*n + 1][col] = 1   # C[1,1] += m1
            col += 1
            
            # m2 = (c+d)e: A-part: A[1,0]=1, A[1,1]=1; B-part: B[0,0]=1
            U[0*m + (b_r*2+0)][col] = 0
            U[1*m + (b_r*2+0)][col] = 1
            U[1*m + (b_r*2+1)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+0)][col] = 1
            W[(b_c*2+0)*n + 1][col] = 1   # C[1,0] += m2
            W[(b_c*2+1)*n + 1][col] = -1  # C[1,1] -= m2
            col += 1
            
            # m3 = a(f-h): A-part: A[0,0]=1; B-part: B[0,1]=1, B[1,1]=-1
            U[0*m + (b_r*2+0)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+1)][col] = 1
            V[(b_r*2+1)*p + (b_c*2+1)][col] = -1
            W[(b_c*2+1)*n + 0][col] = 1   # C[0,1] += m3
            W[(b_c*2+1)*n + 1][col] = 1   # C[1,1] += m3
            col += 1
            
            # m4 = d(g-e): A-part: A[1,1]=1; B-part: B[1,0]=1, B[0,0]=-1
            U[1*m + (b_r*2+1)][col] = 1
            V[(b_r*2+1)*p + (b_c*2+0)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+0)][col] = -1
            W[(b_c*2+0)*n + 0][col] = 1   # C[0,0] += m4
            W[(b_c*2+0)*n + 1][col] = 1   # C[1,0] += m4
            col += 1
            
            # m5 = (a+b)h: A-part: A[0,0]=1, A[0,1]=1; B-part: B[1,1]=1
            U[0*m + (b_r*2+0)][col] = 1
            U[0*m + (b_r*2+1)][col] = 1
            V[(b_r*2+1)*p + (b_c*2+1)][col] = 1
            W[(b_c*2+0)*n + 0][col] = -1  # C[0,0] -= m5
            W[(b_c*2+1)*n + 0][col] = 1   # C[0,1] += m5
            col += 1
            
            # m6 = (c-a)(e+f): A-part: A[1,0]=1, A[0,0]=-1; B-part: B[0,0]=1, B[0,1]=1
            U[0*m + (b_r*2+0)][col] = -1
            U[1*m + (b_r*2+0)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+0)][col] = 1
            V[(b_r*2+0)*p + (b_c*2+1)][col] = 1
            W[(b_c*2+1)*n + 1][col] = 1   # C[1,1] += m6
            col += 1
            
            # m7 = (b-d)(g+h): A-part: A[0,1]=1, A[1,1]=-1; B-part: B[1,0]=1, B[1,1]=1
            U[0*m + (b_r*2+1)][col] = 1
            U[1*m + (b_r*2+1)][col] = -1
            V[(b_r*2+1)*p + (b_c*2+0)][col] = 1
            V[(b_r*2+1)*p + (b_c*2+1)][col] = 1
            W[(b_c*2+0)*n + 0][col] = 1   # C[0,0] += m7
            col += 1
            
            # Trivial part for row i=2: 4 terms
            for j_local in range(2):
                for k_local in range(2):
                    U[2*m + (b_r*2+j_local)][col] = 1
                    V[(b_r*2+j_local)*p + (b_c*2+k_local)][col] = 1
                    W[(b_c*2+k_local)*n + 2][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
