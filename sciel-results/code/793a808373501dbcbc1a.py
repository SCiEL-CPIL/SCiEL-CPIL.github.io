def add_strassen_222(U, V, W, term_idx, i1, i2, j1, j2, k1, k2):
    # A entries: A[i1,j1]=a, A[i1,j2]=b, A[i2,j1]=c, A[i2,j2]=d
    # B entries: B[j1,k1]=e, B[j1,k2]=g, B[j2,k1]=f, B[j2,k2]=h
    # C entries: C[i1,k1], C[i1,k2], C[i2,k1], C[i2,k2]
    
    n, m, p = 2, 4, 7
    
    # m1 = (a+d)(e+h)
    U[i1*m+j1][term_idx] += 1
    U[i2*m+j2][term_idx] += 1
    V[j1*p+k1][term_idx] += 1
    V[j2*p+k2][term_idx] += 1
    W[k1*n+i1][term_idx] += 1
    W[k2*n+i2][term_idx] += 1
    
    # m2 = (c+d)(e)
    U[i2*m+j1][term_idx] += 1
    U[i2*m+j2][term_idx] += 1
    V[j1*p+k1][term_idx] += 1
    W[k1*n+i2][term_idx] += 1
    W[k2*n+i2][term_idx] += -1
    
    # m3 = a(g-h)
    U[i1*m+j1][term_idx] += 1
    V[j1*p+k2][term_idx] += 1
    V[j2*p+k2][term_idx] += -1
    W[k2*n+i1][term_idx] += 1
    W[k2*n+i2][term_idx] += 1
    
    # m4 = d(f-e)
    U[i2*m+j2][term_idx] += 1
    V[j2*p+k1][term_idx] += 1
    V[j1*p+k1][term_idx] += -1
    W[k1*n+i1][term_idx] += 1
    W[k1*n+i2][term_idx] += 1
    
    # m5 = (a+b)(h)
    U[i1*m+j1][term_idx] += 1
    U[i1*m+j2][term_idx] += 1
    V[j2*p+k2][term_idx] += 1
    W[k1*n+i1][term_idx] += -1
    W[k2*n+i1][term_idx] += 1
    
    # m6 = (c-a)(e+g)
    U[i2*m+j1][term_idx] += 1
    U[i1*m+j1][term_idx] += -1
    V[j1*p+k1][term_idx] += 1
    V[j1*p+k2][term_idx] += 1
    W[k2*n+i2][term_idx] += 1
    
    # m7 = (b-d)(f+h)
    U[i1*m+j2][term_idx] += 1
    U[i2*m+j2][term_idx] += -1
    V[j2*p+k1][term_idx] += 1
    V[j2*p+k2][term_idx] += 1
    W[k1*n+i1][term_idx] += 1
    
    return term_idx + 7
