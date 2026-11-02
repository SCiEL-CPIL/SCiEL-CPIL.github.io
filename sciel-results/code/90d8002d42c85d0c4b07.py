"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    # We use a block decomposition approach.
    # For i in {0, 1}, we split j and k into two groups of 2.
    # This gives 4 blocks of size 2x2x2 for each i.
    # Each 2x2x2 block uses Strassen's algorithm (rank 7).
    # For i = 2, we use the trivial decomposition (rank 16).
    # Total rank = 2 * 4 * 7 + 16 = 56 + 16 = 72.
    
    # Wait, let me re-check the Strassen decomposition for 2x2x2.
    # The 7 rank-1 tensors for 2x2x2 matrix multiplication are:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    #
    # The output is:
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    #
    # Let's map this to our tensor decomposition.
    # For a block with i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}:
    # U indices: i0*m+j0, i0*m+j0+1, (i0+1)*m+j0, (i0+1)*m+j0+1
    # V indices: j0*p+k0, j0*p+k0+1, (j0+1)*p+k0, (j0+1)*p+k0+1
    # W indices: k0*n+i0, k0*n+(i0+1), (k0+1)*n+i0, (k0+1)*n+(i0+1)
    
    U = []
    V = []
    W = []
    
    def add_term(u, v, w):
        U.append(u)
        V.append(v)
        W.append(w)
    
    # Strassen terms for a 2x2x2 block
    # a = A[i, j], b = B[j, k], c = C[i, k]
    # For block (i0, j0, k0):
    # a11 = A[i0, j0], a12 = A[i0, j0+1], a21 = A[i0+1, j0], a22 = A[i0+1, j0+1]
    # b11 = B[j0, k0], b12 = B[j0, k0+1], b21 = B[j0+1, k0], b22 = B[j0+1, k0+1]
    # c11 = C[i0, k0], c12 = C[i0, k0+1], c21 = C[i0+1, k0], c22 = C[i0+1, k0+1]
    
    def add_strassen_block(i0, j0, k0):
        # M1 = (a11+a22)(b11+b22)
        u1 = [0]*(n*m); u1[i0*m+j0] = 1; u1[(i0+1)*m+j0+1] = 1
        v1 = [0]*(m*p); v1[j0*p+k0] = 1; v1[(j0+1)*p+k0+1] = 1
        w1 = [0]*(p*n); w1[k0*n+i0] = 1; w1[k0*n+i0+1] = -1; w1[(k0+1)*n+i0] = 1; w1[(k0+1)*n+i0+1] = -1
        add_term(u1, v1, w1)
        
        # M2 = (a21+a22)b11
        u2 = [0]*(n*m); u2[(i0+1)*m+j0] = 1; u2[(i0+1)*m+j0+1] = 1
        v2 = [0]*(m*p); v2[j0*p+k0] = 1
        w2 = [0]*(p*n); w2[k0*n+i0] = -1; w2[(k0+1)*n+i0+1] = 1
        add_term(u2, v2, w2)
        
        # M3 = a11(b12-b22)
        u3 = [0]*(n*m); u3[i0*m+j0] = 1
        v3 = [0]*(m*p); v3[j0*p+k0+1] = 1; v3[(j0+1)*p+k0+1] = -1
        w3 = [0]*(p*n); w3[k0*n+i0] = 1; w3[(k0+1)*n+i0+1] = 1
        add_term(u3, v3, w3)
        
        # M4 = a22(b21-b11)
        u4 = [0]*(n*m); u4[(i0+1)*m+j0+1] = 1
        v4 = [0]*(m*p); v4[(j0+1)*p+k0] = 1; v4[j0*p+k0] = -1
        w4 = [0]*(p*n); w4[k0*n+i0] = 1; w4[(k0+1)*n+i0+1] = 1
        add_term(u4, v4, w4)
        
        # M5 = (a11+a12)b22
        u5 = [0]*(n*m); u5[i0*m+j0] = 1; u5[i0*m+j0+1] = 1
        v5 = [0]*(m*p); v5[(j0+1)*p+k0+1] = 1
        w5 = [0]*(p*n); w5[k0*n+i0] = -1; w5[(k0+1)*n+i0+1] = -1
        add_term(u5, v5, w5)
        
        # M6 = (a21-a11)(b11+b12)
        u6 = [0]*(n*m); u6[(i0+1)*m+j0] = 1; u6[i0*m+j0] = -1
        v6 = [0]*(m*p); v6[j0*p+k0] = 1; v6[j0*p+k0+1] = 1
        w6 = [0]*(p*n); w6[(k0+1)*n+i0+1] = 1
        add_term(u6, v6, w6)
        
        # M7 = (a12-a22)(b21+b22)
        u7 = [0]*(n*m); u7[i0*m+j0+1] = 1; u7[(i0+1)*m+j0+1] = -1
        v7 = [0]*(m*p); v7[(j0+1)*p+k0] = 1; v7[(j0+1)*p+k0+1] = 1
        w7 = [0]*(p*n); w7[k0*n+i0] = 1
        add_term(u7, v7, w7)
    
    # Add Strassen blocks for i in {0, 1}
    for i0 in [0, 1]:
        for j0 in [0, 2]:
            for k0 in [0, 2]:
                add_strassen_block(i0, j0, k0)
    
    # Add trivial decomposition for i = 2
    for j in range(m):
        for k in range(p):
            u = [0]*(n*m); u[2*m+j] = 1
            v = [0]*(m*p); v[j*p+k] = 1
            w = [0]*(p*n); w[k*n+2] = 1
            add_term(u, v, w)
    
    rank = len(U)
    
    # Convert to list of lists format
    U_mat = [[u[i] for u in U] for i in range(n*m)]
    V_mat = [[v[i] for v in V] for i in range(m*p)]
    W_mat = [[w[i] for w in W] for i in range(p*n)]
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}
# EVOLVE-BLOCK-END
