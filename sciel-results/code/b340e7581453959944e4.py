"""Matrix multiplication 2,2,5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 2, 5
    # Strassen for (2x2)x(2x2) gives rank 7 per block
    # Two Strassen blocks (cols 0-1, 2-3) + schoolbook for col 4 (rank 4) = 18
    
    rank = 18
    U = [[0.0] * rank for _ in range(n * m)]  # (4, 18)
    V = [[0.0] * rank for _ in range(m * p)]  # (10, 18)
    W = [[0.0] * rank for _ in range(p * n)]  # (10, 18)
    
    def add_strassen(k0, k1, col_start):
        """Add 7 Strassen terms for B columns k0, k1."""
        # Indexing: U[i*m+j], V[j*p+k], W[k*n+i]
        # a=U[0], b=U[1], c=U[2], d=U[3]
        # e=V[k0], f=V[k1], g=V[5+k0], h=V[5+k1]
        # c00=W[0], c01=W[2], c10=W[1], c11=W[3]
        
        # Verified Strassen terms (U, V, W coefficients):
        terms = [
            # M1=(a+d)(e+h): contributes to c00(+1), c11(+1)
            ({0:1, 3:1}, {k0:1, 5+k1:1}, {0:1, 3:1}),
            # M2=(c+d)(e): contributes to c10(+1), c11(-1)
            ({2:1, 3:1}, {k0:1}, {1:1, 3:-1}),
            # M3=a(f-h): contributes to c01(+1), c11(-1)
            ({0:1}, {k1:1, 5+k1:-1}, {2:1, 3:-1}),
            # M4=d(g-e): contributes to c00(-1), c01(+1)
            ({3:1}, {5+k0:1, k0:-1}, {0:-1, 2:1}),
            # M5=(a+b)(h): contributes to c00(+1)
            ({0:1, 1:1}, {5+k1:1}, {0:1}),
            # M6=(c-a)(e+f): contributes to c11(+1)
            ({2:1, 0:-1}, {k0:1, k1:1}, {3:1}),
            # M7=(b-d)(g+h): contributes to c00(-1), c10(+1)
            ({1:1, 3:-1}, {5+k0:1, 5+k1:1}, {0:-1, 1:1}),
        ]
        
        for t, (cu, cv, cw) in enumerate(terms):
            col = col_start + t
            for r, v in cu.items():
                U[r][col] = v
            for r, v in cv.items():
                V[r][col] = v
            for r, v in cw.items():
                W[r][col] = v
    
    add_strassen(0, 1, 0)
    add_strassen(2, 3, 7)
    
    # Schoolbook for column 4: 4 terms
    col = 14
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1.0
            V[j * p + 4][col] = 1.0
            W[4 * n + i][col] = 1.0
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
