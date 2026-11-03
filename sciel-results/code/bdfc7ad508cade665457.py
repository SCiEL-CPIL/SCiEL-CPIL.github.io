"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 44: 28 from Strassen (4 blocks of 2x2x2) + 16 from schoolbook (2x1x8)
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: 2x2x8 using Strassen on 4 blocks of 2x2x2
    # A is 2x2 (cols 0,1), B is 2x8 (rows 0,1)
    # We process 4 blocks of 2 columns in B.
    
    # Strassen coefficients for A (2x2)
    # a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # M1: (a+d)(e+h)
    # M2: (c+d)e
    # M3: a(f-h)
    # M4: d(g-e)
    # M5: (a+c)h
    # M6: (b-d)(g+h)
    # M7: (a-c)(e+f)
    
    # Mapping A indices to U rows (i*m+j)
    # i=0, j=0 -> 0
    # i=0, j=1 -> 1
    # i=1, j=0 -> 3
    # i=1, j=1 -> 4
    
    # Mapping B indices to V rows (j*p+k)
    # For block t, k in {2t, 2t+1}
    # j=0 -> 0*p + 2t = 2t
    # j=0 -> 0*p + 2t+1 = 2t+1
    # j=1 -> 1*p + 2t = 8 + 2t
    # j=1 -> 1*p + 2t+1 = 8 + 2t+1
    
    # Mapping C indices to W rows (k*n+i)
    # k=2t, i=0 -> 4t
    # k=2t, i=1 -> 4t+1
    # k=2t+1, i=0 -> 4t+2
    # k=2t+1, i=1 -> 4t+3
    
    # Strassen Reconstruction:
    # C[0, 2t]   = M1 + M4 - M5 - M7
    # C[1, 2t]   = M2 + M5
    # C[0, 2t+1] = M3 + M7
    # C[1, 2t+1] = M1 + M2 - M3 - M6
    
    for t in range(4):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # M1: (A00 + A11) * (B0,k0 + B1,k1)
        U[0][col] = 1  # A00
        U[4][col] = 1  # A11
        V[k0][col] = 1  # B0,k0
        V[8 + k1][col] = 1  # B1,k1
        W[2*k0][col] = 1  # C0,k0
        W[2*k1 + 1][col] = 1  # C1,k1
        col += 1
        
        # M2: (A10 + A11) * B0,k0
        U[3][col] = 1  # A10
        U[4][col] = 1  # A11
        V[k0][col] = 1  # B0,k0
        W[2*k0 + 1][col] = 1  # C1,k0
        col += 1
        
        # M3: A00 * (B0,k1 - B1,k1)
        U[0][col] = 1  # A00
        V[k1][col] = 1  # B0,k1
        V[8 + k1][col] = -1  # B1,k1
        W[2*k1][col] = 1  # C0,k1
        col += 1
        
        # M4: A11 * (B1,k0 - B0,k0)
        U[4][col] = 1  # A11
        V[8 + k0][col] = 1  # B1,k0
        V[k0][col] = -1  # B0,k0
        W[2*k0][col] = 1  # C0,k0
        col += 1
        
        # M5: (A00 + A10) * B1,k1
        U[0][col] = 1  # A00
        U[3][col] = 1  # A10
        V[8 + k1][col] = 1  # B1,k1
        W[2*k0][col] = -1  # C0,k0
        W[2*k0 + 1][col] = 1  # C1,k0
        col += 1
        
        # M6: (A01 - A11) * (B1,k0 + B1,k1)
        U[1][col] = 1  # A01
        U[4][col] = -1  # A11
        V[8 + k0][col] = 1  # B1,k0
        V[8 + k1][col] = 1  # B1,k1
        W[2*k1 + 1][col] = -1  # C1,k1
        col += 1
        
        # M7: (A00 - A10) * (B0,k0 + B0,k1)
        U[0][col] = 1  # A00
        U[3][col] = -1  # A10
        V[k0][col] = 1  # B0,k0
        V[k1][col] = 1  # B0,k1
        W[2*k0][col] = -1  # C0,k0
        W[2*k1][col] = 1  # C0,k1
        col += 1

    # Part 2: 2x1x8 schoolbook for j=2
    # A is 2x1 (col 2), B is 1x8 (row 2)
    # C = A * B
    # C[i, k] = A[i, 2] * B[2, k]
    
    for i in range(n):
        for k in range(p):
            # U row: i*m + 2
            U[i * m + 2][col] = 1
            # V row: 2*p + k
            V[2 * p + k][col] = 1
            # W row: k*n + i
            W[k * n + i][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
