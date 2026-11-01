"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    # Strassen's algorithm for 2x2 matrix multiplication (rank 7)
    n, m, p = DIMENSIONS
    
    # Define the 7 Strassen products
    # Let A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # Output C = A*B = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    
    # Strassen's 7 products:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-d)(f+g)
    # M7 = (b-c)(e+h)
    
    # Then:
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    rank = 7
    
    # U: shape (n*m, rank) = (4, 7)
    # U[i*m+j, k] corresponds to coefficient in product k for element (i,j) of A
    # A[0,0]=a, A[0,1]=b, A[1,0]=c, A[1,1]=d
    # Index: 0=a, 1=b, 2=c, 3=d
    
    # V: shape (m*p, rank) = (4, 7)
    # V[j*p+k, l] corresponds to coefficient in product l for element (j,k) of B
    # B[0,0]=e, B[0,1]=f, B[1,0]=g, B[1,1]=h
    # Index: 0=e, 1=f, 2=g, 3=h
    
    # W: shape (p*n, rank) = (4, 7)
    # W[k*n+i, l] corresponds to coefficient in product l for element (k,i) of C
    # C[0,0], C[0,1], C[1,0], C[1,1]
    # Index: 0=C[0,0], 1=C[0,1], 2=C[1,0], 3=C[1,1]
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Product 0: M1 = (a+d)(e+h)
    # a=idx0, d=idx3 in U; e=idx0, h=idx3 in V
    U[0][0] = 1  # a
    U[3][0] = 1  # d
    V[0][0] = 1  # e
    V[3][0] = 1  # h
    W[0][0] = 1  # C[0,0]
    W[3][0] = 1  # C[1,1]
    
    # Product 1: M2 = (c+d)e
    U[2][1] = 1  # c
    U[3][1] = 1  # d
    V[0][1] = 1  # e
    W[2][1] = 1  # C[1,0]
    W[3][1] = -1 # C[1,1]
    
    # Product 2: M3 = a(f-h)
    U[0][2] = 1  # a
    V[1][2] = 1  # f
    V[3][2] = -1 # h
    W[1][2] = 1  # C[0,1]
    W[3][2] = 1  # C[1,1]
    
    # Product 3: M4 = d(g-e)
    U[3][3] = 1  # d
    V[2][3] = 1  # g
    V[0][3] = -1 # e
    W[0][3] = 1  # C[0,0]
    W[2][3] = 1  # C[1,0]
    
    # Product 4: M5 = (a+b)h
    U[0][4] = 1  # a
    U[1][4] = 1  # b
    V[3][4] = 1  # h
    W[0][4] = -1 # C[0,0]
    W[1][4] = 1  # C[0,1]
    
    # Product 5: M6 = (c-d)(f+g)
    U[2][5] = 1  # c
    U[3][5] = -1 # d
    V[1][5] = 1  # f
    V[2][5] = 1  # g
    W[3][5] = 1  # C[1,1]
    
    # Product 6: M7 = (b-c)(e+h)
    U[1][6] = 1  # b
    U[2][6] = -1 # c
    V[0][6] = 1  # e
    V[3][6] = 1  # h
    W[0][6] = 1  # C[0,0]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
