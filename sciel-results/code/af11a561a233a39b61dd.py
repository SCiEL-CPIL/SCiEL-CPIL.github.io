"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    # Strassen's rank-7 decomposition for 2x2 matrix multiplication.
    # U is (n*m, rank) = (4, 7), V is (m*p, rank) = (4, 7), W is (p*n, rank) = (4, 7).
    # Indexing: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r].
    #
    # Strassen's 7 products:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B11)
    # M4 = A11*(B01-B10)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    #
    # Result:
    # C00 = M1 - M4 + M5 - M7
    # C10 = M2 + M4
    # C01 = M3 + M5
    # C11 = M1 - M2 + M3 + M6
    #
    # U rows: (i,j) for A[i,j]. i*m+j where m=2.
    #   row 0: (0,0) -> A00
    #   row 1: (0,1) -> A01
    #   row 2: (1,0) -> A10
    #   row 3: (1,1) -> A11
    #
    # V rows: (j,k) for B[j,k]. j*p+k where p=2.
    #   row 0: (0,0) -> B00
    #   row 1: (0,1) -> B01
    #   row 2: (1,0) -> B10
    #   row 3: (1,1) -> B11
    #
    # W rows: (k,i) for C[i,k]. k*n+i where n=2.
    #   row 0: (0,0) -> C00
    #   row 1: (0,1) -> C01
    #   row 2: (1,0) -> C10
    #   row 3: (1,1) -> C11
    #
    # U (4x7): each column r corresponds to factor of product M_r on A side.
    # M1: A00=1, A01=0, A10=0, A11=1
    # M2: A00=0, A01=0, A10=1, A11=1
    # M3: A00=1, A01=0, A10=0, A11=0
    # M4: A00=0, A01=0, A10=0, A11=1
    # M5: A00=1, A01=1, A10=0, A11=0
    # M6: A00=-1, A01=0, A10=1, A11=0
    # M7: A00=0, A01=1, A10=0, A11=-1
    U = [
        [1, 0, 1, 0, 1, -1, 0],   # A00
        [0, 0, 0, 0, 1,  0, 1],   # A01
        [0, 1, 0, 0, 0,  1, 0],   # A10
        [1, 1, 0, 1, 0,  0,-1],   # A11
    ]
    #
    # V (4x7): each column r corresponds to factor of product M_r on B side.
    # M1: B00=1, B01=0, B10=0, B11=1
    # M2: B00=1, B01=0, B10=0, B11=0
    # M3: B00=0, B01=0, B10=1, B11=-1
    # M4: B00=0, B01=1, B10=-1, B11=0
    # M5: B00=0, B01=0, B10=0, B11=1
    # M6: B00=1, B01=0, B10=1, B11=0
    # M7: B00=0, B01=1, B10=0, B11=1
    V = [
        [1, 1, 0, 0, 0, 1, 0],   # B00
        [0, 0, 0, 1, 0, 0, 1],   # B01
        [0, 0, 1,-1, 0, 1, 0],   # B10
        [1, 0,-1, 0, 1, 0, 1],   # B11
    ]
    #
    # W (4x7): each column r is the coefficient of M_r in the output C.
    # C00 = M1 - M4 + M5 - M7  -> W[0] = [1, 0, 0, -1, 1, 0, -1]
    # C01 = M3 + M5            -> W[1] = [0, 0, 1, 0, 1, 0, 0]
    # C10 = M2 + M4            -> W[2] = [0, 1, 0, 1, 0, 0, 0]
    # C11 = M1 - M2 + M3 + M6  -> W[3] = [1, -1, 1, 0, 0, 1, 0]
    W = [
        [1,  0, 0, -1, 1, 0, -1],  # C00 (k=0,i=0)
        [0,  0, 1,  0, 1, 0,  0],  # C01 (k=0,i=1)
        [0,  1, 0,  1, 0, 0,  0],  # C10 (k=1,i=0)
        [1, -1, 1,  0, 0, 1,  0],  # C11 (k=1,i=1)
    ]
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
