"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=10
    # Rank 55 decomposition using 5 blocks of M_{2,3,2} (rank 11 each)
    rank = 55
    U = [[0] * rank for _ in range(n * m)]  # 6 x 55
    V = [[0] * rank for _ in range(m * p)]  # 30 x 55
    W = [[0] * rank for _ in range(p * n)]  # 20 x 55
    
    # Loop over 5 blocks of 2 columns each
    for b in range(5):
        c0 = 2 * b
        c1 = 2 * b + 1
        base_col = b * 11
        
        # Helper to add to matrix
        def add_U(row, val):
            U[row][base_col] += val
        def add_V(row, val):
            V[row][base_col] += val
        def add_W(row, val):
            W[row][base_col] += val

        # Strassen Part (7 terms)
        # M1 = (A00+A11)(B00+B11) -> A indices 0,4; B indices c0, 10+c1
        # Note: B indices in V are j*p + k.
        # B00 is j=0, k=c0 -> c0
        # B11 is j=1, k=c1 -> 10+c1
        # C11 (2c0) += 1, C22 (2c1+1) += 1
        add_U(0, 1); add_U(4, 1)
        add_V(c0, 1); add_V(10+c1, 1)
        add_W(2*c0, 1); add_W(2*c1+1, 1)

        # M2 = (A21+A22)B00 -> A indices 3,4; B index c0
        # C21 (2c0+1) += 1, C22 (2c1+1) -= 1
        add_U(3, 1); add_U(4, 1)
        add_V(c0, 1)
        add_W(2*c0+1, 1); add_W(2*c1+1, -1)

        # M3 = A00(B01-B11) -> A index 0; B indices c1, 10+c1
        # C12 (2c1) += 1, C22 (2c1+1) += 1
        add_U(0, 1)
        add_V(c1, 1); add_V(10+c1, -1)
        add_W(2*c1, 1); add_W(2*c1+1, 1)

        # M4 = A11(B10-B00) -> A index 4; B indices 10+c0, c0
        # C11 (2c0) += 1, C21 (2c0+1) += 1
        add_U(4, 1)
        add_V(10+c0, 1); add_V(c0, -1)
        add_W(2*c0, 1); add_W(2*c0+1, 1)

        # M5 = (A00+A01)B11 -> A indices 0,1; B index 10+c1
        # C11 (2c0) -= 1, C12 (2c1) += 1
        add_U(0, 1); add_U(1, 1)
        add_V(10+c1, 1)
        add_W(2*c0, -1); add_W(2*c1, 1)

        # M6 = (A21-A00)(B00+B01) -> A indices 3,0; B indices c0, c1
        # C22 (2c1+1) += 1
        add_U(3, 1); add_U(0, -1)
        add_V(c0, 1); add_V(c1, 1)
        add_W(2*c1+1, 1)

        # M7 = (A01-A11)(B10+B11) -> A indices 1,4; B indices 10+c0, 10+c1
        # C11 (2c0) += 1
        add_U(1, 1); add_U(4, -1)
        add_V(10+c0, 1); add_V(10+c1, 1)
        add_W(2*c0, 1)

        # Standard Part (4 terms)
        # N1 = A02 B20 -> A index 2; B index 20+c0
        # C00 (2c0) += 1
        add_U(2, 1)
        add_V(20+c0, 1)
        add_W(2*c0, 1)

        # N2 = A02 B21 -> A index 2; B index 20+c1
        # C01 (2c1) += 1
        add_U(2, 1)
        add_V(20+c1, 1)
        add_W(2*c1, 1)

        # N3 = A12 B20 -> A index 5; B index 20+c0
        # C10 (2c0+1) += 1
        add_U(5, 1)
        add_V(20+c0, 1)
        add_W(2*c0+1, 1)

        # N4 = A12 B21 -> A index 5; B index 20+c1
        # C11 (2c1+1) += 1
        add_U(5, 1)
        add_V(20+c1, 1)
        add_W(2*c1+1, 1)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
