"""Program for matrix multiplication 2x3 by 3x6 tensor decomposition."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank: 12 (for j=0, i in {0,1}, k in {0..5}) + 3 * 7 (for j=1,2 and 3 blocks of k)
    # Total rank = 12 + 21 = 33
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    # Part 1: j=0, scalar multiplication
    # A is 2x3, B is 3x6.
    # When j=0, A[i,0] * B[0,k] contributes to C[i,k].
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    # Part 2: j=1,2, split into 3 blocks of 2 for k
    # A_R is A[:, 1:3] (2x2), B blocks are B[1:3, k_start:k_start+2] (2x2)
    # Using Strassen's algorithm for 2x2x2
    # Strassen coefficients:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # We have 3 blocks for k: [0,1], [2,3], [4,5]
    for block in range(3):
        k0 = block * 2
        k1 = k0 + 1
        
        # A_R indices: j=1 (local 0), j=2 (local 1)
        # B block indices: j=1 (local 0), j=2 (local 1)
        # C block indices: i=0 (local 0), i=1 (local 1), k=k0 (local 0), k=k1 (local 1)
        
        # M1: (A00+A11)(B00+B11)
        # A00: i=0, j=1 -> U[0*3+1]
        # A11: i=1, j=2 -> U[1*3+2]
        # B00: j=1, k=k0 -> V[1*6+k0]
        # B11: j=2, k=k1 -> V[2*6+k1]
        # C00: i=0, k=k0 -> W[k0*2+0]
        # C11: i=1, k=k1 -> W[k1*2+1]
        U[0 * m + 1][col] = 1
        U[1 * m + 2][col] = 1
        V[1 * p + k0][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1

        # M2: (A10+A11)B00
        # A10: i=1, j=1 -> U[1*3+1]
        # A11: i=1, j=2 -> U[1*3+2]
        # B00: j=1, k=k0 -> V[1*6+k0]
        # C10: i=1, k=k0 -> W[k0*2+1]
        U[1 * m + 1][col] = 1
        U[1 * m + 2][col] = 1
        V[1 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1

        # M3: A00(B01-B11)
        # A00: i=0, j=1 -> U[0*3+1]
        # B01: j=1, k=k1 -> V[1*6+k1]
        # B11: j=2, k=k1 -> V[2*6+k1]
        # C01: i=0, k=k1 -> W[k1*2+0]
        U[0 * m + 1][col] = 1
        V[1 * p + k1][col] = 1
        V[2 * p + k1][col] = -1
        W[k1 * n + 0][col] = 1
        col += 1

        # M4: A11(B10-B00)
        # A11: i=1, j=2 -> U[1*3+2]
        # B10: j=2, k=k0 -> V[2*6+k0]
        # B00: j=1, k=k0 -> V[1*6+k0]
        # C00: i=0, k=k0 -> W[k0*2+0]
        # C10: i=1, k=k0 -> W[k0*2+1]
        U[1 * m + 2][col] = 1
        V[2 * p + k0][col] = 1
        V[1 * p + k0][col] = -1
        W[k0 * n + 0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1

        # M5: (A00+A01)B11
        # A00: i=0, j=1 -> U[0*3+1]
        # A01: i=0, j=2 -> U[0*3+2]
        # B11: j=2, k=k1 -> V[2*6+k1]
        # C00: i=0, k=k0 -> W[k0*2+0] (with -1)
        # C01: i=0, k=k1 -> W[k1*2+0]
        U[0 * m + 1][col] = 1
        U[0 * m + 2][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = -1
        W[k1 * n + 0][col] = 1
        col += 1

        # M6: (A10-A00)(B00+B01)
        # A10: i=1, j=1 -> U[1*3+1]
        # A00: i=0, j=1 -> U[0*3+1]
        # B00: j=1, k=k0 -> V[1*6+k0]
        # B01: j=1, k=k1 -> V[1*6+k1]
        # C11: i=1, k=k1 -> W[k1*2+1]
        U[1 * m + 1][col] = 1
        U[0 * m + 1][col] = -1
        V[1 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1

        # M7: (A01-A11)(B10+B11)
        # A01: i=0, j=2 -> U[0*3+2]
        # A11: i=1, j=2 -> U[1*3+2]
        # B10: j=2, k=k0 -> V[2*6+k0]
        # B11: j=2, k=k1 -> V[2*6+k1]
        # C00: i=0, k=k0 -> W[k0*2+0]
        # C11: i=1, k=k1 -> W[k1*2+1]
        U[0 * m + 2][col] = 1
        U[1 * m + 2][col] = -1
        V[2 * p + k0][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
