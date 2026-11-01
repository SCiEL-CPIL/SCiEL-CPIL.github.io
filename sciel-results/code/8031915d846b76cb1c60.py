"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=13
    # Use Strassen for 2x2 blocks of B (columns 0-11), schoolbook for last column
    num_pairs = 6  # columns 0..11 in pairs
    rank = num_pairs * 7 + 4  # 46

    U = [[0] * rank for _ in range(n * m)]  # 4 x 46
    V = [[0] * rank for _ in range(m * p)]  # 26 x 46
    W = [[0] * rank for _ in range(p * n)]  # 26 x 46

    col = 0
    for t in range(num_pairs):
        k0 = 2 * t
        k1 = 2 * t + 1

        # Strassen term 1: M1 = (a+d)(e+h)
        # U: A[0,0] + A[1,0]
        U[0][col] += 1  # A[0,0]
        U[2][col] += 1  # A[1,0]
        # V: B[0,k0] + B[0,k1]
        V[0 * p + k0][col] += 1
        V[0 * p + k1][col] += 1
        # W: C[0,k0] += 1, C[1,k1] -= 1, C[0,k1] += 1, C[1,k0] += 1
        W[k0 * n + 0][col] += 1   # C[0,k0]
        W[k1 * n + 1][col] -= 1   # C[1,k1]
        W[k1 * n + 0][col] += 1   # C[0,k1]
        W[k0 * n + 1][col] += 1   # C[1,k0]
        col += 1

        # Strassen term 2: M2 = (c+d)e = (A[1,0]+A[1,1]) * B[0,k0]
        U[2][col] += 1  # A[1,0]
        U[3][col] += 1  # A[1,1]
        V[0 * p + k0][col] += 1
        # W: C[1,k0] += 1, C[1,k1] -= 1
        W[k0 * n + 1][col] += 1   # C[1,k0]
        W[k1 * n + 1][col] -= 1   # C[1,k1]
        col += 1

        # Strassen term 3: M3 = a(f-h) = A[0,0] * (B[0,k1] - B[1,k1])
        U[0][col] += 1  # A[0,0]
        V[0 * p + k1][col] += 1
        V[1 * p + k1][col] -= 1
        # W: C[0,k1] += 1, C[1,k1] += 1
        W[k1 * n + 0][col] += 1   # C[0,k1]
        W[k1 * n + 1][col] += 1   # C[1,k1]
        col += 1

        # Strassen term 4: M4 = d(g-e) = A[1,1] * (B[1,k0] - B[0,k0])
        U[3][col] += 1  # A[1,1]
        V[1 * p + k0][col] += 1
        V[0 * p + k0][col] -= 1
        # W: C[1,k0] += 1, C[0,k0] += 1
        W[k0 * n + 1][col] += 1   # C[1,k0]
        W[k0 * n + 0][col] += 1   # C[0,k0]
        col += 1

        # Strassen term 5: M5 = (a+b)h = (A[0,0]+A[0,1]) * B[1,k1]
        U[0][col] += 1  # A[0,0]
        U[1][col] += 1  # A[0,1]
        V[1 * p + k1][col] += 1
        # W: C[0,k0] -= 1, C[0,k1] += 1
        W[k0 * n + 0][col] -= 1   # C[0,k0]
        W[k1 * n + 0][col] += 1   # C[0,k1]
        col += 1

        # Strassen term 6: M6 = (c-a)(e+f) = (A[1,0]-A[0,0]) * (B[0,k0]+B[0,k1])
        U[2][col] += 1  # A[1,0]
        U[0][col] -= 1  # A[0,0]
        V[0 * p + k0][col] += 1
        V[0 * p + k1][col] += 1
        # W: C[1,k1] += 1
        W[k1 * n + 1][col] += 1   # C[1,k1]
        col += 1

        # Strassen term 7: M7 = (b-d)(g+h) = (A[0,1]-A[1,1]) * (B[1,k0]+B[1,k1])
        U[1][col] += 1  # A[0,1]
        U[3][col] -= 1  # A[1,1]
        V[1 * p + k0][col] += 1
        V[1 * p + k1][col] += 1
        # W: C[0,k0] += 1
        W[k0 * n + 0][col] += 1   # C[0,k0]
        col += 1

    # Last column (k=12): schoolbook 4 terms
    k = 12
    # C[0,k] += A[0,0]*B[0,k]
    U[0][col] += 1
    V[0 * p + k][col] += 1
    W[k * n + 0][col] += 1
    col += 1
    # C[0,k] += A[0,1]*B[1,k]
    U[1][col] += 1
    V[1 * p + k][col] += 1
    W[k * n + 0][col] += 1
    col += 1
    # C[1,k] += A[1,0]*B[0,k]
    U[2][col] += 1
    V[0 * p + k][col] += 1
    W[k * n + 1][col] += 1
    col += 1
    # C[1,k] += A[1,1]*B[1,k]
    U[3][col] += 1
    V[1 * p + k][col] += 1
    W[k * n + 1][col] += 1
    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
