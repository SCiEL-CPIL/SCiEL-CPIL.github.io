"""Rank-44 decomposition for 2x3x8 matrix multiplication.
16 schoolbook + 28 Strassen terms (7 products x 4 sub-blocks)."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 8
    R = 44
    U = [[0.0] * R for _ in range(n * m)]   # 6 x 44
    V = [[0.0] * R for _ in range(m * p)]   # 24 x 44
    W = [[0.0] * R for _ in range(p * n)]   # 16 x 44

    col = 0
    # 4 non-overlapping 2x2 sub-blocks in (j,k) space
    subblocks = [(0, 0), (0, 2), (0, 4), (0, 6)]

    for (j0, k0) in subblocks:
        j1 = j0 + 1
        k1 = k0 + 1

        # M1 = (A[0,j0]+A[1,j1]) * (B[j0,k0]+B[j1,k1])
        U[0*m+j0][col] = 1.0
        U[1*m+j1][col] = 1.0
        V[j0*p+k0][col] = 1.0
        V[j1*p+k1][col] = 1.0
        W[k0*n+0][col] = 1.0   # C[0,k0] += M1
        W[k1*n+1][col] = 1.0   # C[1,k1] += M1
        col += 1

        # M2 = (A[1,j0]+A[1,j1]) * B[j0,k0]
        U[1*m+j0][col] = 1.0
        U[1*m+j1][col] = 1.0
        V[j0*p+k0][col] = 1.0
        W[k1*n+0][col] = 1.0   # C[0,k1] += M2
        W[k1*n+1][col] = -1.0  # C[1,k1] -= M2
        col += 1

        # M3 = A[0,j0] * (B[j1,k0]-B[j1,k1])
        U[0*m+j0][col] = 1.0
        V[j1*p+k0][col] = 1.0
        V[j1*p+k1][col] = -1.0
        W[k0*n+1][col] = 1.0   # C[1,k0] += M3
        W[k1*n+1][col] = 1.0   # C[1,k1] += M3
        col += 1

        # M4 = A[1,j1] * (B[j0,k1]-B[j0,k0])
        U[1*m+j1][col] = 1.0
        V[j0*p+k1][col] = 1.0
        V[j0*p+k0][col] = -1.0
        W[k0*n+0][col] = 1.0   # C[0,k0] += M4
        col += 1

        # M5 = (A[0,j0]+A[0,j1]) * B[j1,k1]
        U[0*m+j0][col] = 1.0
        U[0*m+j1][col] = 1.0
        V[j1*p+k1][col] = 1.0
        W[k0*n+0][col] = -1.0  # C[0,k0] -= M5
        W[k1*n+0][col] = 1.0   # C[0,k1] += M5
        col += 1

        # M6 = (A[1,j0]-A[0,j0]) * (B[j0,k0]+B[j0,k1])
        U[1*m+j0][col] = 1.0
        U[0*m+j0][col] = -1.0
        V[j0*p+k0][col] = 1.0
        V[j0*p+k1][col] = 1.0
        W[k1*n+1][col] = 1.0   # C[1,k1] += M6
        col += 1

        # M7 = (A[1,j1]-A[1,j0]) * (B[j1,k0]+B[j1,k1])
        U[1*m+j1][col] = 1.0
        U[1*m+j0][col] = -1.0
        V[j1*p+k0][col] = 1.0
        V[j1*p+k1][col] = 1.0
        W[k0*n+0][col] = -1.0  # C[0,k0] -= M7
        W[k0*n+1][col] = 1.0   # C[1,k0] += M7
        col += 1

    # Schoolbook for j=2 (16 terms)
    for i in range(n):
        for k in range(p):
            U[i*m+2][col] = 1.0
            V[2*p+k][col] = 1.0
            W[k*n+i][col] = 1.0
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
