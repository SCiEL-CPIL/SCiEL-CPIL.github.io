def construct():
    n, m, p = 3, 4, 4
    rank = 32
    U = [[0]*rank for _ in range(n*m)]  # 12 x 32
    V = [[0]*rank for _ in range(m*p)]  # 16 x 32
    W = [[0]*rank for _ in range(p*n)]  # 12 x 32

    col = 0

    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c, start_col):
        """Add Strassen decomposition for 2x2 * 2x2 -> 2x2."""
        c = start_col
        # m1 = (a00+a11)(b00+b11) -> c00 += m1, c11 += m1
        U[a_r*m + a_c][c] += 1
        U[(a_r+1)*m + (a_c+1)][c] += 1
        V[b_r*p + b_c][c] += 1
        V[(b_r+1)*p + (b_c+1)][c] += 1
        W[c_c*n + c_r][c] += 1
        W[(c_c+1)*n + (c_r+1)][c] += 1
        c += 1

        # m2 = (a10+a11)b00 -> c10 += m2, c11 -= m2
        U[(a_r+1)*m + a_c][c] += 1
        U[(a_r+1)*m + (a_c+1)][c] += 1
        V[b_r*p + b_c][c] += 1
        W[c_c*n + (c_r+1)][c] += 1
        W[(c_c+1)*n + (c_r+1)][c] -= 1
        c += 1

        # m3 = a00(b01-b11) -> c01 += m3, c11 += m3
        U[a_r*m + a_c][c] += 1
        V[b_r*p + (b_c+1)][c] += 1
        V[(b_r+1)*p + (b_c+1)][c] -= 1
        W[(c_c+1)*n + c_r][c] += 1
        W[(c_c+1)*n + (c_r+1)][c] += 1
        c += 1

        # m4 = a11(b10-b00) -> c00 += m4, c10 += m4
        U[(a_r+1)*m + (a_c+1)][c] += 1
        V[(b_r+1)*p + b_c][c] += 1
        V[b_r*p + b_c][c] -= 1
        W[c_c*n + c_r][c] += 1
        W[c_c*n + (c_r+1)][c] += 1
        c += 1

        # m5 = (a00+a01)b11 -> c00 -= m5, c01 += m5
        U[a_r*m + a_c][c] += 1
        U[a_r*m + (a_c+1)][c] += 1
        V[(b_r+1)*p + (b_c+1)][c] += 1
        W[c_c*n + c_r][c] -= 1
        W[(c_c+1)*n + c_r][c] += 1
        c += 1

        # m6 = (a10-a00)(b00+b01) -> c11 += m6
        U[(a_r+1)*m + a_c][c] += 1
        U[a_r*m + a_c][c] -= 1
        V[b_r*p + b_c][c] += 1
        V[b_r*p + (b_c+1)][c] += 1
        W[(c_c+1)*n + (c_r+1)][c] += 1
        c += 1

        # m7 = (a01-a11)(b10+b11) -> c00 += m7
        U[a_r*m + (a_c+1)][c] += 1
        U[(a_r+1)*m + (a_c+1)][c] -= 1
        V[(b_r+1)*p + b_c][c] += 1
        V[(b_r+1)*p + (b_c+1)][c] += 1
        W[c_c*n + c_r][c] += 1
        c += 1

        return c

    # Top 2 rows of A (rows 0,1) times B (4x4)
    # C[0:2, 0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]
    col = add_strassen(0, 0, 0, 0, 0, 0, col)  # A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    col = add_strassen(0, 2, 2, 0, 0, 0, col)  # A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]

    # C[0:2, 2:4] = A[0:2,0:2]*B[0:2,2:4] + A[0:2,2:4]*B[2:4,2:4]
    col = add_strassen(0, 0, 0, 2, 0, 2, col)  # A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    col = add_strassen(0, 2, 2, 2, 0, 2, col)  # A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]

    # Bottom row of A (row 2) times B (4x4)
    # C[2,k] = sum_j A[2,j] * B[j,k] for k=0,1,2,3
    # Use rank-4 decomposition: one term per output column k
    for k in range(4):
        # u: 1 at A[2,0], A[2,1], A[2,2], A[2,3] (indices 8,9,10,11)
        # v: 1 at B[0,k], B[1,k], B[2,k], B[3,k] (indices k, 4+k, 8+k, 12+k)
        # w: 1 at C[2,k] (index k*3+2)
        for j in range(4):
            U[2*m + j][col] += 1
            V[j*p + k][col] += 1
        W[k*n + 2][col] += 1
        col += 1

    return {"U": U, "V": V, "W": W}
