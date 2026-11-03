def construct():
    n, m, p = 3, 4, 4
    rank = 44
    U = [[0]*rank for _ in range(n*m)]  # 12 x 44
    V = [[0]*rank for _ in range(m*p)]  # 16 x 44
    W = [[0]*rank for _ in range(p*n)]  # 12 x 44

    col = 0

    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c, start_col):
        """Add Strassen decomposition for 2x2 * 2x2 -> 2x2."""
        c = start_col
        
        # m1 = (a00 + a11)(b00 + b11)
        U[a_r*m + a_c][c] += 1
        U[(a_r+1)*m + (a_c+1)][c] += 1
        V[b_r*p + b_c][c] += 1
        V[(b_r+1)*p + (b_c+1)][c] += 1
        W[c_c*n + c_r][c] += 1
        W[(c_c+1)*n + (c_r+1)][c] += 1
        c += 1

        # m2 = (a10 + a11) b00
        U[(a_r+1)*m + a_c][c] += 1
        U[(a_r+1)*m + (a_c+1)][c] += 1
        V[b_r*p + b_c][c] += 1
        W[c_c*n + (c_r+1)][c] += 1
        W[(c_c+1)*n + (c_r+1)][c] -= 1
        c += 1

        # m3 = a00 (b01 - b11)
        U[a_r*m + a_c][c] += 1
        V[b_r*p + (b_c+1)][c] += 1
        V[(b_r+1)*p + (b_c+1)][c] -= 1
        W[(c_c+1)*n + c_r][c] += 1
        W[(c_c+1)*n + (c_r+1)][c] += 1
        c += 1

        # m4 = a11 (b10 - b00)
        U[(a_r+1)*m + (a_c+1)][c] += 1
        V[(b_r+1)*p + b_c][c] += 1
        V[b_r*p + b_c][c] -= 1
        W[c_c*n + c_r][c] += 1
        W[c_c*n + (c_r+1)][c] += 1
        c += 1

        # m5 = (a00 + a01) b11
        U[a_r*m + a_c][c] += 1
        U[a_r*m + (a_c+1)][c] += 1
        V[(b_r+1)*p + (b_c+1)][c] += 1
        W[c_c*n + c_r][c] -= 1
        W[(c_c+1)*n + c_r][c] += 1
        c += 1

        # m6 = (a10 - a00)(b00 + b01)
        U[(a_r+1)*m + a_c][c] += 1
        U[a_r*m + a_c][c] -= 1
        V[b_r*p + b_c][c] += 1
        V[b_r*p + (b_c+1)][c] += 1
        W[(c_c+1)*n + (c_r+1)][c] += 1
        c += 1

        # m7 = (a01 - a11)(b10 + b11)
        U[a_r*m + (a_c+1)][c] += 1
        U[(a_r+1)*m + (a_c+1)][c] -= 1
        V[(b_r+1)*p + b_c][c] += 1
        V[(b_r+1)*p + (b_c+1)][c] += 1
        W[c_c*n + c_r][c] += 1
        c += 1

        return c

    def add_trivial_1x4x2(a_r, b_c, start_col):
        """Add trivial decomposition for 1x4 * 4x2 -> 1x2."""
        c = start_col
        for j in range(4):
            # Term for col b_c
            U[a_r*m + j][c] += 1
            V[j*p + b_c][c] += 1
            W[b_c*n + a_r][c] += 1
            c += 1
            # Term for col b_c+1
            U[a_r*m + j][c] += 1
            V[j*p + (b_c+1)][c] += 1
            W[(b_c+1)*n + a_r][c] += 1
            c += 1
        return c

    # Left Side (cols 0,1 of C)
    col = add_strassen(0, 0, 0, 0, 0, 0, col)
    col = add_strassen(0, 2, 2, 0, 0, 0, col)
    col = add_trivial_1x4x2(2, 0, col)

    # Right Side (cols 2,3 of C)
    col = add_strassen(0, 0, 0, 2, 0, 2, col)
    col = add_strassen(0, 2, 2, 2, 0, 2, col)
    col = add_trivial_1x4x2(2, 2, col)

    return {"U": U, "V": V, "W": W}
