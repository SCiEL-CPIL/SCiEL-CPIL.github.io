DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 4, 5
    R = 36
    U = [[0]*R for _ in range(n*m)]
    V = [[0]*R for _ in range(m*p)]
    W = [[0]*R for _ in range(p*n)]
    k = [0]

    def aidx(r, col):
        return r * m + col
    def bidx(r, col):
        return r * p + col
    def cidx(r, col):
        return col * n + r

    def strassen2x2(ar, ac, br, bc, cr, cc):
        A00, A01 = aidx(ar, ac), aidx(ar, ac + 1)
        A10, A11 = aidx(ar + 1, ac), aidx(ar + 1, ac + 1)
        B00, B01 = bidx(br, bc), bidx(br, bc + 1)
        B10, B11 = bidx(br + 1, bc), bidx(br + 1, bc + 1)
        C00, C01 = cidx(cr, cc), cidx(cr, cc + 1)
        C10, C11 = cidx(cr + 1, cc), cidx(cr + 1, cc + 1)

        idx = k[0]
        U[A00][idx] += 1; U[A11][idx] += 1
        V[B00][idx] += 1; V[B11][idx] += 1
        W[C00][idx] += 1; W[C11][idx] += 1
        k[0] += 1

        idx = k[0]
        U[A10][idx] += 1; U[A11][idx] += 1
        V[B00][idx] += 1
        W[C10][idx] += 1; W[C11][idx] += -1
        k[0] += 1

        idx = k[0]
        U[A00][idx] += 1
        V[B01][idx] += 1; V[B11][idx] += -1
        W[C01][idx] += 1; W[C11][idx] += 1
        k[0] += 1

        idx = k[0]
        U[A11][idx] += 1
        V[B10][idx] += 1; V[B00][idx] += -1
        W[C00][idx] += 1; W[C10][idx] += 1
        k[0] += 1

        idx = k[0]
        U[A00][idx] += 1; U[A01][idx] += 1
        V[B11][idx] += 1
        W[C00][idx] += -1; W[C01][idx] += 1
        k[0] += 1

        idx = k[0]
        U[A10][idx] += 1; U[A00][idx] += -1
        V[B00][idx] += 1; V[B01][idx] += 1
        W[C11][idx] += 1
        k[0] += 1

        idx = k[0]
        U[A01][idx] += 1; U[A11][idx] += -1
        V[B10][idx] += 1; V[B11][idx] += 1
        W[C00][idx] += 1
        k[0] += 1

    def schoolbook2x1(ar, ac, br, bc, cr, cc):
        A00, A01 = aidx(ar, ac), aidx(ar, ac + 1)
        A10, A11 = aidx(ar + 1, ac), aidx(ar + 1, ac + 1)
        B0, B1 = bidx(br, bc), bidx(br + 1, bc)
        C0, C1 = cidx(cr, cc), cidx(cr + 1, cc)

        idx = k[0]
        U[A00][idx] += 1; V[B0][idx] += 1; W[C0][idx] += 1
        k[0] += 1
        idx = k[0]
        U[A01][idx] += 1; V[B1][idx] += 1; W[C0][idx] += 1
        k[0] += 1
        idx = k[0]
        U[A10][idx] += 1; V[B0][idx] += 1; W[C1][idx] += 1
        k[0] += 1
        idx = k[0]
        U[A11][idx] += 1; V[B1][idx] += 1; W[C1][idx] += 1
        k[0] += 1

    strassen2x2(0, 0, 0, 0, 0, 0)
    strassen2x2(0, 0, 0, 2, 0, 2)
    strassen2x2(0, 2, 2, 0, 0, 0)
    strassen2x2(0, 2, 2, 2, 0, 2)
    schoolbook2x1(0, 0, 0, 4, 0, 4)
    schoolbook2x1(0, 2, 2, 4, 0, 4)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
