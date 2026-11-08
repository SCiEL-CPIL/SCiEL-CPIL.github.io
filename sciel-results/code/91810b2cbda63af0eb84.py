"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    RANK = 54  # 2 * (7 + 5*4)
    U = [[0] * RANK for _ in range(n * m)]  # 8 x 54
    V = [[0] * RANK for _ in range(m * p)]  # 28 x 54
    W = [[0] * RANK for _ in range(p * n)]  # 14 x 54

    def set(row, col, val):
        if val != 0:
            U[row][col] += val

    def setv(row, col, val):
        if val != 0:
            V[row][col] += val

    def setw(row, col, val):
        if val != 0:
            W[row][col] += val

    def build_block(j0, c0):
        # Strassen 7 mults for {j0, j0+1} x {k=0,1}
        # M1 = (A00+A11)(B00+B11)
        set(j0 * 2 + 0, c0 + 0, 1)
        set(j0 * 2 + 1, c0 + 0, 1)
        setv(j0 * 7 + 0, c0 + 0, 1)
        setv(j0 * 7 + 1, c0 + 0, 1)
        setw(0 * 2 + 0, c0 + 0, 1)
        setw(1 * 2 + 1, c0 + 0, 1)
        # M2 = (A10+A11)B01
        set(j0 * 2 + 0, c0 + 1, 1)
        set(j0 * 2 + 1, c0 + 1, 1)
        setv(j0 * 7 + 1, c0 + 1, 1)
        setw(1 * 2 + 0, c0 + 1, 1)
        # M3 = A00(B10-B11)
        set(j0 * 2 + 0, c0 + 2, 1)
        setv(j0 * 7 + 0, c0 + 2, -1)
        setv(j0 * 7 + 1, c0 + 2, 1)
        setw(0 * 2 + 1, c0 + 2, 1)
        # M4 = A11(B01-B00)
        set(j0 * 2 + 1, c0 + 3, 1)
        setv(j0 * 7 + 0, c0 + 3, -1)
        setv(j0 * 7 + 1, c0 + 3, 1)
        setw(0 * 2 + 0, c0 + 3, 1)
        setw(1 * 2 + 0, c0 + 3, 1)
        # M5 = (A00+A01)B11
        set(j0 * 2 + 0, c0 + 4, 1)
        set(j0 * 2 + 1, c0 + 4, 1)
        setv(j0 * 7 + 1, c0 + 4, 1)
        setw(0 * 2 + 0, c0 + 4, -1)
        setw(0 * 2 + 1, c0 + 4, 1)
        # M6 = (A10-A00)(B00+B10)
        set(j0 * 2 + 0, c0 + 5, -1)
        set(j0 * 2 + 1, c0 + 5, 1)
        setv(j0 * 7 + 0, c0 + 5, 1)
        setv(j0 * 7 + 1, c0 + 5, 1)
        setw(1 * 2 + 1, c0 + 5, -1)
        # M7 = (A01-A11)(B11+B01)
        set(j0 * 2 + 0, c0 + 6, 1)
        set(j0 * 2 + 1, c0 + 6, -1)
        setv(j0 * 7 + 0, c0 + 6, 1)
        setv(j0 * 7 + 1, c0 + 6, 1)
        setw(0 * 2 + 0, c0 + 6, 1)

        # Schoolbook 20 mults for k=2..6
        c = c0 + 7
        for k in range(2, p):
            # A00*B_k
            set(j0 * 2 + 0, c, 1)
            setv(j0 * 7 + k, c, 1)
            setw(k * 2 + 0, c, 1)
            c += 1
            # A01*B_k
            set(j0 * 2 + 1, c, 1)
            setv(j0 * 7 + k, c, 1)
            setw(k * 2 + 0, c, 1)
            c += 1
            # A10*B_k
            set(j0 * 2 + 0, c, 1)
            setv((j0 + 1) * 7 + k, c, 1)
            setw(k * 2 + 1, c, 1)
            c += 1
            # A11*B_k
            set(j0 * 2 + 1, c, 1)
            setv((j0 + 1) * 7 + k, c, 1)
            setw(k * 2 + 1, c, 1)
            c += 1

    # Block 1: j in {0,1}, columns 0..26
    build_block(0, 0)
    # Block 2: j in {2,3}, columns 27..53
    build_block(2, 27)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
