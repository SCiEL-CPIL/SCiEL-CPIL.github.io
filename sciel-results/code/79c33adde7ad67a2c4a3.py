"""Matrix multiplication 2x3x8 rank decomposition."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    rank = 44
    U = [[0] * rank for _ in range(n * m)]   # shape (6, 44)
    V = [[0] * rank for _ in range(m * p)]   # shape (24, 44)
    W = [[0] * rank for _ in range(p * n)]   # shape (16, 44)

    col = 0

    def emit_strassen(i0, j0, k0):
        nonlocal col
        # 2x2x2 Strassen block: i in {i0,i0+1}, j in {j0,j0+1}, k in {k0,k0+1}
        # m1 = (A00+A11)(B00+B11)
        U[(i0+0)*m + (j0+0)][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[(j0+0)*p + (k0+0)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[(k0+0)*n + (i0+0)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1

        # m2 = (A10+A11)B00
        U[(i0+1)*m + (j0+0)][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[(j0+0)*p + (k0+0)][col] += 1
        W[(k0+0)*n + (i0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += -1
        col += 1

        # m3 = A00(B10-B11)
        U[(i0+0)*m + (j0+0)][col] += 1
        V[(j0+0)*p + (k0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += -1
        W[(k0+1)*n + (i0+0)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1

        # m4 = A11(B01-B00)
        U[(i0+1)*m + (j0+1)][col] += 1
        V[(j0+1)*p + (k0+0)][col] += 1
        V[(j0+0)*p + (k0+0)][col] += -1
        W[(k0+0)*n + (i0+0)][col] += 1
        W[(k0+0)*n + (i0+1)][col] += 1
        col += 1

        # m5 = (A00+A01)B11
        U[(i0+0)*m + (j0+0)][col] += 1
        U[(i0+0)*m + (j0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[(k0+0)*n + (i0+0)][col] += -1
        W[(k0+1)*n + (i0+0)][col] += 1
        col += 1

        # m6 = (A10-A00)(B00+B01)
        U[(i0+1)*m + (j0+0)][col] += 1
        U[(i0+0)*m + (j0+0)][col] += -1
        V[(j0+0)*p + (k0+0)][col] += 1
        V[(j0+0)*p + (k0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1

        # m7 = (A01-A11)(B10+B11)
        U[(i0+0)*m + (j0+1)][col] += 1
        U[(i0+1)*m + (j0+1)][col] += -1
        V[(j0+1)*p + (k0+0)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[(k0+0)*n + (i0+0)][col] += 1
        col += 1

    def emit_schoolbook_2x1x4(i0, j0, k0):
        nonlocal col
        for ii in range(2):
            for kk in range(4):
                U[(i0+ii)*m + (j0+0)][col] = 1
                V[(j0+0)*p + (k0+kk)][col] = 1
                W[(k0+kk)*n + (i0+ii)][col] = 1
                col += 1

    # Four Strassen 2x2x2 blocks for j in {0,1}, k blocks {0,1},{2,3},{4,5},{6,7}
    emit_strassen(0, 0, 0)
    emit_strassen(0, 0, 2)
    emit_strassen(0, 0, 4)
    emit_strassen(0, 0, 6)

    # Two schoolbook 2x1x4 blocks for j=2
    emit_schoolbook_2x1x4(0, 2, 0)
    emit_schoolbook_2x1x4(0, 2, 4)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
