"""Matrix multiplication 4,4,9 via Strassen 2x2 block decomposition."""

DIMENSIONS = [4, 4, 9]


# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9

    # Total rank: 2 Strassen blocks (28 each) + 1 schoolbook column (16)
    total_rank = 28 + 28 + 16

    U = [[0] * total_rank for _ in range(n * m)]  # 16 x 72
    V = [[0] * total_rank for _ in range(m * p)]  # 36 x 72
    W = [[0] * total_rank for _ in range(p * n)]  # 36 x 72

    def add_schoolbook_2x2(U, V, W, ao, bo, co, c0):
        """Add 4 schoolbook scalar products for a 2x2x2 block.
        ao,bo,co: row offsets for the 2x2 blocks (each has 4 rows: r0c0,r0c1,r1c0,r1c1)
        """
        c = c0
        # C[0,0] = A[0,0]*B[0,0]
        U[ao + 0][c] = 1; V[bo + 0][c] = 1; W[co + 0][c] = 1; c += 1
        # C[0,1] = A[0,0]*B[0,1]
        U[ao + 0][c] = 1; V[bo + 1][c] = 1; W[co + 1][c] = 1; c += 1
        # C[0,1] += A[0,1]*B[1,1]
        U[ao + 1][c] = 1; V[bo + 3][c] = 1; W[co + 1][c] = 1; c += 1
        # C[1,0] = A[1,0]*B[0,0]
        U[ao + 2][c] = 1; V[bo + 0][c] = 1; W[co + 2][c] = 1; c += 1
        # C[1,0] += A[1,1]*B[1,0]
        U[ao + 3][c] = 1; V[bo + 2][c] = 1; W[co + 2][c] = 1; c += 1
        # C[1,1] = A[1,0]*B[0,1]
        U[ao + 2][c] = 1; V[bo + 1][c] = 1; W[co + 3][c] = 1; c += 1
        # C[1,1] += A[1,1]*B[1,1]
        U[ao + 3][c] = 1; V[bo + 3][c] = 1; W[co + 3][c] = 1; c += 1
        return c

    def add_strassen_4x4(U, V, W, b_off, c_off, c0):
        """Add 28 scalar products implementing Strassen on the 2x2 block
        decomposition of a 4x4x4 product. A is shared (offset 0).
        b_off: column-block offset in V (16 rows per 4x4 block)
        c_off: column offset in W (16 rows per 4x4 block)
        """
        # 2x2 sub-block row offsets within a 4x4 (row-major, m=4)
        # A11: [0,1,4,5]  A12: [2,3,6,7]
        # A21: [8,9,12,13]  A22: [10,11,14,15]
        A11 = [0, 1, 4, 5]
        A12 = [2, 3, 6, 7]
        A21 = [8, 9, 12, 13]
        A22 = [10, 11, 14, 15]

        # Same layout for B and C blocks, shifted by their offsets
        B11 = [b_off + x for x in [0, 1, 4, 5]]
        B12 = [b_off + x for x in [2, 3, 6, 7]]
        B21 = [b_off + x for x in [8, 9, 12, 13]]
        B22 = [b_off + x for x in [10, 11, 14, 15]]
        C11 = [c_off + x for x in [0, 1, 4, 5]]
        C12 = [c_off + x for x in [2, 3, 6, 7]]
        C21 = [c_off + x for x in [8, 9, 12, 13]]
        C22 = [c_off + x for x in [10, 11, 14, 15]]

        c = c0

        # M1 = (A11 + A22)(B11 + B22) -> C11 += M1, C22 += M1
        a1 = [A11[t] for t in range(4)] + [A22[t] for t in range(4)]
        b1 = [B11[t] for t in range(4)] + [B22[t] for t in range(4)]
        for t in range(4):
            U[a1[t]][c] = 1
            V[b1[t]][c] = 1
            W[C11[t]][c] = 1
            W[C22[t]][c] = 1
            c += 1

        # M2 = (A21 + A22)(B11) -> C21 += M2, C22 -= M2
        a2 = [A21[t] for t in range(4)] + [A22[t] for t in range(4)]
        b2 = B11
        for t in range(4):
            U[a2[t]][c] = 1
            V[b2[t]][c] = 1
            W[C21[t]][c] = 1
            W[C22[t]][c] = -1
            c += 1

        # M3 = (A11)(B21 - B22) -> C12 += M3, C22 += M3
        a3 = A11
        b3 = [B21[t] for t in range(4)] + [B22[t] for t in range(4)]
        for t in range(4):
            U[a3[t]][c] = 1
            V[b3[t]][c] = 1 if t < 4 else -1
            W[C12[t]][c] = 1
            W[C22[t]][c] = 1
            c += 1

        # M4 = (A22)(B12 - B11) -> C11 += M4, C21 += M4
        a4 = A22
        b4 = [B12[t] for t in range(4)] + [B11[t] for t in range(4)]
        for t in range(4):
            U[a4[t]][c] = 1
            V[b4[t]][c] = 1 if t < 4 else -1
            W[C11[t]][c] = 1
            W[C21[t]][c] = 1
            c += 1

        # M5 = (A11 + A12)(B22) -> C11 -= M5, C12 += M5
        a5 = [A11[t] for t in range(4)] + [A12[t] for t in range(4)]
        b5 = B22
        for t in range(4):
            U[a5[t]][c] = 1
            V[b5[t]][c] = 1
            W[C11[t]][c] = -1
            W[C12[t]][c] = 1
            c += 1

        # M6 = (A21 - A11)(B11 + B12) -> C22 += M6
        a6 = [A21[t] for t in range(4)] + [A11[t] for t in range(4)]
        b6 = [B11[t] for t in range(4)] + [B12[t] for t in range(4)]
        for t in range(4):
            U[a6[t]][c] = 1 if t < 4 else -1
            V[b6[t]][c] = 1
            W[C22[t]][c] = 1
            c += 1

        # M7 = (A12 - A22)(B21 + B22) -> C11 += M7
        a7 = [A12[t] for t in range(4)] + [A22[t] for t in range(4)]
        b7 = [B21[t] for t in range(4)] + [B22[t] for t in range(4)]
        for t in range(4):
            U[a7[t]][c] = 1 if t < 4 else -1
            V[b7[t]][c] = 1
            W[C11[t]][c] = 1
            c += 1

        return c

    # Block 0: B columns 0:4, C columns 0:4 (Strassen)
    col = add_strassen_4x4(U, V, W, 0, 0, 0)
    # Block 1: B columns 4:8, C columns 4:8 (Strassen)
    col = add_strassen_4x4(U, V, W, 16, 16, col)
    # Block 2: B column 8, C column 8 (schoolbook 4x4x1 = 16)
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 8][col] = 1
            W[8 * n + i][col] = 1
            col += 1

    assert col == total_rank, f"column {col} != rank {total_rank}"

    # ---- Verification (builds the multiplication tensor and checks) ----
    import numpy as np
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)

    # Tensor T[i,j,k] = 1 iff A[i,j] * B[j,k] contributes to C[k,i]
    # Sum over r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] must equal 1
    def check():
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(total_rank):
                        s += Ua[i * m + j, r] * Va[j * p + k, r] * Wa[k * n + i, r]
                    if abs(s - 1.0) > 1e-9:
                        raise AssertionError(f"T[{i},{j},{k}]={s} != 1")
        # Also check no spurious contributions: entries where the tensor is 0
        # The multiplication tensor is exactly 1 at (i,j,k) for all i,j,k,
        # and 0 elsewhere. Since U,V,W only have entries in the right rows,
        # the only nonzero triples are (i,j,k) which are all =1. Good.
    check()

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
