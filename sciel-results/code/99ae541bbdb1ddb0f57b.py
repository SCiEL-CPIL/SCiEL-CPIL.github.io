"""Matrix multiplication 2x3x8: rank-44 exact tensor decomposition.

Uses block-partitioned Strassen + schoolbook construction.
Strategy:
  Split p=8 into two p=4 blocks.
  For each p=4 block:
    Split m=3 into m=2 (j=0,1) and m=1 (j=2).
    m=2, p=4: Split p=4 into two p=2. Use Strassen (rank 7) for each 2x2x2.
    m=1, p=4: Schoolbook rank 8.
    Total per p=4 block: 14 + 8 = 22.
  Total rank: 44.
"""

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 44
    U = [[0] * rank for _ in range(n * m)]   # 6 x 44
    V = [[0] * rank for _ in range(m * p)]   # 24 x 44
    W = [[0] * rank for _ in range(p * n)]   # 16 x 44

    col = 0

    def add_strassen_2x2x2(i_off, j_off, k_off):
        """Add 7 rank-1 terms for a 2x2x2 Strassen block.
        U row index: i*m + j  (i in 0..1, j in 0..1 relative to offsets)
        V row index: j*p + k
        W row index: k*n + i
        """
        nonlocal col
        # Mult 1: (a11+a22)(b11+b22) -> c11, c22
        U[i_off * m + j_off][col] += 1
        U[(i_off + 1) * m + (j_off + 1)][col] += 1
        V[j_off * p + k_off][col] += 1
        V[(j_off + 1) * p + (k_off + 1)][col] += 1
        W[k_off * n + i_off][col] += 1
        W[(k_off + 1) * n + (i_off + 1)][col] += 1
        col += 1

        # Mult 2: (a21+a22)(b11) -> c21, c22
        U[(i_off + 1) * m + j_off][col] += 1
        U[(i_off + 1) * m + (j_off + 1)][col] += 1
        V[j_off * p + k_off][col] += 1
        W[k_off * n + (i_off + 1)][col] += 1
        W[(k_off + 1) * n + (i_off + 1)][col] += -1
        col += 1

        # Mult 3: (a11)(b21-b22) -> c12, c22
        U[i_off * m + j_off][col] += 1
        V[j_off * p + (k_off + 1)][col] += 1
        V[(j_off + 1) * p + (k_off + 1)][col] += -1
        W[(k_off + 1) * n + i_off][col] += 1
        W[(k_off + 1) * n + (i_off + 1)][col] += 1
        col += 1

        # Mult 4: (a22)(b12-b11) -> c11, c12
        U[(i_off + 1) * m + (j_off + 1)][col] += 1
        V[(j_off + 1) * p + k_off][col] += 1
        V[j_off * p + k_off][col] += -1
        W[k_off * n + i_off][col] += 1
        W[k_off * n + (i_off + 1)][col] += 1
        col += 1

        # Mult 5: (a11+a12)(b22) -> c11, c21
        U[i_off * m + j_off][col] += 1
        U[i_off * m + (j_off + 1)][col] += 1
        V[(j_off + 1) * p + (k_off + 1)][col] += 1
        W[k_off * n + i_off][col] += -1
        W[(k_off + 1) * n + i_off][col] += 1
        col += 1

        # Mult 6: (a21-a11)(b11+b12) -> c22
        U[(i_off + 1) * m + j_off][col] += 1
        U[i_off * m + j_off][col] += -1
        V[j_off * p + k_off][col] += 1
        V[j_off * p + (k_off + 1)][col] += 1
        W[(k_off + 1) * n + (i_off + 1)][col] += 1
        col += 1

        # Mult 7: (a12-a22)(b21+b22) -> c11
        U[i_off * m + (j_off + 1)][col] += 1
        U[(i_off + 1) * m + (j_off + 1)][col] += -1
        V[(j_off + 1) * p + k_off][col] += 1
        V[(j_off + 1) * p + (k_off + 1)][col] += 1
        W[k_off * n + i_off][col] += 1
        col += 1

    def add_schoolbook_2x1x4(i_off, j_off, k_off):
        """Add 8 rank-1 terms for a 2x1x4 schoolbook block."""
        nonlocal col
        for ii in range(2):
            for kk in range(4):
                U[(i_off + ii) * m + j_off][col] = 1
                V[j_off * p + (k_off + kk)][col] = 1
                W[(k_off + kk) * n + (i_off + ii)][col] = 1
                col += 1

    # --- First p=4 block (k=0,1,2,3) ---
    # m=2 sub-blocks (j=0,1): two 2x2x2 Strassen
    add_strassen_2x2x2(0, 0, 0)  # k=0,1
    add_strassen_2x2x2(0, 0, 2)  # k=2,3
    # m=1 sub-block (j=2): 2x1x4 schoolbook
    add_schoolbook_2x1x4(0, 2, 0)  # k=0,1,2,3

    # --- Second p=4 block (k=4,5,6,7) ---
    # m=2 sub-blocks (j=0,1): two 2x2x2 Strassen
    add_strassen_2x2x2(0, 0, 4)  # k=4,5
    add_strassen_2x2x2(0, 0, 6)  # k=6,7
    # m=1 sub-block (j=2): 2x1x4 schoolbook
    add_schoolbook_2x1x4(0, 2, 4)  # k=4,5,6,7

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
