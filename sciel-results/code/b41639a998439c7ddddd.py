"""Program for matrix multiplication tensor decomposition: 2x4 * 4x5."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def _construct_rank_36():
    """
    Construct rank-36 decomposition for 2x4 x 4x5 matrix multiplication.
    Splits the 4 columns of A (rows of B) into two blocks of 2.
    Each 2x2x5 block uses:
      - Strassen (rank 7) for cols 0-1 of the 5-column sub-problem
      - Strassen (rank 7) for cols 2-3
      - Schoolbook (rank 4) for col 4
    Total per block: 7 + 7 + 4 = 18. Two blocks: 36.
    """
    n, m, p = 2, 4, 5
    rank = 36

    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36

    def add_strassen_222(i1, i2, j1, j2, k1, k2, col_offset):
        """Add a Strassen 2x2x2 block using indices from the full matrices."""
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2

        b11 = j1 * p + k1
        b12 = j1 * p + k2
        b21 = j2 * p + k1
        b22 = j2 * p + k2

        c11 = k1 * n + i1
        c12 = k2 * n + i1
        c21 = k1 * n + i2
        c22 = k2 * n + i2

        # M1 = (A11 + A22)(B11 + B22)
        col = col_offset + 0
        U[a11][col] = 1; U[a22][col] = 1
        V[b11][col] = 1; V[b22][col] = 1
        W[c11][col] = 1; W[c22][col] = 1

        # M2 = (A21 + A22)B11
        col = col_offset + 1
        U[a21][col] = 1; U[a22][col] = 1
        V[b11][col] = 1
        W[c21][col] = 1; W[c22][col] = -1

        # M3 = A11(B12 - B22)
        col = col_offset + 2
        U[a11][col] = 1
        V[b12][col] = 1; V[b22][col] = -1
        W[c12][col] = 1; W[c22][col] = 1

        # M4 = A22(B21 - B11)
        col = col_offset + 3
        U[a22][col] = 1
        V[b21][col] = 1; V[b11][col] = -1
        W[c11][col] = 1; W[c21][col] = 1

        # M5 = (A11 + A12)B22
        col = col_offset + 4
        U[a11][col] = 1; U[a12][col] = 1
        V[b22][col] = 1
        W[c11][col] = -1; W[c12][col] = 1

        # M6 = (A21 - A11)(B11 + B12)
        col = col_offset + 5
        U[a21][col] = 1; U[a11][col] = -1
        V[b11][col] = 1; V[b12][col] = 1
        W[c22][col] = 1

        # M7 = (A12 - A22)(B21 + B22)
        col = col_offset + 6
        U[a12][col] = 1; U[a22][col] = -1
        V[b21][col] = 1; V[b22][col] = 1
        W[c11][col] = 1

    def add_schoolbook_221(i1, i2, j1, j2, k1, col_offset):
        """Add a schoolbook 2x2x1 block (4 multiplications)."""
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2

        b11 = j1 * p + k1
        b21 = j2 * p + k1

        c11 = k1 * n + i1
        c21 = k1 * n + i2

        col = col_offset + 0
        U[a11][col] = 1; V[b11][col] = 1; W[c11][col] = 1

        col = col_offset + 1
        U[a12][col] = 1; V[b21][col] = 1; W[c11][col] = 1

        col = col_offset + 2
        U[a21][col] = 1; V[b11][col] = 1; W[c21][col] = 1

        col = col_offset + 3
        U[a22][col] = 1; V[b21][col] = 1; W[c21][col] = 1

    col = 0
    # Block 1: columns j=0,1 of A (rows 0,1 of B)
    add_strassen_222(0, 1, 0, 1, 0, 1, col); col += 7
    add_strassen_222(0, 1, 0, 1, 2, 3, col); col += 7
    add_schoolbook_221(0, 1, 0, 1, 4, col); col += 4

    # Block 2: columns j=2,3 of A (rows 2,3 of B)
    add_strassen_222(0, 1, 2, 3, 0, 1, col); col += 7
    add_strassen_222(0, 1, 2, 3, 2, 3, col); col += 7
    add_schoolbook_221(0, 1, 2, 3, 4, col); col += 4

    return U, V, W

def construct():
    """
    Return factor matrices U, V, W for the rank-36 decomposition of
    2x4 * 4x5 matrix multiplication.
    """
    # Deadline guard
    t0 = time.time()
    deadline = 140  # seconds (10s safety margin under 150s limit)

    try:
        U, V, W = _construct_rank_36()
    except Exception:
        # Fallback: pure schoolbook rank-40 (should not be reached)
        n, m, p = 2, 4, 5
        rank_fb = n * m * p  # 40
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U_fb[a][idx] = 1
                    V_fb[b][idx] = 1
                    W_fb[c][idx] = 1
                    idx += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    elapsed = time.time() - t0
    if elapsed > deadline:
        # Should never happen for this fixed construction, but guard anyway
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
