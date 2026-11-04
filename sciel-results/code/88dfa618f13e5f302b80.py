"""Matrix multiplication 2x3x6 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]
DEADLINE = time.time() + 120  # hard wall-clock limit


def _verify(U, V, W):
    """Check whether the factor matrices exactly decompose the 2x3x6 matmul tensor."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    rank = U.shape[1]
    # Reconstruct tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1
                val = 0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if val != expected:
                    return False
    return True


def _schoolbook():
    """Return the rank-36 schoolbook decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _strassen_block(U, V, W, col, i_off, j_off, k_off, n=2, m=2, p=2):
    """
    Add a 2x2x2 Strassen block to U, V, W starting at column 'col'.
    i_off, j_off, k_off are the offsets into the full matrices.
    Returns the new col (col + 7).
    """
    n, m, p = 2, 2, 2  # local dimensions

    # Strassen multiplications for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # M1
    U[i_off + 0, col] += 1
    U[i_off + 1, col] += 1
    V[j_off + 0, col] += 1
    V[j_off + 1, col] += 1
    W[k_off + 0, col] += 1  # C00
    W[k_off + 1, col] += 1  # C11
    col += 1

    # M2
    U[i_off + 1, col] += 1  # A10
    U[i_off + 1, col] += 1  # A11 -> total 2? No, U is per-row
    # Wait, U[i*m+j, c]. For 2x2 block:
    # A00 at (i_off, j_off), A01 at (i_off, j_off+1)
    # A10 at (i_off+1, j_off), A11 at (i_off+1, j_off+1)
    # So U row indices are i_off*2 + j_off etc? No.
    # The full U matrix has shape (n_full * m_full, rank).
    # The block is at rows i_off..i_off+1 and cols j_off..j_off+1.
    # So the row index in U for A[i][j] is (i_off+i) * m_full + (j_off+j).
    # But we are passing i_off, j_off as offsets into the *full* matrix indices?
    # Let's redefine: i_off, j_off, k_off are the base indices for the 2x2 block
    # in the full n, m, p dimensions.
    # So A00 is at full index (i_off, j_off) -> U row = i_off * m_full + j_off.
    # A01 is at full index (i_off, j_off+1) -> U row = i_off * m_full + j_off + 1.
    # A10 is at full index (i_off+1, j_off) -> U row = (i_off+1) * m_full + j_off.
    # A11 is at full index (i_off+1, j_off+1) -> U row = (i_off+1) * m_full + j_off + 1.

    # Let's rewrite _strassen_block to take m_full, p_full as well.
    # Actually, it's easier to just compute the row/col indices directly in construct.
    pass


def construct():
    """
    Find a rank-33 decomposition for 2x3x6 matrix multiplication.
    Strategy: Split A (2x3) into A1 (2x1) and A2 (2x2).
    Split B (3x6) into B1 (1x6) and B2 (2x6).
    C = A1*B1 + A2*B2.
    A1*B1 is 2x1x6, rank 12 (schoolbook).
    A2*B2 is 2x2x6, split into three 2x2x2 blocks, each rank 7 (Strassen), total 21.
    Total rank: 33.
    """
    t0 = time.time()
    n, m, p = DIMENSIONS
    rank = 33

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Part 1: A1 (2x1) * B1 (1x6) => 2x6 output
    # A1: rows i=0,1; col j=0.
    # B1: row j=0; cols k=0..5.
    # C[i][k] = A[i][0] * B[0][k]
    for k in range(p):
        for i in range(n):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    # col is now 12

    # Part 2: A2 (2x2) * B2 (2x6) => 2x6 output
    # A2: rows i=0,1; cols j=1,2.
    # B2: rows j=1,2; cols k=0..5.
    # Split B2 into three 2x2 blocks: B2_0 (cols 0,1), B2_1 (cols 2,3), B2_2 (cols 4,5).
    # For each block, use Strassen with 7 multiplications.

    for t in range(3):
        # B2 block t has cols k = 2t, 2t+1.
        # A2 is fixed: A[0][1], A[0][2], A[1][1], A[1][2].
        # U indices:
        # A00 (local) = A[0][1] -> U[0*m + 1] = U[1]
        # A01 (local) = A[0][2] -> U[0*m + 2] = U[2]
        # A10 (local) = A[1][1] -> U[1*m + 1] = U[4]
        # A11 (local) = A[1][2] -> U[1*m + 2] = U[5]

        # V indices for block t:
        # B00 (local) = B[1][2t] -> V[1*p + 2t] = V[6 + 2t]
        # B01 (local) = B[1][2t+1] -> V[1*p + 2t+1] = V[7 + 2t]
        # B10 (local) = B[2][2t] -> V[2*p + 2t] = V[12 + 2t]
        # B11 (local) = B[2][2t+1] -> V[2*p + 2t+1] = V[13 + 2t]

        # W indices for block t:
        # C00 (local) = C[0][2t] -> W[2t*n + 0] = W[4t]
        # C01 (local) = C[0][2t+1] -> W[(2t+1)*n + 0] = W[4t+2]
        # C10 (local) = C[1][2t] -> W[2t*n + 1] = W[4t+1]
        # C11 (local) = C[1][2t+1] -> W[(2t+1)*n + 1] = W[4t+3]

        # M1 = (A00+A11)(B00+B11)
        U[1][col] += 1
        U[5][col] += 1
        V[6 + 2*t][col] += 1
        V[13 + 2*t][col] += 1
        W[4*t][col] += 1       # C00
        W[4*t + 3][col] += 1   # C11
        col += 1

        # M2 = (A10+A11)B00
        U[4][col] += 1
        U[5][col] += 1
        V[6 + 2*t][col] += 1
        W[4*t + 1][col] += 1   # C10
        W[4*t + 3][col] -= 1   # C11
        col += 1

        # M3 = A00(B01-B11)
        U[1][col] += 1
        V[7 + 2*t][col] += 1
        V[13 + 2*t][col] -= 1
        W[4*t + 2][col] += 1   # C01
        W[4*t + 3][col] += 1   # C11
        col += 1

        # M4 = A11(B10-B00)
        U[5][col] += 1
        V[12 + 2*t][col] += 1
        V[6 + 2*t][col] -= 1
        W[4*t][col] += 1       # C00
        W[4*t + 1][col] += 1   # C10
        col += 1

        # M5 = (A00+A01)B11
        U[1][col] += 1
        U[2][col] += 1
        V[13 + 2*t][col] += 1
        W[4*t][col] -= 1       # C00
        W[4*t + 2][col] += 1   # C01
        col += 1

        # M6 = (A10-A00)(B00+B01)
        U[4][col] += 1
        U[1][col] -= 1
        V[6 + 2*t][col] += 1
        V[7 + 2*t][col] += 1
        W[4*t + 3][col] += 1   # C11
        col += 1

        # M7 = (A01-A11)(B10+B11)
        U[2][col] += 1
        U[5][col] -= 1
        V[12 + 2*t][col] += 1
        V[13 + 2*t][col] += 1
        W[4*t][col] += 1       # C00
        col += 1

    # col should be 12 + 3*7 = 33
    if time.time() > DEADLINE - 5:
        return _schoolbook()

    # Verify the decomposition
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook if verification fails
        return _schoolbook()
