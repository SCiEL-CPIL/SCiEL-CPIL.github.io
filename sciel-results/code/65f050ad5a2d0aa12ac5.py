"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    Uses Strassen's rank-7 decomposition for 2x2x2 blocks applied to overlapping
    2x2 column blocks of B, plus schoolbook for the remaining row.
    """
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin

    def elapsed():
        return time.time() - start_time

    def validate(U, V, W):
        """Check if decomposition exactly computes matrix multiplication."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def schoolbook():
        rank = n * m * p  # 72
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, rank

    # Start with schoolbook as fallback
    best_U, best_V, best_W, best_rank = schoolbook()

    # Strassen's 2x2x2 decomposition (rank 7)
    # For multiplying (2x2) * (2x2) -> (2x2)
    # C = A * B where A, B, C are 2x2
    # Strassen's 7 products:
    # m1 = (a00 + a11) * (b00 + b11)
    # m2 = (a10 + a11) * b00
    # m3 = a00 * (b01 - b11)
    # m4 = a11 * (b10 - b00)
    # m5 = (a00 + a01) * b11
    # m6 = (a10 - a00) * (b00 + b01)
    # m7 = (a01 - a11) * (b10 + b11)
    #
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6

    def strassen_2x2x2():
        """
        Return U, V, W for 2x2x2 Strassen decomposition.
        U: 4x7 (flattened 2x2 A)
        V: 4x7 (flattened 2x2 B)
        W: 4x7 (flattened 2x2 C)
        Index: U[i*2+j], V[j*2+k], W[k*2+i]
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))

        # Term 0: m1 = (a00+a11)(b00+b11)
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[3,0] = 1
        W[0,0] = 1; W[3,0] = 1  # C00 += m1, C11 += m1

        # Term 1: m2 = (a10+a11)b00
        U[2,1] = 1; U[3,1] = 1
        V[0,1] = 1
        W[2,1] = 1  # C10 += m2

        # Term 2: m3 = a00(b01-b11)
        U[0,2] = 1
        V[1,2] = 1; V[3,2] = -1
        W[1,2] = 1  # C01 += m3

        # Term 3: m4 = a11(b10-b00)
        U[3,3] = 1
        V[2,3] = 1; V[0,3] = -1
        W[0,3] = 1; W[2,3] = 1  # C00 += m4, C10 += m4

        # Term 4: m5 = (a00+a01)b11
        U[0,4] = 1; U[1,4] = 1
        V[3,4] = 1
        W[0,4] = -1; W[1,4] = 1  # C00 -= m5, C01 += m5

        # Term 5: m6 = (a10-a00)(b00+b01)
        U[2,5] = 1; U[0,5] = -1
        V[0,5] = 1; V[1,5] = 1
        W[3,5] = 1  # C11 += m6

        # Term 6: m7 = (a01-a11)(b10+b11)
        U[1,6] = 1; U[3,6] = -1
        V[2,6] = 1; V[3,6] = 1
        W[0,6] = 1; W[3,6] = -1  # C00 += m7, C11 -= m7

        # Verify
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    val = np.sum(U[i*2+j] * V[j*2+k] * W[k*2+i])
                    if abs(val - 1.0) > 1e-9:
                        return None

        return U.tolist(), V.tolist(), W.tolist()

    # Strategy: Use Strassen on 2x2 blocks of B for A rows 0,1
    # B is 3x8. We partition B's columns into groups.
    # For A rows 0,1 (2x3) times B (3x8):
    #   We can use 2x2 Strassen blocks on B's columns.
    #   B columns: 0,1,2,3,4,5,6,7
    #   2x2 blocks: (0,1), (2,3), (4,5), (6,7) -> 4 blocks, each rank 7 = 28
    #   But B has 3 rows, and Strassen needs 2x2 B blocks.
    #   So we use B rows 0,1 for the 2x2 block, and handle B row 2 separately.
    #
    # Alternative: Use overlapping 2x2 blocks.
    # B columns 0-7, use blocks starting at even positions: (0,1), (2,3), (4,5), (6,7)
    # Each block uses B rows 0,1 and 2 columns.
    # For each such block: A[0:2, 0:2] (2x2) * B[0:2, cols] (2x2) -> C[0:2, cols] (2x2)
    # But A has 3 columns (j=0,1,2), and B has 3 rows.
    #
    # Let me reconsider. The tensor is T[i,j,k] where i in {0,1,2} (C row),
    # j in {0,1,2} (A col / B row), k in {0,...,7} (B col / C col).
    #
    # For i in {0,1}, we want C[i,k] = sum_j A[i,j] * B[j,k]
    # For i=2, C[2,k] = sum_j A[2,j] * B[j,k]
    #
    # For the i in {0,1} part:
    #   C[0:2, k] = A[0:2, :] * B[:, k]
    #   This is a 2x3 * 3x1 = 2x1 multiplication for each k.
    #   Or equivalently, for a group of 2 columns k1,k2:
    #   C[0:2, k1:k2+1] = A[0:2, :] * B[:, k1:k2+1]  (2x3 * 3x2 = 2x2)
    #
    # For a 2x2 output block C[0:2, k1:k2+1]:
    #   C[0,k1] = A[0,0]*B[0,k1] + A[0,1]*B[1,k1] + A[0,2]*B[2,k1]
    #   C[0,k2] = A[0,0]*B[0,k2] + A[0,1]*B[1,k2] + A[0,2]*B[2,k2]
    #   C[1,k1] = A[1,0]*B[0,k1] + A[1,1]*B[1,k1] + A[1,2]*B[2,k1]
    #   C[1,k2] = A[1,0]*B[0,k2] + A[1,1]*B[1,k2] + A[1,2]*B[2,k2]
    #
    # This is a 2x3 * 3x2 = 2x2 multiplication. We can decompose this using
    # Strassen-like techniques.
    #
    # One approach: split A[0:2, :] into A[0:2, 0:2] (2x2) and A[0:2, 2] (2x1)
    # and B[:, k1:k2+1] into B[0:2, k1:k2+1] (2x2) and B[2, k1:k2+1] (1x2)
    #
    # C = A11 * B11 + A12 * B21 + A21 * B12 + A22 * B22
    # where A11 = A[0:2,0:2], A12 = A[0:2,2], A21 = A[2,0:2], A22 = A[2,2]
    # B11 = B[0:2,k1:k2+1], B12 = B[0:2,k2], B21 = B[2,k1:k2+1], B22 = B[2,k2]
    #
    # This is getting complex. Let me use a simpler but effective approach.

    # Approach: Use Strassen on 2x2x2 blocks where possible
    # For A[0:2, 0:2] * B[0:2, k1:k2+1] -> C[0:2, k1:k2+1] (2x2 * 2x2 = 2x2)
    # This uses Strassen rank 7.
    # Then add correction terms for the j=2 (A col 2, B row 2) contributions.
    #
    # For each 2-column block of B (k1, k2):
    #   Base: Strassen on A[0:2,0:2] * B[0:2, k1:k2+1] -> rank 7
    #   Correction: A[0:2,2] * B[2, k1:k2+1] -> 2x1 * 1x2 = 2x2, rank 4 (schoolbook)
    #   Total per block: 7 + 4 = 11
    #   4 blocks: 44
    #   Plus row 2 of A: A[2,:] * B[:, :] -> 1x3 * 3x8 = 1x8, rank 24 (schoolbook)
    #   Total: 44 + 24 = 68

    # Let's implement this:
    # For 4 column blocks of B: (0,1), (2,3), (4,5), (6,7)
    # For each block, use Strassen on A[0:2,0:2] * B[0:2, block] + schoolbook for A[0:2,2]*B[2,block]
    # Then schoolbook for A[2,:]*B[:, :]

    def build_strassen_based():
        """Build decomposition using Strassen on 2x2 blocks + schoolbook corrections."""
        U = np.zeros((n*m, 68))  # 9 x 68
        V = np.zeros((m*p, 68))  # 24 x 68
        W = np.zeros((p*n, 68))  # 24 x 68

        col = 0

        # Strassen 2x2x2 terms
        strassen = strassen_2x2x2()
        if strassen is None:
            return None
        S_U, S_V, S_W = strassen

        # 4 column blocks of B: (0,1), (2,3), (4,5), (6,7)
        col_blocks = [(0,1), (2,3), (4,5), (6,7)]

        for (k1, k2) in col_blocks:
            # Strassen part: A[0:2, 0:2] * B[0:2, k1:k2+1] -> C[0:2, k1:k2+1]
            # A indices: i in {0,1}, j in {0,1} -> U[i*3+j]
            # B indices: j in {0,1}, k in {k1,k2} -> V[j*8+k]
            # C indices: i in {0,1}, k in {k1,k2} -> W[k*3+i]

            for s in range(7):
                # U: A[0:2, 0:2]
                for i in range(2):
                    for j in range(2):
                        U[i*3+j, col] += S_U[i*2+j][s]

                # V: B[0:2, k1:k2+1]
                for j in range(2):
                    for kk in range(2):
                        k_idx = k1 + kk
                        V[j*8+k_idx, col] += S_V[j*2+kk][s]

                # W: C[0:2, k1:k2+1]
                for i in range(2):
                    for kk in range(2):
                        k_idx = k1 + kk
                        W[k_idx*3+i, col] += S_W[kk*2+i][s]

                col += 1

            # Schoolbook correction: A[0:2, 2] * B[2, k1:k2+1] -> C[0:2, k1:k2+1]
            for i in range(2):
                for kk in range(2):
                    k_idx = k1 + kk
                    U[i*3+2, col] = 1
                    V[2*8+k_idx, col] = 1
                    W[k_idx*3+i, col] = 1
                    col += 1

        # Schoolbook for row 2 of A: A[2, :] * B[:, :] -> C[2, :]
        for j in range(3):
            for k in range(8):
                U[2*3+j, col] = 1
                V[j*8+k, col] = 1
                W[k*3+2, col] = 1
                col += 1

        # Verify
        if not validate(U.tolist(), V.tolist(), W.tolist()):
            return None

        return U.tolist(), V.tolist(), W.tolist(), 68

    result = build_strassen_based()
    if result is not None and elapsed() < deadline - 5:
        U_new, V_new, W_new, rank_new = result
        if rank_new < best_rank:
            best_U, best_V, best_W, best_rank = U_new, V_new, W_new, rank_new

    # Try overlapping Strassen blocks for potentially lower rank
    # Use 2x2 blocks at positions (0,1), (1,2), (2,3), ..., (6,7) -> 7 blocks
    # But this overcounts. Let me try a different partition.

    # Alternative: Use 2x2x4 block decomposition
    # A[0:2, 0:2] * B[0:2, 0:4] -> C[0:2, 0:4] using a 2x2x4 decomposition
    # Then A[0:2, 0:2] * B[0:2, 4:8] -> C[0:2, 4:8]
    # Plus corrections for j=2.
    # This is more complex; skip for now.

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
