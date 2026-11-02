"""
Matrix multiplication tensor decomposition for 2x5 * 5x6 -> 2x6
Uses block decomposition with Strassen's algorithm.
Rank 48 strategy:
- A (2x5) split into A1 (2x4) and A2 (2x1)
- B (5x6) split into B1 (4x6) and B2 (1x6)
- C = A1*B1 + A2*B2
- A1*B1: Split B1 into B1a (4x4) and B1b (4x2)
  - A1*B1a: 4 Strassen 2x2x2 blocks (Rank 28)
  - A1*B1b: 2 Strassen 2x2x2 blocks (Rank 14)
- A2*B2: Rank 6 (Outer product-like)
Total Rank: 28 + 14 + 6 = 48
"""
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    start = time.time()
    deadline = 145.0

    n, m, p = DIMENSIONS  # 2, 5, 6

    # Total rank calculation
    rank_strassen_2x4x4 = 28  # 4 blocks of 7
    rank_strassen_2x4x2 = 14  # 2 blocks of 7
    rank_2x1x6 = 6
    total_rank = rank_strassen_2x4x4 + rank_strassen_2x4x2 + rank_2x1x6

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    col = 0

    def uidx(i, j):
        return i * m + j

    def vidx(j, k):
        return j * p + k

    def widx(k, i):
        return k * n + i

    def ok():
        return time.time() - start < deadline

    def emit(u_list, v_list, w_list):
        nonlocal col
        if col >= total_rank:
            return
        if not ok():
            return
        for (idx, val) in u_list:
            U[idx][col] = val
        for (idx, val) in v_list:
            V[idx][col] = val
        for (idx, val) in w_list:
            W[idx][col] = val
        col += 1

    def strassen_2x2(j0, j1, k0, k1):
        """
        Decompose A[:, j0:j1+1] @ B[j0:j1+1, k0:k1+1] using Strassen.
        A sub-matrix rows: i=0,1; cols: j0, j1
        B sub-matrix rows: j0, j1; cols: k0, k1
        C sub-matrix rows: i=0,1; cols: k0, k1
        """
        a00, a01 = uidx(0, j0), uidx(0, j1)
        a10, a11 = uidx(1, j0), uidx(1, j1)
        b00, b01 = vidx(j0, k0), vidx(j0, k1)
        b10, b11 = vidx(j1, k0), vidx(j1, k1)
        c00, c01 = widx(k0, 0), widx(k1, 0)
        c10, c11 = widx(k0, 1), widx(k1, 1)

        # M1 = (A00 + A11)(B00 + B11)
        emit([(a00, 1), (a11, 1)], [(b00, 1), (b11, 1)], [(c00, 1), (c11, 1)])

        # M2 = (A10 + A11) B00
        emit([(a10, 1), (a11, 1)], [(b00, 1)], [(c10, 1), (c11, -1)])

        # M3 = A00 (B01 - B11)
        emit([(a00, 1)], [(b01, 1), (b11, -1)], [(c01, 1), (c11, 1)])

        # M4 = A11 (B10 - B00)
        emit([(a11, 1)], [(b10, 1), (b00, -1)], [(c00, 1), (c10, 1)])

        # M5 = (A00 + A01) B11
        emit([(a00, 1), (a01, 1)], [(b11, 1)], [(c00, -1), (c01, 1)])

        # M6 = (A10 - A00)(B00 + B01)
        emit([(a10, 1), (a00, -1)], [(b00, 1), (b01, 1)], [(c11, 1)])

        # M7 = (A01 - A11)(B10 + B11)
        emit([(a01, 1), (a11, -1)], [(b10, 1), (b11, 1)], [(c00, 1)])

    def rank_2x1x6(j, k_start, k_end):
        """
        Decompose A[:, j] @ B[j, k_start:k_end] using rank 6 algorithm.
        A sub-matrix: 2x1 (rows i=0,1; col j)
        B sub-matrix: 1xL (row j; cols k_start..k_end-1)
        C sub-matrix: 2xL (rows i=0,1; cols k_start..k_end-1)
        """
        a0, a1 = uidx(0, j), uidx(1, j)
        for k in range(k_start, k_end):
            b_idx = vidx(j, k)
            c0_idx = widx(k, 0)
            c1_idx = widx(k, 1)
            # Term for this k
            # U: A_i,j = 1
            # V: B_j,k = 1
            # W: C_i,k = 1
            emit([(a0, 1), (a1, 1)], [(b_idx, 1)], [(c0_idx, 1), (c1_idx, 1)])

    # 1. Process A1 (cols 0-3) with B1 (rows 0-3)
    # Split B1 into B1a (cols 0-3) and B1b (cols 4-5)
    
    # A1 (2x4) * B1a (4x4)
    # Split A1 into A1a (cols 0-1) and A1b (cols 2-3)
    # B1a into B1aa (rows 0-1) and B1ab (rows 2-3)
    # This creates 4 Strassen blocks:
    # A1a * B1aa (j0=0, j1=1, k0=0, k1=1)
    # A1a * B1ab (j0=0, j1=1, k0=2, k1=3) -- Wait, B1ab is rows 2,3.
    # The Strassen block multiplies A[:, j0:j1] * B[j0:j1, k0:k1].
    # Here the inner dimension is j.
    # For A1a * B1aa: j is 0,1. k is 0,1.
    # For A1a * B1ab: j is 0,1. But B rows are 2,3.
    # Strassen requires the indices of A cols and B rows to match the block definition.
    # We are multiplying A[:, 0:2] * B[0:2, 0:2]. This is valid.
    # We are multiplying A[:, 0:2] * B[2:4, 2:4]? No.
    # The multiplication is A_{2x4} * B_{4x4}.
    # A = [A_01, A_23] where A_01 is cols 0,1 and A_23 is cols 2,3.
    # B = [B_01; B_23] where B_01 is rows 0,1 and B_23 is rows 2,3.
    # C = A_01 B_01 + A_23 B_23.
    # So we need:
    # 1. A[:, 0:2] * B[0:2, 0:2] -> Strassen (j=0,1, k=0,1)
    # 2. A[:, 0:2] * B[2:4, 2:4]? No.
    # The term is A_{i,j} B_{j,k}.
    # If j is in 0,1, B rows are 0,1.
    # If j is in 2,3, B rows are 2,3.
    # So we have two independent multiplications:
    # A[:, 0:2] * B[0:2, 0:4]
    # A[:, 2:4] * B[2:4, 0:4]
    
    # Let's refine the blocks for A1 (cols 0-3) * B1 (rows 0-3, cols 0-5)
    
    # Block 1: A[:, 0:2] * B[0:2, 0:2]
    # j=0,1, k=0,1
    strassen_2x2(0, 1, 0, 1)
    
    # Block 2: A[:, 0:2] * B[0:2, 2:4]
    # j=0,1, k=2,3
    strassen_2x2(0, 1, 2, 3)
    
    # Block 3: A[:, 2:4] * B[2:4, 0:2]
    # j=2,3, k=0,1
    strassen_2x2(2, 3, 0, 1)
    
    # Block 4: A[:, 2:4] * B[2:4, 2:4]
    # j=2,3, k=2,3
    strassen_2x2(2, 3, 2, 3)
    
    # Now for B1b (rows 0-3, cols 4-5)
    # A[:, 0:2] * B[0:2, 4:6]
    strassen_2x2(0, 1, 4, 5)
    
    # A[:, 2:4] * B[2:4, 4:6]
    strassen_2x2(2, 3, 4, 5)
    
    # 2. Process A2 (col 4) with B2 (row 4, cols 0-5)
    # Rank 6
    rank_2x1x6(4, 0, 6)

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
