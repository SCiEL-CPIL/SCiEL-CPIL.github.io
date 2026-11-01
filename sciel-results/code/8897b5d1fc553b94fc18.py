"""Matrix multiplication 2x3 * 3x8: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 8
    deadline = time.time() + 110  # safety margin under 120s

    def schoolbook():
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

    def verify(U, V, W, trials=5):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        for _ in range(trials):
            A = np.random.randn(2, 3)
            B = np.random.randn(3, 8)
            C_expected = A @ B
            C_computed = np.zeros((2, 8))
            for c in range(U_np.shape[1]):
                U_mat = U_np[:, c].reshape(2, 3)
                V_mat = V_np[:, c].reshape(3, 8)
                W_vec = W_np[:, c]
                prod = U_mat @ V_mat
                for i in range(2):
                    for k in range(8):
                        C_computed[i, k] += W_vec[k * 2 + i] * prod[i, k]
            if not np.allclose(C_computed, C_expected, atol=1e-9):
                return False
        return True

    # Strategy: Use rank-7 algorithm for 2x4 matrix multiplication (Strassen)
    # combined with the 3x2 block structure.
    # 
    # A is 2x3, B is 3x8.
    # Split B into 4 blocks of 3x2: B = [B0, B1, B2, B3]
    # C = A * B = [A*B0, A*B1, A*B2, A*B3]
    # 
    # Each A*Bj is 2x2. We can view this as 2x4 multiplication where:
    # - A is 2x2 (but A is 2x3, so we need to handle the extra column)
    # 
    # Actually, let's think differently. The tensor is 2x3x8.
    # We can decompose it as:
    # - Split the 3 dimension into 2+1
    # - Part 1: 2x2x8 (A[:,0:2] * B[0:2,:])
    # - Part 2: 2x1x8 (A[:,2] * B[2,:])
    #
    # For Part 1: 2x2x8 = (2x2) * (2x8) -> 2x8
    # View as 2x4 of 2x2 blocks: rank 7 (Strassen) * 4 (standard 2x2) = 28
    # But we can do better by sharing computations.
    #
    # Alternative: Use the fact that 2x3x8 can be seen as 2x4x4 with padding.
    # Pad A to 2x4 (add zero column), pad B to 4x8 (add zero row).
    # Then it's 2x4x4 multiplication, which has known rank.
    #
    # Actually, the simplest approach: use rank-7 Strassen for 2x2x2,
    # and combine 4 independent 2x2x2 multiplications.
    # Rank: 4 * 7 = 28.
    #
    # But we can do better by sharing the A matrix across blocks.
    # C[:,0:2] = A * B[0:2,0:2]
    # C[:,2:4] = A * B[0:2,2:4]
    # C[:,4:6] = A * B[0:2,4:6]
    # C[:,6:8] = A * B[0:2,6:8]
    # Plus:
    # C[:,0:8] += A[:,2] * B[2,:]
    #
    # For the 4 blocks of A[:,0:2] * B[0:2, block], we can use a shared
    # Strassen-like algorithm. The rank of 2x2x8 is known to be 14
    # (two independent Strassen 2x2x2).
    #
    # Total rank: 14 + 8 (for the 2x1x8 part, which is 8 rank-1 terms) = 22.
    # But 2x1x8 can be done in rank 8 (each output is a separate product).
    #
    # Actually, let me just implement rank 28 and verify.

    # Implementation: 4 independent Strassen 2x2x2 for A[:,0:2] * B[0:2, block]
    # Plus 8 rank-1 terms for A[:,2] * B[2,:]
    
    rank = 36  # Start with 36, try to reduce
    U = [[0.0] * rank for _ in range(6)]
    V = [[0.0] * rank for _ in range(24)]
    W = [[0.0] * rank for _ in range(16)]

    # Strassen 2x2x2: A(2x2) * B(2x2) = C(2x2)
    # M1 = (A00+A11)*(B00+B10)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B10-B00)
    # M4 = A11*(B01+B11)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B01-B11)
    # M7 = (A01-A11)*(B00+B10)
    # C00 = M1 - M3
    # C01 = M4 + M5
    # C10 = M2 + M3
    # C11 = M1 - M2 + M6 - M7

    def add_strassen(base, a_col0, a_col1, b_row0, b_row1, b_col_offset):
        """Add Strassen 2x2x2 decomposition.
        A uses columns a_col0, a_col1 (rows 0,1)
        B uses rows b_row0, b_row1 and columns b_col_offset, b_col_offset+1
        """
        # M1 = (A00+A11)*(B00+B10)
        U[0 * 3 + a_col0][base] += 1
        U[1 * 3 + a_col1][base] += 1
        V[b_row0 * 8 + b_col_offset][base] += 1
        V[b_row1 * 8 + b_col_offset][base] += 1
        W[b_col_offset * 2 + 0][base] += 1
        W[(b_col_offset + 1) * 2 + 1][base] += 1

        # M2 = (A10+A11)*B00
        U[1 * 3 + a_col0][base + 1] += 1
        U[1 * 3 + a_col1][base + 1] += 1
        V[b_row0 * 8 + b_col_offset][base + 1] += 1
        W[b_col_offset * 2 + 1][base + 1] += 1

        # M3 = A00*(B10-B00)
        U[0 * 3 + a_col0][base + 2] += 1
        V[b_row1 * 8 + b_col_offset][base + 2] += 1
        V[b_row0 * 8 + b_col_offset][base + 2] += -1
        W[b_col_offset * 2 + 0][base + 2] += -1
        W[b_col_offset * 2 + 1][base + 2] += 1

        # M4 = A11*(B01+B11)
        U[1 * 3 + a_col1][base + 3] += 1
        V[b_row0 * 8 + b_col_offset + 1][base + 3] += 1
        V[b_row1 * 8 + b_col_offset + 1][base + 3] += 1
        W[b_col_offset * 2 + 0][base + 3] += 1
        W[(b_col_offset + 1) * 2 + 0][base + 3] += 1

        # M5 = (A00+A01)*B11
        U[0 * 3 + a_col0][base + 4] += 1
        U[0 * 3 + a_col1][base + 4] += 1
        V[b_row1 * 8 + b_col_offset + 1][base + 4] += 1
        W[(b_col_offset + 1) * 2 + 0][base + 4] += 1

        # M6 = (A10-A00)*(B01-B11)
        U[1 * 3 + a_col0][base + 5] += 1
        U[0 * 3 + a_col0][base + 5] += -1
        V[b_row0 * 8 + b_col_offset + 1][base + 5] += 1
        V[b_row1 * 8 + b_col_offset + 1][base + 5] += -1
        W[(b_col_offset + 1) * 2 + 1][base + 5] += 1

        # M7 = (A01-A11)*(B00+B10)
        U[0 * 3 + a_col1][base + 6] += 1
        U[1 * 3 + a_col1][base + 6] += -1
        V[b_row0 * 8 + b_col_offset][base + 6] += 1
        V[b_row1 * 8 + b_col_offset][base + 6] += 1
        W[b_col_offset * 2 + 0][base + 6] += -1
        W[(b_col_offset + 1) * 2 + 1][base + 6] += -1

    # 4 blocks of 2x2x2 using Strassen (rank 7 each) = 28
    for block in range(4):
        base = block * 7
        add_strassen(base, 0, 1, 0, 1, block * 2)

    # 8 rank-1 terms for A[:,2] * B[2,:]
    for k in range(8):
        c = 28 + k
        U[0 * 3 + 2][c] += 1
        U[1 * 3 + 2][c] += 1
        V[2 * 8 + k][c] += 1
        W[k * 2 + 0][c] += 1
        W[k * 2 + 1][c] += 1

    if time.time() > deadline:
        return schoolbook()

    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}

    return schoolbook()
# EVOLVE-BLOCK-END
