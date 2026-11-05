import numpy as np

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5

    # We'll build a rank-54 decomposition using Strassen for 2x2x2 blocks
    # and optimized handling of the remaining parts.
    # Strategy: Split the 3x4x5 multiplication into blocks:
    #   - 2x2x2 blocks (Strassen, rank 7 each)
    #   - 2x2x1 blocks (rank 4 each)
    #   - 1x2x2 blocks (rank 4 each)
    #   - 1x2x1 blocks (rank 2 each)
    #
    # We'll use a more efficient decomposition:
    # Split A (3x4) into:
    #   A1 (2x4) - rows 0,1
    #   A2 (1x4) - row 2
    # Split B (4x5) into columns:
    #   B_cols 0-1, 2-3, 4
    #
    # For A1 * B:
    #   A1[:,0:2] * B[0:2,:] + A1[:,2:4] * B[2:4,:]
    #   Each is 2x2x5 = 2x2x2 + 2x2x2 + 2x2x1
    #   Rank: 7 + 7 + 4 = 18 each, total 36
    #
    # For A2 * B:
    #   A2[0:2] * B[0:2,:] + A2[2:4] * B[2:4,:]
    #   Each is 1x2x5 = 1x2x2 + 1x2x2 + 1x2x1
    #   Rank: 4 + 4 + 2 = 10 each, total 20
    #
    # Total: 36 + 20 = 56
    #
    # To get rank 54, we can try to optimize the 1x2x2 blocks.
    # Actually, 1x2x2 has rank at most 4, but can we do better?
    # C0 = A0*B00 + A1*B10
    # C1 = A0*B01 + A1*B11
    # This is 4 terms. Can we reduce to 3?
    # Let's try: M1 = (A0+A1)B00, M2 = A0(B01-B11), M3 = A1(B10+B11)
    # C0 = M1 + M3 - M2? No, that doesn't work cleanly.
    # Actually, the rank of 1x2x2 is exactly 4 (it's a 2x2 matrix multiplication
    # where one side is 1x2, so it's just 4 independent products).
    #
    # Let's try a different approach: use rank 9 for 2x2x3.
    # If R(2x2x3) = 9, then 2x2x5 = 2x2x3 + 2x2x2 has rank 9+7=16.
    # Then A1*B would be 16+16=32, and A2*B would be 10+10=20, total 52.
    # But I need the explicit rank-9 decomposition for 2x2x3.
    #
    # Since I'm uncertain about the rank-9 coefficients, let me stick with
    # the known correct rank-56 decomposition but write it more cleanly.
    # Actually, let me try to use a simpler approach: just use Strassen for
    # all 2x2x2 blocks and schoolbook for the rest, but be very careful with indices.

    U_parts = []
    V_parts = []
    W_parts = []

    def strassen_2x2x2():
        """Returns U(4x7), V(4x7), W(4x7) for 2x2x2 Strassen."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))

        # M1 = (A00+A11)(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1

        # M2 = (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1

        # M3 = A00(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1

        # M4 = A11(B10-B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1

        # M5 = (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1

        # M6 = (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1

        # M7 = (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1

        return U, V, W

    def schoolbook_2x2x1():
        """Returns U(4x4), V(2x4), W(2x4) for 2x2x1 schoolbook."""
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        return U, V, W

    def schoolbook_1x2x2():
        """Returns U(2x4), V(4x4), W(2x4) for 1x2x2 schoolbook."""
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
        U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
        U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
        return U, V, W

    def schoolbook_1x2x1():
        """Returns U(2x2), V(2x2), W(1x2) for 1x2x1 schoolbook."""
        U = np.zeros((2, 2))
        V = np.zeros((2, 2))
        W = np.zeros((1, 2))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        return U, V, W

    def add_block(U_blk, V_blk, W_blk, u_idx, v_idx, w_idx):
        for r in range(U_blk.shape[1]):
            U_vec = np.zeros(n * m)
            for i, idx in enumerate(u_idx):
                U_vec[idx] = U_blk[i, r]
            V_vec = np.zeros(m * p)
            for i, idx in enumerate(v_idx):
                V_vec[idx] = V_blk[i, r]
            W_vec = np.zeros(p * n)
            for i, idx in enumerate(w_idx):
                W_vec[idx] = W_blk[i, r]
            U_parts.append(U_vec)
            V_parts.append(V_vec)
            W_parts.append(W_vec)

    # U indices: row i, col j -> i*m + j = i*4 + j
    # V indices: row i, col j -> i*p + j = i*5 + j
    # W indices: col k, row i -> k*n + i = k*3 + i

    # Block 1: A[0:2,0:2] * B[0:2,0:2] (2x2x2 Strassen)
    U_b, V_b, W_b = strassen_2x2x2()
    add_block(U_b, V_b, W_b,
              u_idx=[0, 1, 4, 5],
              v_idx=[0, 1, 5, 6],
              w_idx=[0, 3, 1, 4])

    # Block 2: A[0:2,0:2] * B[0:2,2:4] (2x2x2 Strassen)
    U_b, V_b, W_b = strassen_2x2x2()
    add_block(U_b, V_b, W_b,
              u_idx=[0, 1, 4, 5],
              v_idx=[2, 3, 7, 8],
              w_idx=[6, 9, 7, 10])

    # Block 3: A[0:2,0:2] * B[0:2,4] (2x2x1 schoolbook)
    U_b, V_b, W_b = schoolbook_2x2x1()
    add_block(U_b, V_b, W_b,
              u_idx=[0, 1, 4, 5],
              v_idx=[4, 9],
              w_idx=[12, 13])

    # Block 4: A[0:2,2:4] * B[2:4,0:2] (2x2x2 Strassen)
    U_b, V_b, W_b = strassen_2x2x2()
    add_block(U_b, V_b, W_b,
              u_idx=[2, 3, 6, 7],
              v_idx=[10, 11, 15, 16],
              w_idx=[0, 3, 1, 4])

    # Block 5: A[0:2,2:4] * B[2:4,2:4] (2x2x2 Strassen)
    U_b, V_b, W_b = strassen_2x2x2()
    add_block(U_b, V_b, W_b,
              u_idx=[2, 3, 6, 7],
              v_idx=[12, 13, 17, 18],
              w_idx=[6, 9, 7, 10])

    # Block 6: A[0:2,2:4] * B[2:4,4] (2x2x1 schoolbook)
    U_b, V_b, W_b = schoolbook_2x2x1()
    add_block(U_b, V_b, W_b,
              u_idx=[2, 3, 6, 7],
              v_idx=[14, 19],
              w_idx=[12, 13])

    # Block 7: A[2,0:2] * B[0:2,0:2] (1x2x2 schoolbook)
    U_b, V_b, W_b = schoolbook_1x2x2()
    add_block(U_b, V_b, W_b,
              u_idx=[8, 9],
              v_idx=[0, 1, 5, 6],
              w_idx=[2, 5])

    # Block 8: A[2,0:2] * B[0:2,2:4] (1x2x2 schoolbook)
    U_b, V_b, W_b = schoolbook_1x2x2()
    add_block(U_b, V_b, W_b,
              u_idx=[8, 9],
              v_idx=[2, 3, 7, 8],
              w_idx=[8, 11])

    # Block 9: A[2,0:2] * B[0:2,4] (1x2x1 schoolbook)
    U_b, V_b, W_b = schoolbook_1x2x1()
    add_block(U_b, V_b, W_b,
              u_idx=[8, 9],
              v_idx=[4, 9],
              w_idx=[14])

    # Block 10: A[2,2:4] * B[2:4,0:2] (1x2x2 schoolbook)
    U_b, V_b, W_b = schoolbook_1x2x2()
    add_block(U_b, V_b, W_b,
              u_idx=[10, 11],
              v_idx=[10, 11, 15, 16],
              w_idx=[2, 5])

    # Block 11: A[2,2:4] * B[2:4,2:4] (1x2x2 schoolbook)
    U_b, V_b, W_b = schoolbook_1x2x2()
    add_block(U_b, V_b, W_b,
              u_idx=[10, 11],
              v_idx=[12, 13, 17, 18],
              w_idx=[8, 11])

    # Block 12: A[2,2:4] * B[2:4,4] (1x2x1 schoolbook)
    U_b, V_b, W_b = schoolbook_1x2x1()
    add_block(U_b, V_b, W_b,
              u_idx=[10, 11],
              v_idx=[14, 19],
              w_idx=[14])

    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
