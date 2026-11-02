"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    
    # Target rank is 28 based on block Strassen decomposition
    # M_2,4,4 decomposes into 4 blocks of M_2,2,2
    # Rank(M_2,2,2) = 7
    # Total rank = 4 * 7 = 28
    
    def build_tensor():
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T

    T = build_tensor()

    def verify(U, V, W, tol=1e-8):
        """Check exact decomposition."""
        recon = np.einsum('at,bt,ct->abc', U, V, W)
        return np.max(np.abs(recon - T)) < tol

    def to_lists(U, V, W, r):
        """Convert numpy arrays to list-of-lists format."""
        U_list = [[float(U[i, t]) for t in range(r)] for i in range(n * m)]
        V_list = [[float(V[i, t]) for t in range(r)] for i in range(m * p)]
        W_list = [[float(W[i, t]) for t in range(r)] for i in range(p * n)]
        return {"U": U_list, "V": V_list, "W": W_list}

    def strassen_2x2_decomp():
        """
        Returns U, V, W for 2x2 matrix multiplication with rank 7.
        Indices for 2x2 matrices:
        A: 0, 1, 2, 3 (row-major)
        B: 0, 1, 2, 3 (row-major)
        C: 0, 1, 2, 3 (row-major)
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))

        # t=0: M1 = (A00+A11)*(B00+B11)
        U[0, 0] = 1.0; U[3, 0] = 1.0
        V[0, 0] = 1.0; V[3, 0] = 1.0
        W[0, 0] = 1.0; W[3, 0] = 1.0

        # t=1: M2 = (A10+A11)*B00
        U[2, 1] = 1.0; U[3, 1] = 1.0
        V[0, 1] = 1.0
        W[1, 1] = 1.0; W[3, 1] = -1.0

        # t=2: M3 = A00*(B01-B11)
        U[0, 2] = 1.0
        V[1, 2] = 1.0; V[3, 2] = -1.0
        W[2, 2] = 1.0; W[3, 2] = 1.0

        # t=3: M4 = A11*(B10-B00)
        U[3, 3] = 1.0
        V[2, 3] = 1.0; V[0, 3] = -1.0
        W[0, 3] = 1.0; W[1, 3] = 1.0

        # t=4: M5 = (A00+A01)*B11
        U[0, 4] = 1.0; U[1, 4] = 1.0
        V[3, 4] = 1.0
        W[0, 4] = -1.0; W[2, 4] = 1.0

        # t=5: M6 = (A10-A00)*(B00+B01)
        U[2, 5] = 1.0; U[0, 5] = -1.0
        V[0, 5] = 1.0; V[1, 5] = 1.0
        W[3, 5] = 1.0

        # t=6: M7 = (A01-A11)*(B10+B11)
        U[1, 6] = 1.0; U[3, 6] = -1.0
        V[2, 6] = 1.0; V[3, 6] = 1.0
        W[0, 6] = 1.0

        return U, V, W

    U_s, V_s, W_s = strassen_2x2_decomp()

    full_rank = 28
    U_full = np.zeros((n * m, full_rank))
    V_full = np.zeros((m * p, full_rank))
    W_full = np.zeros((p * n, full_rank))

    # Define the 4 blocks
    # Block (row_block, col_block)
    # A is 2x4, split into A_left (2x2) and A_right (2x2)
    # B is 4x4, split into 4 2x2 blocks
    # C is 2x4, split into C_left (2x2) and C_right (2x2)
    
    # A indices: 0,1 (row 0), 2,3 (row 1) for left block?
    # A is 2x4. Row 0: 0,1,2,3. Row 1: 4,5,6,7.
    # Left block A_L: cols 0,1. Indices 0,1 (row 0), 4,5 (row 1).
    # Right block A_R: cols 2,3. Indices 2,3 (row 0), 6,7 (row 1).
    
    # B is 4x4.
    # B_TL: rows 0,1, cols 0,1. Indices 0,1,4,5.
    # B_TR: rows 0,1, cols 2,3. Indices 2,3,6,7.
    # B_BL: rows 2,3, cols 0,1. Indices 8,9,12,13.
    # B_BR: rows 2,3, cols 2,3. Indices 10,11,14,15.
    
    # C is 2x4.
    # C_L: cols 0,1. Indices 0,1 (row 0), 2,3 (row 1).
    # C_R: cols 2,3. Indices 4,5 (row 0), 6,7 (row 1).
    
    # Mapping local 2x2 indices (0,1,2,3) to global indices
    def map_a(global_row, global_col, local_idx):
        # local_idx: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
        lr = local_idx // 2
        lc = local_idx % 2
        return (global_row + lr) * 4 + (global_col + lc)

    def map_b(global_row, global_col, local_idx):
        lr = local_idx // 2
        lc = local_idx % 2
        return (global_row + lr) * 4 + (global_col + lc)

    def map_c(global_row, global_col, local_idx):
        # C is 2x4.
        # local_idx: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
        lr = local_idx // 2
        lc = local_idx % 2
        return (global_row + lr) * 4 + (global_col + lc)

    # Block 1: A_L * B_TL -> C_L
    # A_L starts at (0,0). B_TL starts at (0,0). C_L starts at (0,0).
    for t in range(7):
        col = 0 * 7 + t
        for li in range(4):
            U_full[map_a(0, 0, li), col] = U_s[li, t]
            V_full[map_b(0, 0, li), col] = V_s[li, t]
            W_full[map_c(0, 0, li), col] = W_s[li, t]

    # Block 2: A_R * B_BL -> C_L
    # A_R starts at (0,2). B_BL starts at (2,0). C_L starts at (0,0).
    for t in range(7):
        col = 1 * 7 + t
        for li in range(4):
            U_full[map_a(0, 2, li), col] = U_s[li, t]
            V_full[map_b(2, 0, li), col] = V_s[li, t]
            W_full[map_c(0, 0, li), col] = W_s[li, t]

    # Block 3: A_L * B_TR -> C_R
    # A_L starts at (0,0). B_TR starts at (0,2). C_R starts at (0,2).
    for t in range(7):
        col = 2 * 7 + t
        for li in range(4):
            U_full[map_a(0, 0, li), col] = U_s[li, t]
            V_full[map_b(0, 2, li), col] = V_s[li, t]
            W_full[map_c(0, 2, li), col] = W_s[li, t]

    # Block 4: A_R * B_BR -> C_R
    # A_R starts at (0,2). B_BR starts at (2,2). C_R starts at (0,2).
    for t in range(7):
        col = 3 * 7 + t
        for li in range(4):
            U_full[map_a(0, 2, li), col] = U_s[li, t]
            V_full[map_b(2, 2, li), col] = V_s[li, t]
            W_full[map_c(0, 2, li), col] = W_s[li, t]

    # Verify rank 28
    if verify(U_full, V_full, W_full, tol=1e-8):
        return to_lists(U_full, V_full, W_full, full_rank)
    else:
        # Fallback schoolbook rank 32
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return to_lists(U, V, W, rank)
# EVOLVE-BLOCK-END
