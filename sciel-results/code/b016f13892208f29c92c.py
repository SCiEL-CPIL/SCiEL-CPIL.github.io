"""Tensor decomposition for 4x4 * 4x7 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 7
    start_time = time.time()
    deadline = start_time + 105.0  # safety margin before 120s limit

    def make_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Strassen 2x2x2 terms: (U_coeffs, V_coeffs, W_coeffs)
    # Each is dict from (row,col) to coefficient
    S = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1})
    ]

    # Outer Strassen for 4x4x4: block-level coefficients
    # Blocks: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    OUTER = [
        ({0:1, 3:1}, {0:1, 3:1}, {(0,0):1, (1,1):1}),
        ({2:1, 3:1}, {0:1}, {(1,0):1, (1,1):-1}),
        ({0:1}, {1:1, 3:-1}, {(0,1):1, (1,1):1}),
        ({3:1}, {2:1, 0:-1}, {(0,0):1, (1,0):1}),
        ({0:1, 1:1}, {3:1}, {(0,0):-1, (0,1):1}),
        ({2:1, 0:-1}, {0:1, 1:1}, {(1,1):1}),
        ({1:1, 3:-1}, {2:1, 3:1}, {(0,0):1})
    ]

    # Build the 4x4x4 Strassen part (rank 49)
    def build_4x4x4():
        U = np.zeros((n * m, 49))
        V = np.zeros((m * 4, 49))  # only first 4 columns of B
        W = np.zeros((4 * n, 49))  # only first 4 rows of C
        col = 0
        for o_idx in range(7):
            x_def, y_def, c_block = OUTER[o_idx]
            for s_idx in range(7):
                u_c, v_c, w_c = S[s_idx]
                r = col
                col += 1
                for (li, lj), cx in u_c.items():
                    for b, cb in x_def.items():
                        bi, bj = b // 2, b % 2
                        i = 2 * bi + li
                        j = 2 * bj + lj
                        U[i * m + j, r] += cx * cb
                for (li, lj), cy in v_c.items():
                    for b, cb in y_def.items():
                        bi, bj = b // 2, b % 2
                        i = 2 * bi + li
                        j = 2 * bj + lj
                        V[i * 4 + j, r] += cy * cb
                for (li, lj), cc in w_c.items():
                    for (br, bc), cb_c in c_block.items():
                        i_final = 2 * br + li
                        k_final = 2 * bc + lj
                        W[k_final * n + i_final, r] += cc * cb_c
        return U, V, W

    # Build 2x2x1 schoolbook for a given block
    def build_2x2x1(a_rows, a_cols, b_row, b_col, c_row, c_col):
        """Multiply 2x2 A block by 1x1 B element, producing 2x1 C block."""
        U = np.zeros((n * m, 4))
        V = np.zeros((m * p, 4))
        W = np.zeros((p * n, 4))
        col = 0
        for li in range(2):
            for lj in range(2):
                i = a_rows[li]
                j = a_cols[lj]
                U[i * m + j, col] = 1.0
                V[b_row * p + b_col, col] = 1.0
                W[c_col * n + c_row, col] = 1.0
                col += 1
        return U, V, W

    # Try to find a better decomposition using tensor network approach
    # For 4x4x7, we can decompose as:
    # - 4x4x4 part: Strassen (rank 49)
    # - 4x4x3 part: need to minimize rank
    
    # For the 4x4x3 part, try different strategies:
    # Strategy 1: Schoolbook (rank 48)
    # Strategy 2: Use Strassen-like structure on the 3 columns
    
    # Let's try a hybrid approach for 4x4x3:
    # Split 3 columns into 2+1
    # 4x4x2: Use 2x2 Strassen blocks (4 blocks * 7 = 28)
    # 4x4x1: Schoolbook (16)
    # Total: 49 + 28 + 16 = 93
    
    # Alternative: Try to use a more efficient decomposition for 4x4x3
    # Can we do 4x4x3 in fewer than 44 terms?
    # Lower bound for 4x4x3 is at least 4*4*3/2 = 24 (by Strassen-like bounds)
    # But exact minimal rank is unknown. Let's try to construct something.
    
    # For now, use the known 93-rank decomposition and verify it's correct
    U444, V444, W444 = build_4x4x4()
    
    # 4x4x2 part: 4 blocks of 2x2 Strassen
    U442 = np.zeros((n * m, 28))
    V442 = np.zeros((m * p, 28))
    W442 = np.zeros((p * n, 28))
    
    col = 0
    # A blocks
    A00_r = [0, 1]; A00_c = [0, 1]
    A01_r = [0, 1]; A01_c = [2, 3]
    A10_r = [2, 3]; A10_c = [0, 1]
    A11_r = [2, 3]; A11_c = [2, 3]
    
    # B columns 4,5
    B_top_r = [0, 1]; B_top_c = [4, 5]
    B_bot_r = [2, 3]; B_bot_c = [4, 5]
    
    # C rows for columns 4,5
    C_top_r = [0, 1]; C_top_c = [4, 5]
    C_bot_r = [2, 3]; C_bot_c = [4, 5]
    
    def add_2x2_strassen(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, U, V, W, col):
        for s in S:
            u_c, v_c, w_c = s
            r = col
            col += 1
            for (li, lj), val in u_c.items():
                i = a_rows[li]
                j = a_cols[lj]
                U[i * m + j, r] += val
            for (li, lj), val in v_c.items():
                i = b_rows[li]
                j = b_cols[lj]
                V[i * p + j, r] += val
            for (li, lj), val in w_c.items():
                i = c_rows[li]
                k = c_cols[lj]
                W[k * n + i, r] += val
        return col
    
    col = add_2x2_strassen(A00_r, A00_c, B_top_r, B_top_c, C_top_r, C_top_c, U442, V442, W442, col)
    col = add_2x2_strassen(A01_r, A01_c, B_bot_r, B_bot_c, C_top_r, C_top_c, U442, V442, W442, col)
    col = add_2x2_strassen(A10_r, A10_c, B_top_r, B_top_c, C_bot_r, C_bot_c, U442, V442, W442, col)
    col = add_2x2_strassen(A11_r, A11_c, B_bot_r, B_bot_c, C_bot_r, C_bot_c, U442, V442, W442, col)
    
    assert col == 28, f"4x4x2 rank mismatch: {col} != 28"
    
    # 4x4x1 part: Schoolbook (rank 16)
    U441 = np.zeros((n * m, 16))
    V441 = np.zeros((m * p, 16))
    W441 = np.zeros((p * n, 16))
    col = 0
    for i in range(n):
        for j in range(m):
            U441[i * m + j, col] = 1.0
            V441[j * p + 6, col] = 1.0
            W441[6 * n + i, col] = 1.0
            col += 1
    
    assert col == 16, f"4x4x1 rank mismatch: {col} != 16"
    
    # Combine all parts
    total_rank = 49 + 28 + 16
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Part 1: 4x4x4 (columns 0-3 of B, rows 0-3 of C)
    U[:, :49] = U444
    V[:, :49] = V444
    W[:, :49] = W444
    
    # Part 2: 4x4x2 (columns 4-5 of B, rows 0-3 of C)
    U[:, 49:77] = U442
    V[:, 49:77] = V442
    W[:, 49:77] = W442
    
    # Part 3: 4x4x1 (column 6 of B, row 6 of C)
    U[:, 77:93] = U441
    V[:, 77:93] = V441
    W[:, 77:93] = W441
    
    # Verify the decomposition
    Ua = np.array(U.tolist())
    Va = np.array(V.tolist())
    Wa = np.array(W.tolist())
    
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    u_idx = i_idx * m + j_idx
    v_idx = j_idx * p + k_idx
    w_idx = k_idx * n + i_idx
    
    err = np.max(np.abs((Ua[u_idx] * Va[v_idx] * Wa[w_idx]).sum(axis=1) - 1.0))
    
    if err > 1e-8:
        # Fallback to schoolbook if something went wrong
        return make_schoolbook()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
