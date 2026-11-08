import numpy as np
import time

def construct():
    """
    Returns a rank-44 exact tensor decomposition for multiplying a 3x4 matrix by a 4x4 matrix.
    
    Strategy: Split into 4 blocks of 3x2x2 multiplication.
    Each 3x2x2 block uses Strassen (rank 7) for the 2x2 part (rows 0-1 of A)
    and trivial (rank 4) for the 1x2 part (row 2 of A). Total per block: 11.
    Total rank: 4 * 11 = 44.
    
    W index convention: W[j*n + i, r] where j is column, i is row.
    """
    n, m, p = 3, 4, 4
    rank = 44

    U = np.zeros((n * m, rank))  # 12 x 44
    V = np.zeros((m * p, rank))  # 16 x 44
    W = np.zeros((p * n, rank))  # 12 x 44

    # Build the 11-rank decomposition for 3x2x2 multiplication
    # A_local: 3x2, indices (r, c) -> r*2+c, so 0..5
    # B_local: 2x2, indices (r, c) -> r*2+c, so 0..3
    # C_local: 3x2, W index = j*3+i where j is col (0,1), i is row (0,1,2)
    #   C[0,0]->0, C[1,0]->1, C[2,0]->2, C[0,1]->3, C[1,1]->4, C[2,1]->5

    U_loc = np.zeros((6, 11))
    V_loc = np.zeros((4, 11))
    W_loc = np.zeros((6, 11))

    # Strassen for A[0:2,:] (2x2) * B (2x2) -> C[0:2,:] (2x2)
    # A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # C indices: 0=(0,0), 1=(1,0), 2=(2,0), 3=(0,1), 4=(1,1), 5=(2,1)
    # W indices: C[0,0]->0, C[1,0]->1, C[0,1]->3, C[1,1]->4

    # M1 = (a00+a11)(b00+b11) -> c00 += M1, c11 += M1
    U_loc[0, 0] = 1; U_loc[3, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[4, 0] = 1

    # M2 = (a10+a11)b00 -> c10 += M2, c11 -= M2
    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[1, 1] = 1; W_loc[4, 1] = -1

    # M3 = a00(b01-b11) -> c01 += M3, c11 += M3
    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[3, 2] = -1
    W_loc[3, 2] = 1; W_loc[4, 2] = 1

    # M4 = a11(b10-b00) -> c00 += M4, c10 += M4
    U_loc[3, 3] = 1
    V_loc[2, 3] = 1; V_loc[0, 3] = -1
    W_loc[0, 3] = 1; W_loc[1, 3] = 1

    # M5 = (a00+a01)b11 -> c00 -= M5, c01 += M5
    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[3, 4] = 1

    # M6 = (a10-a00)(b00+b01) -> c11 += M6
    U_loc[2, 5] = 1; U_loc[0, 5] = -1
    V_loc[0, 5] = 1; V_loc[1, 5] = 1
    W_loc[4, 5] = 1

    # M7 = (a01-a11)(b10+b11) -> c00 += M7
    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[2, 6] = 1; V_loc[3, 6] = 1
    W_loc[0, 6] = 1

    # Trivial for A[2,:] (1x2) * B (2x2) -> C[2,:] (1x2)
    # A indices: 4=(2,0), 5=(2,1)
    # C indices: C[2,0]->2, C[2,1]->5

    # a20*b00 -> c20
    U_loc[4, 7] = 1
    V_loc[0, 7] = 1
    W_loc[2, 7] = 1

    # a21*b10 -> c20
    U_loc[5, 8] = 1
    V_loc[2, 8] = 1
    W_loc[2, 8] = 1

    # a20*b01 -> c21
    U_loc[4, 9] = 1
    V_loc[1, 9] = 1
    W_loc[5, 9] = 1

    # a21*b11 -> c21
    U_loc[5, 10] = 1
    V_loc[3, 10] = 1
    W_loc[5, 10] = 1

    # Helper to add a block
    def add_block(block_idx, a_col_start, b_row_start, b_col_start, c_col_start):
        col_offset = block_idx * 11

        # Global A indices: row * 4 + col for 3 rows, 2 cols starting at a_col_start
        a_indices = [(r * 4 + (a_col_start + c)) for r in range(3) for c in range(2)]

        # Global B indices: row * 4 + col for 2 rows, 2 cols
        b_indices = [(b_row_start + r) * 4 + (b_col_start + c) for r in range(2) for c in range(2)]

        # Global W indices: col * 3 + row
        # Local W: j_loc * 3 + i_loc, global: (c_col_start + j_loc) * 3 + i_loc
        w_map = []
        for i_loc in range(3):
            for j_loc in range(2):
                local_idx = j_loc * 3 + i_loc
                global_idx = (c_col_start + j_loc) * 3 + i_loc
                w_map.append((local_idx, global_idx))

        for i, idx in enumerate(a_indices):
            U[idx, col_offset:col_offset+11] += U_loc[i, :]
        for i, idx in enumerate(b_indices):
            V[idx, col_offset:col_offset+11] += V_loc[i, :]
        for local_idx, global_idx in w_map:
            W[global_idx, col_offset:col_offset+11] += W_loc[local_idx, :]

    # Block 0: A cols 0-1, B rows 0-1 cols 0-1 -> C cols 0-1
    add_block(0, 0, 0, 0, 0)
    # Block 1: A cols 2-3, B rows 2-3 cols 0-1 -> C cols 0-1
    add_block(1, 2, 2, 0, 0)
    # Block 2: A cols 0-1, B rows 0-1 cols 2-3 -> C cols 2-3
    add_block(2, 0, 0, 2, 2)
    # Block 3: A cols 2-3, B rows 2-3 cols 2-3 -> C cols 2-3
    add_block(3, 2, 2, 2, 2)

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    np.random.seed(42)
    A = np.random.rand(3, 4)
    B = np.random.rand(4, 4)
    C_exact = A @ B
    
    C_decomp = np.zeros((3, 4))
    for r in range(44):
        u_mat = U[:, r].reshape(3, 4)
        v_mat = V[:, r].reshape(4, 4)
        w_mat = W[:, r].reshape(4, 3)
        M_r = u_mat @ v_mat
        term_r = (M_r.T * w_mat).T
        C_decomp += term_r
    
    diff = np.max(np.abs(C_exact - C_decomp))
    print(f"Max diff: {diff}")
    print("PASS" if diff < 1e-9 else "FAIL")
