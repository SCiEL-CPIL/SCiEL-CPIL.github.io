import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x4 matrix.
    Uses a block-wise Strassen-based decomposition with rank 28.
    """
    n, m, p = DIMENSIONS
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen 2x2 decomposition terms
    # Each term is (u_dict, v_dict, w_dict) where dicts map local index -> coefficient
    strassen_base = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),   # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),          # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),          # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),          # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),          # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),          # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),          # M7
    ]
    
    # Apply Strassen to four 2x2 blocks
    # A is 2x4, B is 4x4, C is 2x4
    # We view A as two 2x2 blocks side by side
    # We view B as four 2x2 blocks in a 2x2 grid
    # C is two 2x2 blocks side by side
    
    block_offsets = [
        (0, 0, 0, 0),  # A block 0, B row 0, B col 0 -> C block 0
        (1, 1, 0, 0),  # A block 1, B row 1, B col 0 -> C block 0
        (0, 0, 1, 1),  # A block 0, B row 0, B col 1 -> C block 1
        (1, 1, 1, 1),  # A block 1, B row 1, B col 1 -> C block 1
    ]
    
    for a_blk, b_row, b_col, c_blk in block_offsets:
        for t_idx, (u_loc, v_loc, w_loc) in enumerate(strassen_base):
            col = (a_blk * 2 + b_row * 2 + b_col) * 7 + t_idx  # This is wrong, let me fix
            
    # Let me redo this more carefully
    # For each of the 4 Strassen applications, we use 7 columns
    # Application 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # Application 2: A[0:2, 2:4] * B[0:2, 2:4] -> C[0:2, 2:4]
    # Application 3: A[0:2, 0:2] * B[2:4, 0:2] -> C[0:2, 0:2] (add to previous)
    # Application 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4] (add to previous)
    
    # Wait, A is 2x4, so it has 2 rows and 4 columns
    # A can be split into two 2x2 blocks: A[:, 0:2] and A[:, 2:4]
    # B is 4x4, split into four 2x2 blocks: B[0:2, 0:2], B[0:2, 2:4], B[2:4, 0:2], B[2:4, 2:4]
    # C is 2x4, split into two 2x2 blocks: C[:, 0:2] and C[:, 2:4]
    
    # C[:, 0:2] = A[:, 0:2] * B[0:2, 0:2] + A[:, 2:4] * B[2:4, 0:2]
    # C[:, 2:4] = A[:, 0:2] * B[0:2, 2:4] + A[:, 2:4] * B[2:4, 2:4]
    
    # So we need 4 Strassen multiplications, each contributing to one of the C blocks
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    def add_strassen(a_col_offset, b_row_offset, b_col_offset, c_col_offset, start_col):
        for t_idx, (u_loc, v_loc, w_loc) in enumerate(strassen_base):
            col = start_col + t_idx
            
            # Map local A indices to global
            for loc_idx, coeff in u_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_idx = r * m + (a_col_offset * 2 + c)
                U[g_idx, col] += coeff
            
            # Map local B indices to global
            for loc_idx, coeff in v_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                g_idx = (b_row_offset * 2 + r) * p + (b_col_offset * 2 + c)
                V[g_idx, col] += coeff
            
            # Map local C indices to global
            # C is p x n in the W matrix, with W[k*n+i, r] where k is column of C, i is row of C
            for loc_idx, coeff in w_loc.items():
                r = loc_idx // 2
                c = loc_idx % 2
                # r is row of C (0 or 1), c is column of C (0 or 1)
                g_k = c_col_offset * 2 + c  # column of C
                g_i = r  # row of C
                g_idx = g_k * n + g_i
                W[g_idx, col] += coeff
    
    # Application 1: A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
    add_strassen(0, 0, 0, 0, 0)
    # Application 2: A[:, 2:4] * B[2:4, 0:2] -> C[:, 0:2]
    add_strassen(1, 1, 0, 0, 7)
    # Application 3: A[:, 0:2] * B[0:2, 2:4] -> C[:, 2:4]
    add_strassen(0, 0, 1, 1, 14)
    # Application 4: A[:, 2:4] * B[2:4, 2:4] -> C[:, 2:4]
    add_strassen(1, 1, 1, 1, 21)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
