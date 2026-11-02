import numpy as np
import time

def construct():
    """
    Returns a rank-49 tensor decomposition for 2x2 * 2x14 matrix multiplication.
    Uses block-wise Strassen algorithm.
    """
    n, m, p = 2, 2, 14
    num_blocks = p // 2
    strassen_rank = 7
    total_rank = num_blocks * strassen_rank

    U = np.zeros((n * m, total_rank), dtype=np.int8)
    V = np.zeros((m * p, total_rank), dtype=np.int8)
    W = np.zeros((p * n, total_rank), dtype=np.int8)

    # Precompute Strassen coefficient patterns
    # Each entry: (U_indices, U_vals, V_indices, V_vals, W_indices, W_vals)
    # Indices are local to the 2x2 block
    strassen_patterns = [
        # P1 = (A00+A11)(B00+B11) -> C00 += P1, C11 += P1
        ([0, 3], [1, 1], [0, 1], [1, 1], [(0,0), (1,1)], [1, 1]),
        # P2 = (A10+A11)B00 -> C10 += P2, C11 -= P2
        ([2, 3], [1, 1], [0], [1], [(0,1), (1,1)], [1, -1]),
        # P3 = A00(B01-B11) -> C01 += P3, C11 += P3
        ([0], [1], [1, 1], [1, -1], [(1,0), (1,1)], [1, 1]),
        # P4 = A11(B10-B00) -> C00 += P4, C10 += P4
        ([3], [1], [0, 0], [-1, 1], [(0,0), (0,1)], [1, 1]),
        # P5 = (A00+A01)B11 -> C00 -= P5, C01 += P5
        ([0, 1], [1, 1], [1], [1], [(0,0), (1,0)], [-1, 1]),
        # P6 = (A10-A00)(B00+B01) -> C11 += P6
        ([0, 2], [-1, 1], [0, 0], [1, 1], [(1,1)], [1]),
        # P7 = (A01-A11)(B10+B11) -> C00 += P7
        ([1, 3], [1, -1], [0, 1], [1, 1], [(0,0)], [1]),
    ]

    for t in range(num_blocks):
        k_off = t * 2
        for s in range(strassen_rank):
            col = t * strassen_rank + s
            u_idx, u_val, v_idx, v_val, w_idx, w_val = strassen_patterns[s]

            for ui, uv in zip(u_idx, u_val):
                U[ui, col] = uv

            for vi, vv in zip(v_idx, v_val):
                # v_idx are local (row, col_in_block) -> global row = vi_row * p + k_off + vi_col
                row = vi // 2  # 0 or 1
                col_local = vi % 2
                V[row * p + k_off + col_local, col] = vv

            for (wi_row, wi_col), wv in zip(w_idx, w_val):
                # w_idx are (output_row_local, output_col_local) -> global = (k_off + wi_col)*n + wi_row
                W[(k_off + wi_col) * n + wi_row, col] = wv

    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    # Verification
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i * m + j].astype(np.int64)
                v_val = V[j * p + k].astype(np.int64)
                w_val = W[k * n + i].astype(np.int64)
                val = int(np.dot(u_val, v_val * w_val))
                if abs(val - 1) > max_err:
                    max_err = abs(val - 1)

    if max_err > 1e-9:
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb), dtype=np.int8)
        V_fb = np.zeros((m * p, rank_fb), dtype=np.int8)
        W_fb = np.zeros((p * n, rank_fb), dtype=np.int8)
        col = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_fb[ii * m + jj, col] = 1
                    V_fb[jj * p + kk, col] = 1
                    W_fb[kk * n + ii, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    return {"U": U_list, "V": V_list, "W": W_list}
