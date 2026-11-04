import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 110.0

    # Strassen rank-7 factors for 2x2
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)

    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)

    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)

    # Build rank-49 decomposition via Kronecker product of Strassen factors
    i_vals = np.repeat(np.arange(4), 4)
    j_vals = np.tile(np.arange(4), 4)
    i_hi = (i_vals // 2).astype(int)
    i_lo = (i_vals % 2).astype(int)
    j_hi = (j_vals // 2).astype(int)
    j_lo = (j_vals % 2).astype(int)

    u_hi_vecs = U2[i_hi * 2 + j_hi]
    u_lo_vecs = U2[i_lo * 2 + j_lo]
    u_outer = u_hi_vecs[:, :, np.newaxis] * u_lo_vecs[:, np.newaxis, :]
    u_mat = u_outer.reshape(16, 49)

    j_vals_v = np.repeat(np.arange(4), 4)
    k_vals_v = np.tile(np.arange(4), 4)
    j_hi_v = (j_vals_v // 2).astype(int)
    j_lo_v = (j_vals_v % 2).astype(int)
    k_hi_v = (k_vals_v // 2).astype(int)
    k_lo_v = (k_vals_v % 2).astype(int)

    v_hi_vecs = V2[j_hi_v * 2 + k_hi_v]
    v_lo_vecs = V2[j_lo_v * 2 + k_lo_v]
    v_outer = v_hi_vecs[:, :, np.newaxis] * v_lo_vecs[:, np.newaxis, :]
    v_mat = v_outer.reshape(16, 49)

    k_vals_w = np.repeat(np.arange(4), 4)
    i_vals_w = np.tile(np.arange(4), 4)
    k_hi_w = (k_vals_w // 2).astype(int)
    k_lo_w = (k_vals_w % 2).astype(int)
    i_hi_w = (i_vals_w // 2).astype(int)
    i_lo_w = (i_vals_w % 2).astype(int)

    w_hi_vecs = W2[k_hi_w * 2 + i_hi_w]
    w_lo_vecs = W2[k_lo_w * 2 + i_lo_w]
    w_outer = w_hi_vecs[:, :, np.newaxis] * w_lo_vecs[:, np.newaxis, :]
    w_mat = w_outer.reshape(16, 49)

    # Verify correctness
    valid = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                prod = np.sum(u_mat[i * 4 + j] * v_mat[j * 4 + k] * w_mat[k * 4 + i])
                if prod != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid or time.time() - start_time > deadline:
        n, m, p = 4, 4, 4
        rank = n * m * p
        U_f = [[0] * rank for _ in range(n * m)]
        V_f = [[0] * rank for _ in range(m * p)]
        W_f = [[0] * rank for _ in range(p * n)]
        col = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_f[ii * m + jj][col] = 1
                    V_f[jj * p + kk][col] = 1
                    W_f[kk * n + ii][col] = 1
                    col += 1
        return {"U": U_f, "V": V_f, "W": W_f}

    # Check if we can reduce rank by finding linearly dependent columns
    # Build the "slice" matrices and check for rank deficiency in the combined system
    # For each column c, the rank-1 term contributes u_mat[:,c] ⊗ v_mat[:,c] ⊗ w_mat[:,c]
    # Two columns c1, c2 are "redundant" if their contributions are linearly dependent
    # This is hard to check directly, so let's try a different approach:
    # Check if any column is all zeros in any of U, V, or W
    # If u_mat[:,c] has all zeros, or v_mat[:,c] has all zeros, or w_mat[:,c] has all zeros,
    # then that column contributes nothing and can be removed.
    
    nonzero_cols = []
    for c in range(49):
        if np.any(u_mat[:, c] != 0) and np.any(v_mat[:, c] != 0) and np.any(w_mat[:, c] != 0):
            nonzero_cols.append(c)
    
    if len(nonzero_cols) < 49 and time.time() - start_time < deadline:
        u_reduced = u_mat[:, nonzero_cols]
        v_reduced = v_mat[:, nonzero_cols]
        w_reduced = w_mat[:, nonzero_cols]
        
        # Verify reduced version
        valid_reduced = True
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    prod = np.sum(u_reduced[i * 4 + j] * v_reduced[j * 4 + k] * w_reduced[k * 4 + i])
                    if prod != 1:
                        valid_reduced = False
                        break
                if not valid_reduced:
                    break
            if not valid_reduced:
                break
        
        if valid_reduced:
            return {
                "U": u_reduced.tolist(),
                "V": v_reduced.tolist(),
                "W": w_reduced.tolist()
            }
    
    return {
        "U": u_mat.tolist(),
        "V": v_mat.tolist(),
        "W": w_mat.tolist()
    }
