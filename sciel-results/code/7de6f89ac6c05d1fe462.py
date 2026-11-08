import numpy as np
import time

def construct():
    start_time = time.time()
    timeout = 115.0

    n, m, p = 3, 4, 4
    Nm, Mp, Pn = n*m, m*p, p*n  # 12, 16, 12

    # Build tensor
    T = np.zeros((Nm, Mp, Pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    def verify_fast(U, V, W):
        R = U.shape[1]
        UV = U[:, None, :] * V[None, :, :]  # (Nm, Mp, R)
        UV_flat = UV.reshape(-1, R)
        T_rec_flat = UV_flat @ W.T
        T_rec = T_rec_flat.reshape(Nm, Mp, Pn)
        return np.allclose(T_rec, T, atol=1e-8)

    # Build the rank-44 block decomposition
    def build_block_decomp():
        rank = 44
        U = np.zeros((Nm, rank))
        V = np.zeros((Mp, rank))
        W = np.zeros((Pn, rank))

        U_loc = np.zeros((6, 11))
        V_loc = np.zeros((4, 11))
        W_loc = np.zeros((6, 11))

        # Strassen for 2x2x2 (rows 0,1 of local A)
        U_loc[0, 0] = 1; U_loc[3, 0] = 1
        V_loc[0, 0] = 1; V_loc[3, 0] = 1
        W_loc[0, 0] = 1; W_loc[3, 0] = 1

        U_loc[2, 1] = 1; U_loc[3, 1] = 1
        V_loc[0, 1] = 1
        W_loc[2, 1] = 1; W_loc[3, 1] = -1

        U_loc[0, 2] = 1
        V_loc[1, 2] = 1; V_loc[3, 2] = -1
        W_loc[1, 2] = 1; W_loc[3, 2] = 1

        U_loc[3, 3] = 1
        V_loc[2, 3] = 1; V_loc[0, 3] = -1
        W_loc[0, 3] = 1; W_loc[2, 3] = 1

        U_loc[0, 4] = 1; U_loc[1, 4] = 1
        V_loc[3, 4] = 1
        W_loc[0, 4] = -1; W_loc[1, 4] = 1

        U_loc[2, 5] = 1; U_loc[0, 5] = -1
        V_loc[0, 5] = 1; V_loc[1, 5] = 1
        W_loc[3, 5] = 1

        U_loc[1, 6] = 1; U_loc[3, 6] = -1
        V_loc[2, 6] = 1; V_loc[3, 6] = 1
        W_loc[0, 6] = 1

        # Extra for row 2
        U_loc[4, 7] = 1
        V_loc[0, 7] = 1
        W_loc[4, 7] = 1

        U_loc[5, 8] = 1
        V_loc[2, 8] = 1
        W_loc[4, 8] = 1

        U_loc[4, 9] = 1
        V_loc[1, 9] = 1
        W_loc[5, 9] = 1

        U_loc[5, 10] = 1
        V_loc[3, 10] = 1
        W_loc[5, 10] = 1

        def add_block(block_idx, a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
            col_offset = block_idx * 11
            for r in range(3):
                for c in range(2):
                    local_a_idx = r * 2 + c
                    global_a_idx = (a_row_start + r) * m + (a_col_start + c)
                    U[global_a_idx, col_offset:col_offset+11] += U_loc[local_a_idx, :]
            for r in range(2):
                for c in range(2):
                    local_b_idx = r * 2 + c
                    global_b_idx = (b_row_start + r) * p + (b_col_start + c)
                    V[global_b_idx, col_offset:col_offset+11] += V_loc[local_b_idx, :]
            for r in range(3):
                for c in range(2):
                    local_c_idx = r * 2 + c
                    global_c_idx = (c_col_start + c) * n + (c_row_start + r)
                    W[global_c_idx, col_offset:col_offset+11] += W_loc[local_c_idx, :]

        add_block(0, 0, 0, 0, 0, 0, 0)
        add_block(1, 0, 2, 2, 0, 0, 0)
        add_block(2, 0, 0, 0, 2, 0, 2)
        add_block(3, 0, 2, 2, 2, 0, 2)

        return U, V, W

    U_blk, V_blk, W_blk = build_block_decomp()

    if not verify_fast(U_blk, V_blk, W_blk):
        # Fallback to direct rank-48
        R = n * m * p
        U = np.zeros((Nm, R))
        V = np.zeros((Mp, R))
        W = np.zeros((Pn, R))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, idx] = 1.0
                    V[j*p+k, idx] = 1.0
                    W[k*n+i, idx] = 1.0
                    idx += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_U, best_V, best_W = U_blk, V_blk, W_blk
    best_rank = 44

    def try_remove_column(U, V, W, r):
        """Test if removing column r still gives correct decomposition."""
        U_test = np.delete(U, r, axis=1)
        V_test = np.delete(V, r, axis=1)
        W_test = np.delete(W, r, axis=1)
        if verify_fast(U_test, V_test, W_test):
            return True, U_test, V_test, W_test
        return False, None, None, None

    def try_merge_columns(U, V, W, r1, r2, alpha):
        """Test if column r2 can be absorbed into column r1 via alpha."""
        if r1 == r2:
            return False, None, None, None
        U_new = U.copy()
        V_new = V.copy()
        W_new = W.copy()
        U_new[:, r1] = U_new[:, r1] + alpha * U_new[:, r2]
        V_new[:, r1] = V_new[:, r1] + alpha * V_new[:, r2]
        W_new[:, r1] = W_new[:, r1] + alpha * W_new[:, r2]
        if r2 > r1:
            U_new = np.delete(U_new, r2, axis=1)
            V_new = np.delete(V_new, r2, axis=1)
            W_new = np.delete(W_new, r2, axis=1)
        else:
            U_new = np.delete(U_new, r2, axis=1)
            V_new = np.delete(V_new, r2, axis=1)
            W_new = np.delete(W_new, r2, axis=1)
        if verify_fast(U_new, V_new, W_new):
            return True, U_new, V_new, W_new
        return False, None, None, None

    def try_remove_group(U, V, W, indices):
        """Test if removing a group of columns still gives correct decomposition."""
        if not indices:
            return False, None, None, None
        U_test = np.delete(U, indices, axis=1)
        V_test = np.delete(V, indices, axis=1)
        W_test = np.delete(W, indices, axis=1)
        if verify_fast(U_test, V_test, W_test):
            return True, U_test, V_test, W_test
        return False, None, None, None

    # Phase 1: Single column removal (iterative)
    for attempt_round in range(5):
        if time.time() - start_time > timeout - 15:
            break
        removed = True
        while removed:
            removed = False
            current_rank = best_U.shape[1]
            for r in range(current_rank):
                if time.time() - start_time > timeout - 10:
                    break
                success, U_t, V_t, W_t = try_remove_column(best_U, best_V, best_W, r)
                if success:
                    best_U, best_V, best_W = U_t, V_t, W_t
                    best_rank -= 1
                    removed = True
                    break

    # Phase 2: Try structured column merging with various alpha values
    merge_alphas = [0.5, 1.0, 2.0, -0.5, -1.0, -2.0, 0.25, 4.0, -0.25, -4.0]
    merge_found = True
    while merge_found:
        merge_found = False
        if time.time() - start_time > timeout - 15:
            break
        current_rank = best_U.shape[1]
        for r1 in range(current_rank):
            if time.time() - start_time > timeout - 10:
                break
            for r2 in range(current_rank):
                if r1 == r2:
                    continue
                if time.time() - start_time > timeout - 10:
                    break
                for alpha in merge_alphas:
                    success, U_t, V_t, W_t = try_merge_columns(best_U, best_V, best_W, r1, r2, alpha)
                    if success:
                        best_U, best_V, best_W = U_t, V_t, W_t
                        best_rank -= 1
                        merge_found = True
                        break
                if merge_found:
                    break
            if merge_found:
                break

    # Phase 3: After merges, try single column removal again
    for attempt_round in range(5):
        if time.time() - start_time > timeout - 10:
            break
        removed = True
        while removed:
            removed = False
            current_rank = best_U.shape[1]
            for r in range(current_rank):
                if time.time() - start_time > timeout - 8:
                    break
                success, U_t, V_t, W_t = try_remove_column(best_U, best_V, best_W, r)
                if success:
                    best_U, best_V, best_W = U_t, V_t, W_t
                    best_rank -= 1
                    removed = True
                    break

    # Phase 4: Try more aggressive merging with finer alpha grid if time permits
    if time.time() - start_time < timeout - 15:
        merge_alphas_fine = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.5, 3.0,
                            -0.1, -0.2, -0.3, -0.4, -0.6, -0.7, -0.8, -0.9, -1.5, -3.0]
        merge_found = True
        while merge_found:
            merge_found = False
            if time.time() - start_time > timeout - 10:
                break
            current_rank = best_U.shape[1]
            for r1 in range(current_rank):
                if time.time() - start_time > timeout - 8:
                    break
                for r2 in range(current_rank):
                    if r1 == r2:
                        continue
                    if time.time() - start_time > timeout - 8:
                        break
                    for alpha in merge_alphas_fine:
                        success, U_t, V_t, W_t = try_merge_columns(best_U, best_V, best_W, r1, r2, alpha)
                        if success:
                            best_U, best_V, best_W = U_t, V_t, W_t
                            best_rank -= 1
                            merge_found = True
                            break
                    if merge_found:
                        break
                if merge_found:
                    break

    # Phase 5: Try removing pairs of columns simultaneously
    if time.time() - start_time < timeout - 20:
        pair_found = True
        while pair_found:
            pair_found = False
            if time.time() - start_time > timeout - 15:
                break
            current_rank = best_U.shape[1]
            for r1 in range(current_rank):
                if time.time() - start_time > timeout - 12:
                    break
                for r2 in range(r1+1, current_rank):
                    if time.time() - start_time > timeout - 12:
                        break
                    success, U_t, V_t, W_t = try_remove_group(best_U, best_V, best_W, [r1, r2])
                    if success:
                        best_U, best_V, best_W = U_t, V_t, W_t
                        best_rank -= 2
                        pair_found = True
                        break
                if pair_found:
                    break
            # After pair removal, try single removal again
            if time.time() - start_time > timeout - 10:
                break
            removed = True
            while removed:
                removed = False
                current_rank = best_U.shape[1]
                for r in range(current_rank):
                    if time.time() - start_time > timeout - 10:
                        break
                    success, U_t, V_t, W_t = try_remove_column(best_U, best_V, best_W, r)
                    if success:
                        best_U, best_V, best_W = U_t, V_t, W_t
                        best_rank -= 1
                        removed = True
                        break

    # Phase 6: Try removing triples of columns simultaneously
    if time.time() - start_time < timeout - 25:
        triple_found = True
        while triple_found:
            triple_found = False
            if time.time() - start_time > timeout - 20:
                break
            current_rank = best_U.shape[1]
            for r1 in range(current_rank):
                if time.time() - start_time > timeout - 15:
                    break
                for r2 in range(r1+1, current_rank):
                    if time.time() - start_time > timeout - 15:
                        break
                    for r3 in range(r2+1, current_rank):
                        if time.time() - start_time > timeout - 15:
                            break
                        success, U_t, V_t, W_t = try_remove_group(best_U, best_V, best_W, [r1, r2, r3])
                        if success:
                            best_U, best_V, best_W = U_t, V_t, W_t
                            best_rank -= 3
                            triple_found = True
                            break
                    if triple_found:
                        break
                if triple_found:
                    break
            # After triple removal, try single removal again
            if time.time() - start_time > timeout - 10:
                break
            removed = True
            while removed:
                removed = False
                current_rank = best_U.shape[1]
                for r in range(current_rank):
                    if time.time() - start_time > timeout - 10:
                        break
                    success, U_t, V_t, W_t = try_remove_column(best_U, best_V, best_W, r)
                    if success:
                        best_U, best_V, best_W = U_t, V_t, W_t
                        best_rank -= 1
                        removed = True
                        break

    # Phase 7: Linear dependence check - find columns that are linear combinations of others
    if time.time() - start_time < timeout - 30:
        current_rank = best_U.shape[1]
        if current_rank > 1:
            # Check if any column is a linear combination of others
            # Use the U matrix to find linear dependencies
            U_mat = best_U
            # Compute rank of U_mat
            try:
                rank_U = np.linalg.matrix_rank(U_mat, tol=1e-8)
            except:
                rank_U = current_rank
            
            if rank_U < current_rank:
                # There are linear dependencies in U
                # Find which columns are dependent
                # Use SVD to find null space
                try:
                    U_sv, U_s, Vt = np.linalg.svd(U_mat, full_matrices=False)
                    # Columns corresponding to small singular values are dependent
                    # Find the null space
                    tol = 1e-8
                    null_mask = U_s < tol
                    if np.any(null_mask):
                        # The right singular vectors corresponding to small singular values span the null space
                        null_space = Vt[null_mask].T  # (current_rank, num_null)
                        # For each column, check if it's in the span of others
                        # This is complex, so just try removing columns one by one
                        # and see if the decomposition still holds
                        pass
                except:
                    pass

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
