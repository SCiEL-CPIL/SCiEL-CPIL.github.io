# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline for safety margin (150s limit)

    n, m, p = 4, 4, 9

    def time_ok():
        return time.time() < deadline

    # 1. Define 2x2x2 Strassen decomposition (Rank 7)
    U_222 = np.zeros((4, 7), dtype=float)
    V_222 = np.zeros((4, 7), dtype=float)
    W_222 = np.zeros((4, 7), dtype=float)

    U_222[0, 0] = 1; U_222[3, 0] = 1
    V_222[0, 0] = 1; V_222[3, 0] = 1
    W_222[0, 0] = 1; W_222[3, 0] = 1

    U_222[2, 1] = 1; U_222[3, 1] = 1
    V_222[0, 1] = 1
    W_222[2, 1] = 1; W_222[3, 1] = 1

    U_222[0, 2] = 1
    V_222[1, 2] = 1; V_222[3, 2] = -1
    W_222[1, 2] = 1; W_222[3, 2] = 1

    U_222[3, 3] = 1
    V_222[2, 3] = 1; V_222[0, 3] = -1
    W_222[0, 3] = 1; W_222[3, 3] = 1

    U_222[0, 4] = 1; U_222[1, 4] = 1
    V_222[3, 4] = 1
    W_222[0, 4] = 1; W_222[1, 4] = -1

    U_222[2, 5] = 1; U_222[0, 5] = -1
    V_222[0, 5] = 1; V_222[1, 5] = 1
    W_222[2, 5] = 1

    U_222[1, 6] = 1; U_222[3, 6] = -1
    V_222[2, 6] = 1; V_222[3, 6] = 1
    W_222[0, 6] = 1; W_222[1, 6] = 1

    # 2. Construct 4x4x4 Strassen decomposition (Rank 49)
    U_444 = np.zeros((16, 49), dtype=float)
    V_444 = np.zeros((16, 49), dtype=float)
    W_444 = np.zeros((16, 49), dtype=float)

    def add_block_term(U_target, V_target, W_target,
                       a_blocks, b_blocks, c_blocks, col_offset):
        for s in range(7):
            col = col_offset + s
            for (br, bc, sign) in a_blocks:
                for i_local in range(2):
                    for j_local in range(2):
                        i_global = 2*br + i_local
                        j_global = 2*bc + j_local
                        u_idx = i_global * 4 + j_global
                        u_local_idx = i_local * 2 + j_local
                        U_target[u_idx, col] += sign * U_222[u_local_idx, s]
            for (br, bc, sign) in b_blocks:
                for i_local in range(2):
                    for j_local in range(2):
                        i_global = 2*br + i_local
                        j_global = 2*bc + j_local
                        v_idx = i_global * 4 + j_global
                        v_local_idx = i_local * 2 + j_local
                        V_target[v_idx, col] += sign * V_222[v_local_idx, s]
            for (br, bc, sign) in c_blocks:
                for i_local in range(2):
                    for j_local in range(2):
                        i_global = 2*br + i_local
                        j_global = 2*bc + j_local
                        w_idx = i_global * 4 + j_global
                        w_local_idx = i_local * 2 + j_local
                        W_target[w_idx, col] += sign * W_222[w_local_idx, s]

    add_block_term(U_444, V_444, W_444,
                   [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], 0)
    add_block_term(U_444, V_444, W_444,
                   [(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,1)], 7)
    add_block_term(U_444, V_444, W_444,
                   [(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)], 14)
    add_block_term(U_444, V_444, W_444,
                   [(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,1,1)], 21)
    add_block_term(U_444, V_444, W_444,
                   [(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,1), (0,1,-1)], 28)
    add_block_term(U_444, V_444, W_444,
                   [(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,0,1)], 35)
    add_block_term(U_444, V_444, W_444,
                   [(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1), (0,1,1)], 42)

    # 3. Construct 4x4x1 Schoolbook decomposition (Rank 16)
    U_441 = np.zeros((16, 16), dtype=float)
    V_441 = np.zeros((4, 16), dtype=float)
    W_441 = np.zeros((4, 16), dtype=float)
    for i in range(4):
        for j in range(4):
            col = i * 4 + j
            U_441[i * 4 + j, col] = 1
            V_441[j, col] = 1
            W_441[i, col] = 1

    # 4. Assemble full 4x4x9 decomposition
    total_rank = 114
    U_total = np.zeros((16, total_rank), dtype=float)
    V_total = np.zeros((36, total_rank), dtype=float)
    W_total = np.zeros((36, total_rank), dtype=float)

    U_total[:, 0:49] = U_444
    V_part1 = np.zeros((36, 49), dtype=float)
    for j in range(4):
        for k in range(4):
            V_part1[j*9 + k, :] = V_444[j*4 + k, :]
    V_total[:, 0:49] = V_part1
    W_part1 = np.zeros((36, 49), dtype=float)
    for k in range(4):
        for i in range(4):
            W_part1[k*4 + i, :] = W_444[k*4 + i, :]
    W_total[:, 0:49] = W_part1

    U_total[:, 49:98] = U_444
    V_part2 = np.zeros((36, 49), dtype=float)
    for j in range(4):
        for k in range(4):
            V_part2[j*9 + (4+k), :] = V_444[j*4 + k, :]
    V_total[:, 49:98] = V_part2
    W_part2 = np.zeros((36, 49), dtype=float)
    for k in range(4):
        for i in range(4):
            W_part2[(4+k)*4 + i, :] = W_444[k*4 + i, :]
    W_total[:, 49:98] = W_part2

    U_total[:, 98:114] = U_441
    V_part3 = np.zeros((36, 16), dtype=float)
    for j in range(4):
        V_part3[j*9 + 8, :] = V_441[j, :]
    V_total[:, 98:114] = V_part3
    W_part3 = np.zeros((36, 16), dtype=float)
    for i in range(4):
        W_part3[8*4 + i, :] = W_441[i, :]
    W_total[:, 98:114] = W_part3

    # 5. Fast validation using tensor contraction (vectorized)
    def validate_fast(U, V, W, tol=1e-9):
        U_arr = np.asarray(U, dtype=float)
        V_arr = np.asarray(V, dtype=float)
        W_arr = np.asarray(W, dtype=float)
        r = U_arr.shape[1]
        if V_arr.shape[1] != r or W_arr.shape[1] != r:
            return False
        # Vectorized check: for each (i,j,k), compute sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        # Build index arrays
        i_idx = np.repeat(np.arange(n), m*p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n*m)
        u_rows = i_idx * m + j_idx
        v_rows = j_idx * p + k_idx
        w_rows = k_idx * n + i_idx
        # Compute all dot products at once
        prod = U_arr[u_rows, :] * V_arr[v_rows, :] * W_arr[w_rows, :]
        sums = prod.sum(axis=1)
        return np.all(np.abs(sums - 1.0) <= tol)

    # 6. Rank-reduction search: try to remove redundant columns
    def try_reduce_columns(U, V, W, max_iters=500):
        """Try to remove columns that are linearly dependent in a tensor sense."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        r = U.shape[1]
        keep = list(range(r))
        improved = True
        iter_count = 0
        while improved and time_ok() and iter_count < max_iters:
            improved = False
            iter_count += 1
            for c in list(keep):
                if len(keep) <= 1:
                    break
                # Try removing column c
                new_keep = [x for x in keep if x != c]
                U_c = U[:, new_keep]
                V_c = V[:, new_keep]
                W_c = W[:, new_keep]
                if validate_fast(U_c, V_c, W_c, tol=1e-8):
                    U, V, W = U_c, V_c, W_c
                    keep = new_keep
                    improved = True
                    break
        return U, V, W, len(keep)

    # 7. Try random orthogonal mixing followed by sparsification (soft threshold)
    def try_random_mixing(U, V, W, n_trials=50, tol=1e-9):
        """Apply random invertible linear transformations and try to sparsify."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        r = U.shape[1]
        best_rank = r
        best_U, best_V, best_W = U, V, W
        rng = np.random.RandomState(12345)
        for t in range(n_trials):
            if not time_ok():
                break
            # Random invertible transform
            Q = rng.randn(r, r)
            Q, _ = np.linalg.qr(Q)
            U_n = U @ Q
            V_n = V @ Q
            W_n = W @ Q
            # Sparsify: zero out small entries
            for thresh in [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]:
                U_s = np.where(np.abs(U_n) < thresh, 0, U_n)
                V_s = np.where(np.abs(V_n) < thresh, 0, V_n)
                W_s = np.where(np.abs(W_n) < thresh, 0, W_n)
                # Remove zero columns
                col_mask = (np.any(U_s != 0, axis=0) |
                            np.any(V_s != 0, axis=0) |
                            np.any(W_s != 0, axis=0))
                if col_mask.sum() < best_rank:
                    U_c = U_s[:, col_mask]
                    V_c = V_s[:, col_mask]
                    W_c = W_s[:, col_mask]
                    if validate_fast(U_c, V_c, W_c, tol=tol):
                        best_rank = col_mask.sum()
                        best_U, best_V, best_W = U_c, V_c, W_c
                        U, V, W = U_c, V_c, W_c
                        r = best_rank
                        break  # Break out of thresh loop if improved
        return best_U, best_V, best_W, best_rank

    # Verify the initial Strassen decomposition
    if not validate_fast(U_total, V_total, W_total):
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = np.zeros((n*m, rank_sb), dtype=float)
        V_sb = np.zeros((m*p, rank_sb), dtype=float)
        W_sb = np.zeros((p*n, rank_sb), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m + j, col] = 1
                    V_sb[j*p + k, col] = 1
                    W_sb[k*n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    # Current best
    best_U, best_V, best_W = U_total, V_total, W_total
    best_rank = total_rank

    # Attempt column reduction (greedy)
    if time_ok():
        U_r, V_r, W_r, r_new = try_reduce_columns(U_total, V_total, W_total)
        if r_new < best_rank:
            best_U, best_V, best_W = U_r, V_r, W_r
            best_rank = r_new

    # Attempt random mixing + sparsification
    if time_ok():
        U_m, V_m, W_m, r_new = try_random_mixing(best_U, best_V, best_W, n_trials=50)
        if r_new < best_rank:
            best_U, best_V, best_W = U_m, V_m, W_m
            best_rank = r_new

    # Second pass: try more aggressive column reduction after mixing
    if time_ok() and best_rank > 50:
        U_r2, V_r2, W_r2, r_new2 = try_reduce_columns(best_U, best_V, best_W, max_iters=300)
        if r_new2 < best_rank:
            best_U, best_V, best_W = U_r2, V_r2, W_r2
            best_rank = r_new2

    # Final validation
    if validate_fast(best_U, best_V, best_W):
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = np.zeros((n*m, rank_sb), dtype=float)
        V_sb = np.zeros((m*p, rank_sb), dtype=float)
        W_sb = np.zeros((p*n, rank_sb), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m + j, col] = 1
                    V_sb[j*p + k, col] = 1
                    W_sb[k*n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
