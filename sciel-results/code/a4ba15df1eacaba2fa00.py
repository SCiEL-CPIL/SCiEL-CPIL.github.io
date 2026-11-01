import time
import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    start_time = time.time()
    deadline = 140

    n, m, p = DIMENSIONS
    # Tensor dimensions: U is (n*m, rank) = (8, rank)
    # V is (m*p, rank) = (16, rank)
    # W is (p*n, rank) = (8, rank)
    dim_u = n * m  # 8
    dim_v = m * p  # 16
    dim_w = p * n  # 8

    # Build the target tensor T[i,j,k]
    # i = row_a * m + col_a
    # j = row_b * p + col_b
    # k = col_c * n + row_c (trace convention)
    T = np.zeros((dim_u, dim_v, dim_w), dtype=float)
    for row_a in range(n):
        for col_a in range(m):
            i = row_a * m + col_a
            for row_b in range(m):
                for col_b in range(p):
                    j = row_b * p + col_b
                    for row_c in range(n):
                        for col_c in range(p):
                            k = col_c * n + row_c
                            if col_a == row_b and row_a == row_c and col_b == col_c:
                                T[i, j, k] = 1.0

    T_flat = T.flatten()  # 512 entries

    def compute_error(U, V, W):
        """Compute max absolute error of the decomposition."""
        # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # Vectorize: for each (i,j), the vector over k is sum_r U[i,r]*V[j,r]*W[k,:]
        max_err = 0.0
        for i in range(dim_u):
            for j in range(dim_v):
                # For fixed i,j: sum_r U[i,r]*V[j,r]*W[k,:] over r
                coeffs = U[i, :] * V[j, :]  # shape (rank,)
                pred = coeffs @ W  # shape (dim_w,)
                err = np.max(np.abs(pred - T[i, j, :]))
                if err > max_err:
                    max_err = err
        return max_err

    def try_rank(r, max_iters=500, n_restarts=200, seed=42):
        """Try to find a rank-r decomposition using alternating least squares."""
        rng = np.random.RandomState(seed)
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None

        for restart in range(n_restarts):
            if time.time() > deadline - 20:
                break

            # Initialize with small random values
            U = rng.randn(dim_u, r) * 0.5
            V = rng.randn(dim_v, r) * 0.5
            W = rng.randn(dim_w, r) * 0.5

            # Add some structure: try to match the known non-zero pattern
            # The tensor has non-zeros where specific (i,j,k) triples match
            # For each non-zero T[i,j,k]=1, we need sum_r U[i,r]*V[j,r]*W[k,r] = 1

            for it in range(max_iters):
                if time.time() > deadline - 10:
                    break

                # Update W: for each k, W[k,:] should satisfy
                # sum_r U[i,r]*V[j,r]*W[k,r] = T[i,j,k] for all (i,j)
                # This is: (U^T * V_hadamard)^T * W[k,:] = T[:, :, k]
                # where V_hadamard[i,j] = V[i,r]*V[j,r] ... no, it's per-r
                # Actually: for fixed k, we need sum_r (sum_{i,j} T[i,j,k] * U[i,r] * V[j,r]) ... 
                # No. The equation is: for each (i,j,k): sum_r U[i,r]*V[j,r]*W[k,r] = T[i,j,k]
                # For fixed k: sum_r W[k,r] * (sum_{i,j} T[i,j,k] * U[i,r] * V[j,r]) -- no that's wrong.
                # 
                # For fixed k: the equation over all (i,j) is:
                #   sum_r U[i,r] * V[j,r] * W[k,r] = T[i,j,k]  for all i,j
                # Let M_k be a (dim_u * dim_v) x r matrix where
                #   M_k[(i,j), r] = U[i,r] * V[j,r]
                # Then M_k @ W[k,:] = T[:,:,k].flatten()
                # This is a least squares problem for W[k,:].

                # Similarly for fixed i: 
                #   sum_r U[i,r] * (V[j,r] * W[k,r]) = T[i,j,k] for all j,k
                # Let N_i be a (dim_v * dim_w) x r matrix where
                #   N_i[(j,k), r] = V[j,r] * W[k,r]
                # Then N_i @ U[i,:] = T[i,:,:].flatten()

                # And for fixed j:
                #   sum_r V[j,r] * (U[i,r] * W[k,r]) = T[i,j,k] for all i,k
                # Let P_j be a (dim_u * dim_w) x r matrix where
                #   P_j[(i,k), r] = U[i,r] * W[k,r]
                # Then P_j @ V[j,:] = T[:,j,:].flatten()

                # Update W
                for k in range(dim_w):
                    # Build M_k: (dim_u * dim_v, r)
                    M_k = np.outer(U.flatten(), V.flatten()).reshape(dim_u, dim_v, r)
                    M_k = M_k.reshape(dim_u * dim_v, r)
                    rhs = T[:, :, k].flatten()
                    W[k, :], _, _, _ = np.linalg.lstsq(M_k, rhs, rcond=None)

                # Update U
                for i in range(dim_u):
                    # Build N_i: (dim_v * dim_w, r)
                    N_i = np.outer(V.flatten(), W.flatten()).reshape(dim_v, dim_w, r)
                    N_i = N_i.reshape(dim_v * dim_w, r)
                    rhs = T[i, :, :].flatten()
                    U[i, :], _, _, _ = np.linalg.lstsq(N_i, rhs, rcond=None)

                # Update V
                for j in range(dim_v):
                    # Build P_j: (dim_u * dim_w, r)
                    P_j = np.outer(U.flatten(), W.flatten()).reshape(dim_u, dim_w, r)
                    P_j = P_j.reshape(dim_u * dim_w, r)
                    rhs = T[:, j, :].flatten()
                    V[j, :], _, _, _ = np.linalg.lstsq(P_j, rhs, rcond=None)

                # Check error periodically
                if it % 50 == 0 or it == max_iters - 1:
                    err = compute_error(U, V, W)
                    if err < best_err:
                        best_err = err
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if err < 1e-9:
                        return U, V, W, err

        return best_U, best_V, best_W, best_err

    def try_rank_structured(r, max_iters=300, n_restarts=100, seed=123):
        """
        Structured approach: Initialize based on Strassen-like patterns.
        For rank r, try to find a decomposition by starting from a rank-28 Strassen
        solution and projecting down to rank r.
        """
        rng = np.random.RandomState(seed)
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None

        # First, get the Strassen rank-28 solution as a starting point
        U_s = np.array([
            [1, 0, 1, 0, 1, 0, 1],
            [0, 0, 1, -1, 0, 1, 0],
            [0, 1, 0, 0, 1, 0, -1],
            [1, 1, 0, 1, 0, -1, 0]
        ], dtype=float)

        V_s = np.array([
            [1, 1, 0, 0, 1, 0, 1],
            [0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 0, 1, 1]
        ], dtype=float)

        T_2 = np.zeros((4, 4, 4), dtype=float)
        for ra in range(2):
            for ca in range(2):
                ii = ra * 2 + ca
                for rb in range(2):
                    for cb in range(2):
                        jj = rb * 2 + cb
                        for rc in range(2):
                            for cc in range(2):
                                kk = cc * 2 + rc
                                if ca == rb and ra == rc and cb == cc:
                                    T_2[ii, jj, kk] = 1.0

        M_sys = np.zeros((64, 28), dtype=float)
        T_vec2 = np.zeros(64, dtype=float)
        for ii in range(4):
            for jj in range(4):
                for kk in range(4):
                    eq = ii * 16 + jj * 4 + kk
                    T_vec2[eq] = T_2[ii, jj, kk]
                    for s in range(7):
                        M_sys[eq, kk * 7 + s] = U_s[ii, s] * V_s[jj, s]

        W_vec, _, _, _ = np.linalg.lstsq(M_sys, T_vec2, rcond=None)
        W_s = W_vec.reshape(4, 7)

        # Build rank-28 for 2x4x4
        U28 = np.zeros((8, 28), dtype=float)
        V28 = np.zeros((16, 28), dtype=float)
        W28 = np.zeros((8, 28), dtype=float)

        offset = 0
        for bi in range(2):
            for bj in range(2):
                for ra in range(2):
                    for ca in range(2):
                        local_i = ra * 2 + ca
                        global_i = ra * 4 + (2 * bi + ca)
                        U28[global_i, offset:offset + 7] = U_s[local_i, :]
                for rb in range(2):
                    for cb in range(2):
                        local_j = rb * 2 + cb
                        global_j = (2 * bi + rb) * 4 + (2 * bj + cb)
                        V28[global_j, offset:offset + 7] = V_s[local_j, :]
                for rc in range(2):
                    for cc in range(2):
                        local_k = cc * 2 + rc
                        global_k = (2 * bj + cc) * 2 + rc
                        W28[global_k, offset:offset + 7] = W_s[local_k, :]
                offset += 7

        # Now try to project to rank r by random linear combinations of columns
        for restart in range(n_restarts):
            if time.time() > deadline - 15:
                break

            # Create a random mixing matrix to project 28 dims to r dims
            # U_new = U28 @ M_u, V_new = V28 @ M_v, W_new = W28 @ M_w
            # But this won't preserve the tensor structure unless M_u=M_v=M_w
            # Actually, for the tensor to be preserved, we need:
            # sum_{r'=1}^{r} U[i,r']*V[j,r']*W[k,r'] = T[i,j,k]
            # If U = U28 @ A, V = V28 @ B, W = W28 @ C where A,B,C are 28xr
            # then sum_r' U[i,r']*V[j,r']*W[k,r'] = sum_r' (U28@A)[i,r']*(V28@B)[j,r']*(W28@C)[k,r']
            # This is NOT equal to sum_s U28[i,s]*V28[j,s]*W28[k,s] in general.
            
            # Better approach: use random initialization seeded from Strassen
            U = U28[:, :r].copy() if r <= 28 else np.hstack([U28, rng.randn(8, r-28)])
            V = V28[:, :r].copy() if r <= 28 else np.hstack([V28, rng.randn(16, r-28)])
            W = W28[:, :r].copy() if r <= 28 else np.hstack([W28, rng.randn(8, r-28)])
            
            # Add random perturbation
            U += rng.randn(8, r) * 0.1
            V += rng.randn(16, r) * 0.1
            W += rng.randn(8, r) * 0.1

            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break

                # Alternating least squares
                for k in range(dim_w):
                    M_k = np.outer(U.flatten(), V.flatten()).reshape(dim_u, dim_v, r)
                    M_k = M_k.reshape(dim_u * dim_v, r)
                    rhs = T[:, :, k].flatten()
                    W[k, :], _, _, _ = np.linalg.lstsq(M_k, rhs, rcond=None)

                for i in range(dim_u):
                    N_i = np.outer(V.flatten(), W.flatten()).reshape(dim_v, dim_w, r)
                    N_i = N_i.reshape(dim_v * dim_w, r)
                    rhs = T[i, :, :].flatten()
                    U[i, :], _, _, _ = np.linalg.lstsq(N_i, rhs, rcond=None)

                for j in range(dim_v):
                    P_j = np.outer(U.flatten(), W.flatten()).reshape(dim_u, dim_w, r)
                    P_j = P_j.reshape(dim_u * dim_w, r)
                    rhs = T[:, j, :].flatten()
                    V[j, :], _, _, _ = np.linalg.lstsq(P_j, rhs, rcond=None)

                if it % 20 == 0 or it == max_iters - 1:
                    err = compute_error(U, V, W)
                    if err < best_err:
                        best_err = err
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if err < 1e-9:
                        return U, V, W, err

        return best_U, best_V, best_W, best_err

    # Try ranks from 27 down to 24
    best_rank = 32
    best_U, best_V, best_W = None, None, None

    # Fallback: schoolbook rank 32
    U_fb = np.zeros((8, 32))
    V_fb = np.zeros((16, 32))
    W_fb = np.zeros((8, 32))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1

    for target_rank in [27, 26, 25, 24]:
        if time.time() > deadline - 30:
            break

        # Try structured approach first (faster convergence)
        U_try, V_try, W_try, err = try_rank_structured(target_rank, max_iters=200, n_restarts=50, seed=target_rank*1000)
        if err < 1e-9:
            best_rank = target_rank
            best_U, best_V, best_W = U_try, V_try, W_try
            break

        # Try pure random approach
        U_try2, V_try2, W_try2, err2 = try_rank(target_rank, max_iters=200, n_restarts=100, seed=target_rank*2000)
        if err2 < 1e-9:
            best_rank = target_rank
            best_U, best_V, best_W = U_try2, V_try2, W_try2
            break

        if err < best_err if best_err != float('inf') else True:
            pass  # Keep trying lower ranks

    # If we didn't find anything better than 32, use the Strassen rank-28
    if best_rank > 28:
        # Build Strassen rank-28
        U_s = np.array([
            [1, 0, 1, 0, 1, 0, 1],
            [0, 0, 1, -1, 0, 1, 0],
            [0, 1, 0, 0, 1, 0, -1],
            [1, 1, 0, 1, 0, -1, 0]
        ], dtype=float)

        V_s = np.array([
            [1, 1, 0, 0, 1, 0, 1],
            [0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 0, 1, 1]
        ], dtype=float)

        T_2 = np.zeros((4, 4, 4), dtype=float)
        for ra in range(2):
            for ca in range(2):
                ii = ra * 2 + ca
                for rb in range(2):
                    for cb in range(2):
                        jj = rb * 2 + cb
                        for rc in range(2):
                            for cc in range(2):
                                kk = cc * 2 + rc
                                if ca == rb and ra == rc and cb == cc:
                                    T_2[ii, jj, kk] = 1.0

        M_sys = np.zeros((64, 28), dtype=float)
        T_vec2 = np.zeros(64, dtype=float)
        for ii in range(4):
            for jj in range(4):
                for kk in range(4):
                    eq = ii * 16 + jj * 4 + kk
                    T_vec2[eq] = T_2[ii, jj, kk]
                    for s in range(7):
                        M_sys[eq, kk * 7 + s] = U_s[ii, s] * V_s[jj, s]

        W_vec, _, _, _ = np.linalg.lstsq(M_sys, T_vec2, rcond=None)
        W_s = W_vec.reshape(4, 7)

        U28 = np.zeros((8, 28), dtype=float)
        V28 = np.zeros((16, 28), dtype=float)
        W28 = np.zeros((8, 28), dtype=float)

        offset = 0
        for bi in range(2):
            for bj in range(2):
                for ra in range(2):
                    for ca in range(2):
                        local_i = ra * 2 + ca
                        global_i = ra * 4 + (2 * bi + ca)
                        U28[global_i, offset:offset + 7] = U_s[local_i, :]
                for rb in range(2):
                    for cb in range(2):
                        local_j = rb * 2 + cb
                        global_j = (2 * bi + rb) * 4 + (2 * bj + cb)
                        V28[global_j, offset:offset + 7] = V_s[local_j, :]
                for rc in range(2):
                    for cc in range(2):
                        local_k = cc * 2 + rc
                        global_k = (2 * bj + cc) * 2 + rc
                        W28[global_k, offset:offset + 7] = W_s[local_k, :]
                offset += 7

        # Verify
        err28 = compute_error(U28, V28, W28)
        if err28 < 1e-9:
            best_rank = 28
            best_U, best_V, best_W = U28, V28, W28

    # Final check: if we still have no valid solution, use schoolbook
    if best_U is None or best_rank >= 32:
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    # Final verification
    final_err = compute_error(best_U, best_V, best_W)
    if final_err > 1e-9:
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
