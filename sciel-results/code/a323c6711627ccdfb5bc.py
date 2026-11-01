"""Tensor rank minimization for 2x5x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    start_time = time.time()
    deadline = start_time + 105.0

    def build_decomposition(ja1, jb1, ja2, jb2, ka, kb):
        """Build a decomposition with two Strassen blocks covering j in {ja1,jb1} and {ja2,jb2}, k in {ka,kb}."""
        rank_max = 60
        U = np.zeros((n * m, rank_max), dtype=float)
        V = np.zeros((m * p, rank_max), dtype=float)
        W = np.zeros((p * n, rank_max), dtype=float)
        col = 0

        def add_term(u_coeffs, v_coeffs, w_coeffs):
            nonlocal col
            if col >= rank_max:
                return
            for u_idx, coeff in u_coeffs:
                U[u_idx, col] += coeff
            for v_idx, coeff in v_coeffs:
                V[v_idx, col] += coeff
            for w_idx, coeff in w_coeffs:
                W[w_idx, col] += coeff
            col += 1

        def strassen_block(ja, jb, ka, kb):
            ua = ja
            ub = m + ja
            va = jb
            vb = m + jb
            wka0 = ka * n
            wka1 = ka * n + 1
            wkb0 = kb * n
            wkb1 = kb * n + 1
            vka = ja * p + ka
            vkb = ja * p + kb
            wka2 = jb * p + ka
            wkb2 = jb * p + kb

            # M1
            add_term([(ua, 1), (vb, 1)], [(vka, 1), (wkb2, 1)], [(wka0, 1), (wkb1, 1)])
            # M2
            add_term([(ub, 1), (vb, 1)], [(vka, 1)], [(wka1, 1), (wkb1, -1)])
            # M3
            add_term([(ua, 1)], [(vkb, 1), (wkb2, -1)], [(wka0 + 2, 1), (wkb1, 1)])
            # M4
            add_term([(vb, 1)], [(wka2, 1), (vka, -1)], [(wka0, 1), (wka1, 1)])
            # M5
            add_term([(ua, 1), (va, 1)], [(wkb2, 1)], [(wka0, -1), (wka0 + 2, 1)])
            # M6
            add_term([(ub, 1), (ua, -1)], [(vka, 1), (vkb, 1)], [(wkb1, 1)])
            # M7
            add_term([(va, 1), (vb, -1)], [(wka2, 1), (wkb2, 1)], [(wka0, 1)])

        strassen_block(ja1, jb1, ka, kb)
        strassen_block(ja2, jb2, ka, kb)

        strassen_set = set()
        for i in range(n):
            for j in [ja1, jb1, ja2, jb2]:
                for k in [ka, kb]:
                    strassen_set.add((i, j, k))

        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) in strassen_set:
                        continue
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx, col] = 1
                    V[v_idx, col] = 1
                    W[w_idx, col] = 1
                    col += 1

        return U[:, :col].copy(), V[:, :col].copy(), W[:, :col].copy(), col

    def verify(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U[i * m + j, :]
                    v_row = V[j * p + k, :]
                    w_row = W[k * n + i, :]
                    result = np.sum(u_row * v_row * w_row)
                    if abs(result - 1) > 1e-10:
                        return False
        return True

    def try_reduce_rank(U, V, W, current_rank, time_limit):
        """Try to reduce rank by finding linear dependencies among rank-1 terms."""
        r = current_rank
        if r <= 1:
            return U, V, W, r

        # Build the tensor as a 3D array T[i,j,k] = sum over c of U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # We want to find if any column c can be expressed as a linear combination of others
        # in the tensor space.

        # Flatten the tensor: T is n*m*p = 60 dimensional
        # Each rank-1 term c gives a vector in R^60:
        # vec_c[i*m+j, j*p+k, k*n+i] = U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]

        # Build matrix M where M[:, c] = flattened tensor for term c
        M = np.zeros((n * m * p, r))
        for c in range(r):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        M[i * m * p + j * p + k, c] = U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]

        # Check numerical rank of M
        try:
            sv = np.linalg.svd(M, compute_uv=False)
        except:
            return U, V, W, r

        # Count how many singular values are above tolerance
        tol = 1e-10
        num_indep = np.sum(sv > tol)

        if num_indep < r:
            # There are dependencies. Try to find a minimal set of independent columns.
            # Use QR with column pivoting to find independent columns
            try:
                # Find a basis via rank-revealing decomposition
                Q, R = np.linalg.qr(M, mode='complete')
                # R is r x r upper triangular
                # Find pivot columns
                diag = np.abs(np.diag(R))
                pivot_cols = np.where(diag > tol)[0]
                num_pivots = len(pivot_cols)
            except:
                return U, V, W, r

            if num_pivots < r:
                # We can potentially reduce rank
                # But this is tricky - the tensor decomposition isn't just about the flattened vectors
                # A linear dependency in the flattened space doesn't directly give a rank reduction
                # in the tensor sense unless we can express the dependent term as a sum of others
                # with the correct rank-1 structure.

                # For now, just note the potential reduction
                pass

        return U, V, W, r

    # Try different Strassen block configurations
    # Original: j in {0,1} and {2,3}, k in {0,1}
    configs = [
        (0, 1, 2, 3, 0, 1),  # original
        (0, 1, 2, 3, 1, 2),
        (0, 1, 2, 3, 2, 3),
        (0, 1, 2, 3, 3, 4),
        (0, 1, 2, 3, 4, 5),
        (0, 2, 1, 3, 0, 1),
        (0, 3, 1, 2, 0, 1),
    ]

    best_result = None
    best_rank = 60

    for cfg in configs:
        if time.time() > deadline:
            break
        ja1, jb1, ja2, jb2, ka, kb = cfg
        # Ensure ja != jb and ka != kb
        if ja1 == jb1 or ja2 == jb2 or ka == kb:
            continue
        # Ensure the two j-pairs are disjoint
        if len({ja1, jb1, ja2, jb2}) != 4:
            continue

        U, V, W, r = build_decomposition(ja1, jb1, ja2, jb2, ka, kb)

        if verify(U, V, W):
            if r < best_rank:
                best_rank = r
                best_result = (U.copy(), V.copy(), W.copy())

    if best_result is None:
        # Fallback to trivial rank 60
        U_fallback = np.zeros((n * m, 60), dtype=float)
        V_fallback = np.zeros((m * p, 60), dtype=float)
        W_fallback = np.zeros((p * n, 60), dtype=float)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fallback[i * m + j, c] = 1
                    V_fallback[j * p + k, c] = 1
                    W_fallback[k * n + i, c] = 1
                    c += 1
        return {"U": U_fallback.tolist(), "V": V_fallback.tolist(), "W": W_fallback.tolist()}

    U, V, W = best_result

    # Try rank reduction on the best decomposition
    if time.time() < deadline - 10:
        U, V, W, r = try_reduce_rank(U, V, W, best_rank, deadline)
        if verify(U, V, W) and r < best_rank:
            best_rank = r
            best_result = (U, V, W)
            U, V, W = best_result

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
