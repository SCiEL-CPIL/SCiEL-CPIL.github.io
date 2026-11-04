import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # safety margin well before the 150s kill

    def now():
        return time.time()

    def time_left():
        return deadline - now()

    # ---- Build the target tensor T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i ----
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Precompute for fast residual: residual R = sum_r U[:,r] V[:,r] W[:,r]
    # R[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # We compute via einsum over r for the full tensor.

    def residual(U, V, W):
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        return np.einsum('ar,br,cr->abc', U, V, W)

    def verify(U, V, W, atol=1e-7):
        R = residual(U, V, W)
        return np.max(np.abs(T - R)) <= atol

    # ---- Schoolbook fallback (rank n*m*p = 56) ----
    def schoolbook():
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
        return U, V, W

    # ---- Rank-50 Strassen-based base ----
    # A 2x4x7 product. Use Strassen on 2x2 blocks in (i,k) and split j into
    # 3 windows of width 2 (j=0,1), (j=2,3), (j=4,5) plus a leftover column k=6.
    # Each Strassen 2x2 block over (i in {i0,i0+1}, k in {k0,k0+1}) with a fixed
    # 2-wide j window {j0, j0+1} costs 7 rank-1 terms.
    def strassen_rank50():
        rank = 50
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0

        def add_strassen(i0, j0, k0):
            nonlocal col
            # 7 standard Strassen terms for C[i0:i0+2, k0:k0+2] from A[i0:i0+2,j0:j0+2]*B[j0:j0+2,k0:k0+2]
            # M1 = (A00+A11)(B00+B11)
            U[i0 * m + j0, col] += 1; U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1; V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1; W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1
            # M2 = A11*(B00-B10)
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1; V[j0 * p + (k0), col] -= 0
            V[(j0) * p + (k0), col] -= 0
            V[j0 * p + k0, col] += 0
            # (recompute cleanly below)
            # Reset this column to avoid confusion
            U[(i0 + 1) * m + (j0 + 1), col - 1]  # no-op guard
            # --- clean rewrite of the 7 terms ---
            # (we'll just build from scratch in a helper to avoid bugs)
            return col

        # The above inline approach is error-prone. Build the 7 terms explicitly and
        # unambiguously for a 2x2 Strassen block.
        def strassen_block(i0, j0, k0):
            nonlocal col
            # Term 1: (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
            U[i0 * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1
            # Term 2: A11*(B00 - B10) -> C10 += M2, C11 -= M2
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            V[(j0 + 1) * p + k0, col] -= 1
            W[k0 * n + (i0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] -= 1
            col += 1
            # Term 3: (A01 - A11)*B11 -> C01 += M3, C11 += M3
            U[i0 * m + (j0 + 1), col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] -= 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[(k0 + 1) * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1
            # Term 4: (A00 - A01)*B00 -> C00 += M4, C10 += M4
            U[i0 * m + j0, col] += 1
            U[i0 * m + (j0 + 1), col] -= 1
            V[j0 * p + k0, col] += 1
            W[k0 * n + i0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            col += 1
            # Term 5: (A00+A10)*B11 -> C01 += M5, C11 += M5
            U[i0 * m + j0, col] += 1
            U[(i0 + 1) * m + j0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[(k0 + 1) * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1
            # Term 6: (A01-A11)*(B00+B10) -> C00 -= M6, C10 += M6
            U[i0 * m + (j0 + 1), col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] -= 1
            V[j0 * p + k0, col] += 1
            V[(j0 + 1) * p + k0, col] += 1
            W[k0 * n + i0, col] -= 1
            W[k0 * n + (i0 + 1), col] += 1
            col += 1
            # Term 7: (A00-A10)*(B11-B10) -> C01 += M7, C11 -= M7
            U[i0 * m + j0, col] += 1
            U[(i0 + 1) * m + j0, col] -= 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            V[(j0 + 1) * p + k0, col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] -= 1
            col += 1

        # 3 j-windows of width 2, each with 2 k-blocks (k0 in {0,2}) -> 3*2 = 6 Strassen blocks * 7 = 42
        for j0 in (0, 2, 4):
            for k0 in (0, 2):
                strassen_block(0, j0, k0)

        # Leftover: k=6 column, all i in {0,1}, j in {0..3} -> 8 terms
        for i in range(2):
            for j in range(4):
                U[i * m + j, col] = 1
                V[j * p + 6, col] = 1
                W[6 * n + i, col] = 1
                col += 1

        assert col == rank, f"expected rank {rank}, got {col}"
        return U, V, W

    # ---- Column removal: remove any column whose removal keeps the residual valid ----
    def try_reduce_columns(U, V, W, max_removals=30):
        r = U.shape[1]
        removed = 0
        changed = True
        while changed and removed < max_removals:
            changed = False
            if time_left() < 8:
                break
            for c in range(r):
                mask = np.ones(r, dtype=bool)
                mask[c] = False
                U_sub = U[:, mask]
                V_sub = V[:, mask]
                W_sub = W[:, mask]
                if verify(U_sub, V_sub, W_sub):
                    U = U_sub
                    V = V_sub
                    W = W_sub
                    r -= 1
                    removed += 1
                    changed = True
                    break
        return U, V, W, removed

    # ---- ALS with incremental residual ----
    def fast_als(U_init, V_init, W_init, target_rank, max_iters=60, tol=1e-10):
        U = U_init.astype(float).copy()
        V = V_init.astype(float).copy()
        W = W_init.astype(float).copy()
        R = residual(U, V, W)
        for _ in range(max_iters):
            if time_left() < 8:
                break
            # Update U
            for c in range(target_rank):
                R -= np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
                vw = V[:, c] * W[:, c]
                denom = np.dot(vw, vw)
                if denom > 1e-12:
                    U[:, c] = np.einsum('abk,bk->a', T + R, vw) / denom
                R += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
            # Update V
            for c in range(target_rank):
                R -= np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
                uw = U[:, c] * W[:, c]
                denom = np.dot(uw, uw)
                if denom > 1e-12:
                    V[:, c] = np.einsum('abk,ak->b', T + R, uw) / denom
                R += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
            # Update W
            for c in range(target_rank):
                R -= np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
                uv = U[:, c] * V[:, c]
                denom = np.dot(uv, uv)
                if denom > 1e-12:
                    W[:, c] = np.einsum('abk,ab->k', T + R, uv) / denom
                R += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
            err = np.linalg.norm(T - R)
            if err < tol:
                break
        return U, V, W, R

    # ---- Build base and verify ----
    U_s, V_s, W_s = strassen_rank50()
    if not verify(U_s, V_s, W_s):
        U_s, V_s, W_s = schoolbook()

    best_U, best_V, best_W = U_s, V_s, W_s

    # First, cheap column removal on the base
    U_red, V_red, W_red, n_removed = try_reduce_columns(best_U, best_V, best_W, max_removals=30)
    if n_removed > 0:
        best_U, best_V, best_W = U_red, V_red, W_red

    # Now attempt to reach lower ranks via ALS from truncated/perturbed starts
    current_rank = best_U.shape[1]
    for target_rank in range(current_rank - 1, max(current_rank - 25, 30), -1):
        if time_left() < 15:
            break
        success = False
        for restart in range(14):
            if time_left() < 15:
                break
            if restart == 0:
                U_init = best_U[:, :target_rank]
                V_init = best_V[:, :target_rank]
                W_init = best_W[:, :target_rank]
            elif restart == 1:
                ns = 0.1
                U_init = best_U[:, :target_rank] + ns * np.random.randn(n * m, target_rank)
                V_init = best_V[:, :target_rank] + ns * np.random.randn(m * p, target_rank)
                W_init = best_W[:, :target_rank] + ns * np.random.randn(p * n, target_rank)
            elif restart == 2:
                ns = 0.5
                U_init = best_U[:, :target_rank] + ns * np.random.randn(n * m, target_rank)
                V_init = best_V[:, :target_rank] + ns * np.random.randn(m * p, target_rank)
                W_init = best_W[:, :target_rank] + ns * np.random.randn(p * n, target_rank)
            elif restart == 3:
                U_init = best_U[:, :target_rank] * 0.8
                V_init = best_V[:, :target_rank] * 0.8
                W_init = best_W[:, :target_rank] * 0.8
            elif restart == 4:
                U_init = best_U[:, :target_rank] * 1.2
                V_init = best_V[:, :target_rank] * 1.2
                W_init = best_W[:, :target_rank] * 1.2
            else:
                idx = np.random.choice(current_rank, target_rank, replace=False)
                U_init = best_U[:, idx]
                V_init = best_V[:, idx]
                W_init = best_W[:, idx]

            U_try, V_try, W_try, R_try = fast_als(U_init, V_init, W_try if False else W_init, target_rank,
                                                  max_iters=70, tol=1e-11)
            if np.linalg.norm(T - R_try) < 1e-8:
                U_try, V_try, W_try, n_rem = try_reduce_columns(U_try, V_try, W_try, max_removals=10)
                best_U, best_V, best_W = U_try, V_try, W_try
                success = True
                break

        if not success:
            # a few fully-random restarts
            for restart in range(6):
                if time_left() < 15:
                    break
                np.random.seed(2000 + restart * 13)
                U_init = np.random.randn(n * m, target_rank)
                V_init = np.random.randn(m * p, target_rank)
                W_init = np.random.randn(p * n, target_rank)
                U_try, V_try, W_try, R_try = fast_als(U_init, V_init, W_init, target_rank,
                                                      max_iters=90, tol=1e-11)
                if np.linalg.norm(T - R_try) < 1e-8:
                    U_try, V_try, W_try, n_rem = try_reduce_columns(U_try, V_try, W_try, max_removals=10)
                    best_U, best_V, best_W = U_try, V_try, W_try
                    success = True
                    break

    # Final exact high-precision polish pass on the best found
    if time_left() > 12:
        r = best_U.shape[1]
        U_pol, V_pol, W_pol, R_pol = fast_als(best_U, best_V, best_W, r, max_iters=200, tol=1e-13)
        if np.linalg.norm(T - R_pol) < 1e-9:
            best_U, best_V, best_W = U_pol, V_pol, W_pol
            best_U, best_V, best_W, _ = try_reduce_columns(best_U, best_V, best_W, max_removals=20)

    # Round to clean values and verify once more; fall back to schoolbook if invalid
    U_out = np.round(best_U, decimals=9)
    V_out = np.round(best_V, decimals=9)
    W_out = np.round(best_W, decimals=9)
    if not verify(U_out, V_out, W_out, atol=1e-6):
        # try without rounding
        if verify(best_U, best_V, best_W, atol=1e-6):
            U_out, V_out, W_out = best_U, best_V, best_W
        else:
            # last resort: schoolbook
            U_out, V_out, W_out = schoolbook()

    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
