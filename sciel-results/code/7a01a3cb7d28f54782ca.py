import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115

    # Build the target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify_decomp(U, V, W, atol=1e-8):
        """Check if sum_r U[:,r] V[:,r] W[:,r] == T"""
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(recon, T, atol=atol)

    def get_schoolbook():
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

    def get_rank50():
        """Construct a known valid rank-50 decomposition using Strassen blocks."""
        rank = 50
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0

        def add_strassen(i0, j0, k0):
            nonlocal col
            # M1: (a+d)(e+h)
            U[i0 * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M2: (c+d)e
            U[(i0 + 1) * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] -= 1
            col += 1

            # M3: a(f-h)
            U[i0 * m + j0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M4: d(g-e)
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + k0, col] += 1
            V[j0 * p + k0, col] -= 1
            W[k0 * n + i0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            col += 1

            # M5: (a+b)h
            U[i0 * m + j0, col] += 1
            U[i0 * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            col += 1

            # M6: (c-a)(e+f)
            U[(i0 + 1) * m + j0, col] += 1
            U[i0 * m + j0, col] -= 1
            V[j0 * p + k0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            # M7: (b-d)(g+h)
            U[i0 * m + (j0 + 1), col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] -= 1
            V[(j0 + 1) * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            col += 1

        # Tile 2x4 x 4x7:
        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                add_strassen(0, j0, k0)

        # Handle column 6 of B (and C) separately with rank-1 terms
        for j in range(m):
            U[0 * m + j, col] = 1
            V[j * p + 6, col] = 1
            W[6 * n + 0, col] = 1
            col += 1
            U[1 * m + j, col] = 1
            V[j * p + 6, col] = 1
            W[6 * n + 1, col] = 1
            col += 1

        assert col == rank, f"Expected rank 50, got {col}"
        return U, V, W

    def khatri_rao(A, B):
        """Compute Khatri-Rao product of A (M x R) and B (N x R)."""
        M, R = A.shape
        N, _ = B.shape
        KR = np.zeros((M * N, R))
        for r in range(R):
            KR[:, r] = np.kron(A[:, r], B[:, r])
        return KR

    def als_decompose(T1, T2, T3, U_init, V_init, W_init, max_iters=150, tol=1e-12):
        """Alternating Least Squares for CP decomposition with high-precision polish."""
        R = U_init.shape[1]
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()

        for it in range(max_iters):
            # Solve for U
            KR_VW = khatri_rao(V, W)
            try:
                U_new = T1 @ np.linalg.pinv(KR_VW)
            except:
                break

            # Solve for V
            KR_WU = khatri_rao(W, U_new)
            try:
                V_new = T2 @ np.linalg.pinv(KR_WU)
            except:
                break

            # Solve for W
            KR_UV = khatri_rao(U_new, V_new)
            try:
                W_new = T3 @ np.linalg.pinv(KR_UV)
            except:
                break

            # Check convergence
            change = (np.linalg.norm(U_new - U) + np.linalg.norm(V_new - V) +
                      np.linalg.norm(W_new - W))
            U, V, W = U_new, V_new, W_new

            if change < tol:
                break

        return U, V, W

    def try_rank_reduction(U50, V50, W50, T1, T2, T3, target_rank, rng, max_restarts=8):
        """Try to find a decomposition of target_rank using ALS with smart init."""
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                return None

            if restart == 0 and target_rank == 49:
                # Drop one tail column (columns 42-47 are the tail rank-1 terms)
                drop_col = rng.integers(42, 50)
                keep = [c for c in range(50) if c != drop_col]
                U_init = U50[:, keep].copy()
                V_init = V50[:, keep].copy()
                W_init = W50[:, keep].copy()
                # Add small noise to break degeneracy
                noise_scale = 0.05
                U_init += rng.normal(0, noise_scale, U_init.shape)
                V_init += rng.normal(0, noise_scale, V_init.shape)
                W_init += rng.normal(0, noise_scale, W_init.shape)
            elif restart == 1 and target_rank == 49:
                # Drop a Strassen column instead
                drop_col = rng.integers(0, 42)
                keep = [c for c in range(50) if c != drop_col]
                U_init = U50[:, keep].copy()
                V_init = V50[:, keep].copy()
                W_init = W50[:, keep].copy()
                noise_scale = 0.1
                U_init += rng.normal(0, noise_scale, U_init.shape)
                V_init += rng.normal(0, noise_scale, V_init.shape)
                W_init += rng.normal(0, noise_scale, W_init.shape)
            else:
                # Random initialization
                U_init = rng.normal(0, 1, (n * m, target_rank))
                V_init = rng.normal(0, 1, (m * p, target_rank))
                W_init = rng.normal(0, 1, (p * n, target_rank))

            U_sol, V_sol, W_sol = als_decompose(T1, T2, T3, U_init, V_init, W_init,
                                                max_iters=200, tol=1e-12)

            if verify_decomp(U_sol, V_sol, W_sol, atol=1e-8):
                return (U_sol, V_sol, W_sol)

        return None

    def border_rank_search(U50, V50, W50, T1, T2, T3, rng):
        """
        Border rank / limit method: construct a family of rank-49 decompositions
        for a perturbed tensor, then take the limit as perturbation -> 0.
        """
        for tail_col in range(42, 50):
            if time.time() > deadline - 10:
                break

            # Remove this tail column
            keep = [c for c in range(50) if c != tail_col]
            U_base = U50[:, keep].copy()
            V_base = V50[:, keep].copy()
            W_base = W50[:, keep].copy()

            U_tail = U50[:, tail_col:tail_col+1]
            V_tail = V50[:, tail_col:tail_col+1]
            W_tail = W50[:, tail_col:tail_col+1]

            for absorb_idx in range(min(42, len(keep))):
                if time.time() > deadline - 10:
                    break

                U_init = U_base.copy()
                V_init = V_base.copy()
                W_init = W_base.copy()

                perturbation = rng.normal(0, 0.3, (n * m,))
                U_init[:, absorb_idx] += perturbation
                perturbation = rng.normal(0, 0.3, (m * p,))
                V_init[:, absorb_idx] += perturbation
                perturbation = rng.normal(0, 0.3, (p * n,))
                W_init[:, absorb_idx] += perturbation

                U_sol, V_sol, W_sol = als_decompose(T1, T2, T3, U_init, V_init, W_init,
                                                    max_iters=200, tol=1e-12)

                if verify_decomp(U_sol, V_sol, W_sol, atol=1e-8):
                    return (U_sol, V_sol, W_sol)

        return None

    # Get the rank-50 decomposition
    U50, V50, W50 = get_rank50()

    # Verify it's correct
    if not verify_decomp(U50, V50, W50):
        U, V, W = get_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_rank = 50
    best_U, best_V, best_W = U50.copy(), V50.copy(), W50.copy()

    # Precompute unfoldings of T
    T1 = T.reshape(n * m, -1)  # 8 x 392
    T2 = T.reshape(m * p, -1)  # 28 x 112
    T3 = T.reshape(p * n, -1)  # 14 x 224

    rng = np.random.default_rng(42)

    # Try border rank search for rank 49
    if time.time() < deadline - 30:
        result = border_rank_search(U50, V50, W50, T1, T2, T3, rng)
        if result is not None:
            best_rank = 49
            best_U, best_V, best_W = result

    # Try standard rank reduction for rank 49
    if best_rank > 49 and time.time() < deadline - 15:
        result = try_rank_reduction(U50, V50, W50, T1, T2, T3, 49, rng, max_restarts=10)
        if result is not None:
            best_rank = 49
            best_U, best_V, best_W = result

    # Try rank 48
    if best_rank > 48 and time.time() < deadline - 10:
        result = try_rank_reduction(U50, V50, W50, T1, T2, T3, 48, rng, max_restarts=5)
        if result is not None:
            best_rank = 48
            best_U, best_V, best_W = result

    # High-precision polish on the best solution found
    if best_rank < 50 and time.time() < deadline - 5:
        T1, T2, T3 = T.reshape(n * m, -1), T.reshape(m * p, -1), T.reshape(p * n, -1)
        U_polished, V_polished, W_polished = als_decompose(
            T1, T2, T3, best_U, best_V, best_W, max_iters=300, tol=1e-14)
        if verify_decomp(U_polished, V_polished, W_polished, atol=1e-10):
            best_U, best_V, best_W = U_polished, V_polished, W_polished

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
