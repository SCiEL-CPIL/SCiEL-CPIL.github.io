import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # safety margin

    # Build target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def verify_decomp(U, V, W, atol=1e-9):
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
        """Strassen-based rank-50 decomposition."""
        rank = 50
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0

        def add_strassen(i0, j0, k0):
            nonlocal col
            U[i0 * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            U[(i0 + 1) * m + j0, col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[j0 * p + k0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] -= 1
            col += 1

            U[i0 * m + j0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            U[(i0 + 1) * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + k0, col] += 1
            V[j0 * p + k0, col] -= 1
            W[k0 * n + i0, col] += 1
            W[k0 * n + (i0 + 1), col] += 1
            col += 1

            U[i0 * m + j0, col] += 1
            U[i0 * m + (j0 + 1), col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] -= 1
            W[(k0 + 1) * n + i0, col] += 1
            col += 1

            U[(i0 + 1) * m + j0, col] += 1
            U[i0 * m + j0, col] -= 1
            V[j0 * p + k0, col] += 1
            V[j0 * p + (k0 + 1), col] += 1
            W[(k0 + 1) * n + (i0 + 1), col] += 1
            col += 1

            U[i0 * m + (j0 + 1), col] += 1
            U[(i0 + 1) * m + (j0 + 1), col] -= 1
            V[(j0 + 1) * p + k0, col] += 1
            V[(j0 + 1) * p + (k0 + 1), col] += 1
            W[k0 * n + i0, col] += 1
            col += 1

        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                add_strassen(0, j0, k0)

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

    # Get baseline
    U50, V50, W50 = get_rank50()
    if not verify_decomp(U50, V50, W50):
        U, V, W = get_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_rank = 50
    best_U, best_V, best_W = U50, V50, W50

    # Matricizations for rank lower bounds and ALS
    T1f = T.reshape(n * m, -1)  # 8 x 196
    T2f = T.reshape(m * p, -1)  # 28 x 56
    T3f = T.reshape(p * n, -1)  # 14 x 112

    r1 = np.linalg.matrix_rank(T1f)
    r2 = np.linalg.matrix_rank(T2f)
    r3 = np.linalg.matrix_rank(T3f)
    lower_bound = max(r1, r2, r3)

    def khatri_rao(A, B):
        M, R = A.shape
        N, _ = B.shape
        KR = np.zeros((M * N, R))
        for r in range(R):
            KR[:, r] = np.kron(A[:, r], B[:, r])
        return KR

    def als_polish(U, V, W, max_iters=300, tol=1e-14):
        """High-precision ALS polish."""
        for it in range(max_iters):
            if time.time() > deadline - 2:
                break
            KR_VW = khatri_rao(V, W)
            try:
                U_new = T1f @ np.linalg.pinv(KR_VW)
            except:
                break
            KR_WU = khatri_rao(W, U)
            try:
                V_new = T2f @ np.linalg.pinv(KR_WU)
            except:
                break
            KR_UV = khatri_rao(U, V)
            try:
                W_new = T3f @ np.linalg.pinv(KR_UV)
            except:
                break
            change = (np.linalg.norm(U_new - U) + np.linalg.norm(V_new - V) +
                      np.linalg.norm(W_new - W))
            U, V, W = U_new, V_new, W_new
            if change < tol:
                break
        return U, V, W

    def residual(U, V, W):
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.linalg.norm(recon - T)

    rng = np.random.default_rng(42)

    # Strategy: exploit n=2 structure.
    # The tensor T has the property that for each (j,k) pair, only 2 entries are nonzero:
    # T[0*m+j, j*p+k, k*n+0] = 1 and T[1*m+j, j*p+k, k*n+1] = 1
    # This means the "fiber" along the first mode for fixed (j,k) has support {j, 4+j}
    # and the "fiber" along the third mode for fixed (i,j) has support {j*p, j*p+1} (for i=0)
    # etc.
    
    # Key structural approach: use the fact that n=2 to create "shared" terms.
    # For each j, the contribution to C[0,:] and C[1,:] from B[j,:] is:
    #   C[0,k] += A[0,j] * B[j,k]
    #   C[1,k] += A[1,j] * B[j,k]
    # We can factor B[j,:] as a shared factor.
    
    # Try direct construction with structured patterns:
    # For each j in {0,1,2,3}, we need terms that multiply A[:,j] with B[j,:]
    # A[:,j] is a 2-vector, B[j,:] is a 7-vector.
    # The outer product A[:,j] ⊗ B[j,:] ⊗ (identity in output) gives rank 14 per j if naive.
    # But we can use Strassen-like tricks.

    # Better: try rank reduction via structured search.
    # Use the insight that the problem decomposes into overlapping 2x2x2 blocks.
    
    # Direct approach: try to find rank r by solving the polynomial system
    # using a "support pattern" search combined with linear solves.
    
    def try_rank(target_rank, n_restarts=30):
        """Try to find a valid rank-target_rank decomposition."""
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                return None, None, None, False

            # Generate structured initial guess
            U_init = np.zeros((n * m, target_rank))
            V_init = np.zeros((m * p, target_rank))
            W_init = np.zeros((p * n, target_rank))

            if restart == 0:
                # Truncate rank-50
                U_init = U50[:, :target_rank].copy()
                V_init = V50[:, :target_rank].copy()
                W_init = W50[:, :target_rank].copy()
            elif restart == 1:
                # Random
                U_init = rng.normal(0, 1, (n * m, target_rank))
                V_init = rng.normal(0, 1, (m * p, target_rank))
                W_init = rng.normal(0, 1, (p * n, target_rank))
            elif restart == 2:
                # Structured: assign each term to a (j,k) window
                for r in range(target_rank):
                    j = r % m
                    k = (r // m) % p
                    i0 = (r // (m * p)) % n
                    U_init[i0 * m + j, r] = 1.0
                    V_init[j * p + k, r] = 1.0
                    W_init[k * n + i0, r] = 1.0
                # Add noise
                U_init += rng.normal(0, 0.1, (n * m, target_rank))
                V_init += rng.normal(0, 0.1, (m * p, target_rank))
                W_init += rng.normal(0, 0.1, (p * n, target_rank))
            elif restart == 3:
                # Strassen-like structured
                for r in range(target_rank):
                    i0 = r % n
                    j0 = (r // n) % m
                    k0 = (r // (n * m)) % p
                    # 2x2x2 Strassen pattern
                    for di in range(n):
                        for dj in range(2):
                            for dk in range(2):
                                if j0 + dj < m and k0 + dk < p:
                                    sign = (-1) ** (di + dj + dk)
                                    U_init[(i0+di)*m + j0+dj, r] += sign
                                    V_init[(j0+dj)*p + k0+dk, r] += sign
                                    W_init[(k0+dk)*n + i0+di, r] += sign
                U_init += rng.normal(0, 0.05, (n * m, target_rank))
                V_init += rng.normal(0, 0.05, (m * p, target_rank))
                W_init += rng.normal(0, 0.05, (p * n, target_rank))
            elif restart == 4:
                # Blend
                n_copy = min(target_rank, 50)
                U_init[:, :n_copy] = U50[:, :n_copy]
                V_init[:, :n_copy] = V50[:, :n_copy]
                W_init[:, :n_copy] = W50[:, :n_copy]
                if target_rank > n_copy:
                    U_init[:, n_copy:] = rng.normal(0, 1, (n * m, target_rank - n_copy))
                    V_init[:, n_copy:] = rng.normal(0, 1, (m * p, target_rank - n_copy))
                    W_init[:, n_copy:] = rng.normal(0, 1, (p * n, target_rank - n_copy))
            elif restart == 5:
                # Sparse binary
                U_init = rng.choice([-1, 0, 1], (n * m, target_rank)).astype(float)
                V_init = rng.choice([-1, 0, 1], (m * p, target_rank)).astype(float)
                W_init = rng.choice([-1, 0, 1], (p * n, target_rank)).astype(float)
                for r in range(target_rank):
                    if np.all(U_init[:, r] == 0):
                        U_init[rng.integers(0, n*m), r] = 1
                    if np.all(V_init[:, r] == 0):
                        V_init[rng.integers(0, m*p), r] = 1
                    if np.all(W_init[:, r] == 0):
                        W_init[rng.integers(0, p*n), r] = 1
            elif restart == 6:
                # SVD-based
                try:
                    U_svd, _, _ = np.linalg.svd(T1f, full_matrices=False)
                    U_init = U_svd[:, :min(target_rank, U_svd.shape[1])]
                    if U_init.shape[1] < target_rank:
                        U_init = np.hstack([U_init, rng.normal(0, 0.1, (n*m, target_rank - U_init.shape[1]))])
                except:
                    U_init = rng.normal(0, 1, (n * m, target_rank))
                V_init = rng.normal(0, 1, (m * p, target_rank))
                W_init = rng.normal(0, 1, (p * n, target_rank))
            elif restart == 7:
                # Geometric mean scaling
                s = target_rank ** (-0.25)
                U_init = rng.normal(0, s, (n * m, target_rank))
                V_init = rng.normal(0, 1.0, (m * p, target_rank))
                W_init = rng.normal(0, 1.0/s, (p * n, target_rank))
            elif restart == 8:
                # Perturbed rank-50
                U_init = U50[:, :target_rank].copy() + rng.normal(0, 0.05, (n*m, target_rank))
                V_init = V50[:, :target_rank].copy() + rng.normal(0, 0.05, (m*p, target_rank))
                W_init = W50[:, :target_rank].copy() + rng.normal(0, 0.05, (p*n, target_rank))
            elif restart == 9:
                # Two-scale: half from rank-50, half random
                half = target_rank // 2
                U_init[:, :half] = U50[:, :half]
                V_init[:, :half] = V50[:, :half]
                W_init[:, :half] = W50[:, :half]
                U_init[:, half:] = rng.normal(0, 1, (n * m, target_rank - half))
                V_init[:, half:] = rng.normal(0, 1, (m * p, target_rank - half))
                W_init[:, half:] = rng.normal(0, 1, (p * n, target_rank - half))
            else:
                # Varying perturbation
                noise = 0.01 * (restart - 9)
                U_init = U50[:, :target_rank].copy() + rng.normal(0, max(noise, 0.001), (n*m, target_rank))
                V_init = V50[:, :target_rank].copy() + rng.normal(0, max(noise, 0.001), (m*p, target_rank))
                W_init = W50[:, :target_rank].copy() + rng.normal(0, max(noise, 0.001), (p*n, target_rank))

            # Run ALS
            U_sol, V_sol, W_sol = als_polish(U_init, V_init, W_init, max_iters=200)
            res = residual(U_sol, V_sol, W_sol)

            if res < 1e-6:
                U_sol, V_sol, W_sol = als_polish(U_sol, V_sol, W_sol, max_iters=200)
                if verify_decomp(U_sol, V_sol, W_sol):
                    return U_sol, V_sol, W_sol, True

        return None, None, None, False

    # Try ranks from 49 down to lower_bound
    min_try = max(lower_bound, 30)
    for target_rank in range(49, min_try - 1, -1):
        if time.time() > deadline - 10:
            break

        time_left = deadline - 8 - time.time()
        if time_left < 5:
            break

        n_restarts = max(10, min(40, int(time_left / 1.0)))
        U_sol, V_sol, W_sol, success = try_rank(target_rank, n_restarts)

        if success:
            best_rank = target_rank
            best_U, best_V, best_W = U_sol, V_sol, W_sol
            # Continue trying lower ranks if time permits
            if time.time() < deadline - 15:
                continue
            else:
                break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
