def construct():
    import time
    import numpy as np
    _t0 = time.time()
    _DEADLINE = 130.0  # safety margin before 150s kill

    n, m, p = 3, 4, 6
    # Tensor dimensions: A is n*m = 12, B is m*p = 24, C is p*n = 18
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(i,k) for some i,j,k
    # Indexing: a = i*m + j, b = j*p + k, c = k*n + i

    def build_tensor():
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T

    T = build_tensor()

    def build_strassen_decomposition():
        """Build the rank-66 Strassen+schoolbook decomposition."""
        terms = []

        def add_strassen_block(i1, i2, j1, j2, k1, k2):
            a = (i1, j1)
            b = (i1, j2)
            c = (i2, j1)
            d = (i2, j2)
            e = (j1, k1)
            f = (j1, k2)
            g = (j2, k1)
            h = (j2, k2)
            C11 = (i1, k1)
            C12 = (i1, k2)
            C21 = (i2, k1)
            C22 = (i2, k2)

            terms.append(({a: 1, d: 1}, {e: 1, h: 1}, {C11: 1, C22: 1}))
            terms.append(({c: 1, d: 1}, {e: 1}, {C21: 1, C22: -1}))
            terms.append(({a: 1}, {f: 1, h: -1}, {C12: 1, C22: 1}))
            terms.append(({d: 1}, {g: 1, e: -1}, {C11: 1, C21: 1}))
            terms.append(({a: 1, b: 1}, {h: 1}, {C11: -1, C12: 1}))
            terms.append(({c: 1, a: -1}, {e: 1, f: 1}, {C22: 1}))
            terms.append(({b: 1, d: -1}, {g: 1, h: 1}, {C11: 1}))

        def add_schoolbook_term(i, j, k):
            terms.append((
                {(i, j): 1},
                {(j, k): 1},
                {(i, k): 1}
            ))

        # Row 0: schoolbook (12 terms)
        for j in range(m):
            for k in range(p):
                add_schoolbook_term(0, j, k)

        # Rows 1-2: Strassen blocks
        for (k1, k2) in [(0, 1), (2, 3), (4, 5)]:
            add_strassen_block(1, 2, 0, 1, k1, k2)
            add_strassen_block(1, 2, 2, 3, k1, k2)

        # Convert to matrices
        rank = len(terms)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        for col, (u_dict, v_dict, w_dict) in enumerate(terms):
            for (i, j), coeff in u_dict.items():
                U[i * m + j, col] = coeff
            for (j, k), coeff in v_dict.items():
                V[j * p + k, col] = coeff
            for (i, k), coeff in w_dict.items():
                W[k * n + i, col] = coeff

        return U, V, W

    U0, V0, W0 = build_strassen_decomposition()
    best_rank = U0.shape[1]
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()

    # Verify the decomposition is correct
    def verify(U, V, W, T, tol=1e-10):
        """Check if sum_r U[:,r] outer V[:,r] outer W[:,r] == T"""
        # Efficient check: for each r, add U_r ⊗ V_r ⊗ W_r
        # Reshape T to (n*m, m*p, p*n)
        recon = np.zeros_like(T)
        for r in range(U.shape[1]):
            recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.max(np.abs(recon - T)) < tol

    if not verify(U0, V0, W0, T):
        # Fallback to schoolbook
        U0 = np.zeros((n * m, n * m * p))
        V0 = np.zeros((m * p, n * m * p))
        W0 = np.zeros((p * n, n * m * p))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U0[i * m + j, idx] = 1
                    V0[j * p + k, idx] = 1
                    W0[k * n + i, idx] = 1
                    idx += 1
        best_rank = n * m * p
        best_U, best_V, best_W = U0, V0, W0

    # Strategy 1: Iterative rank-1 peeling (de Zeeuw style)
    # Start from the rank-66 decomposition and try to find linear dependencies
    # that allow reducing the rank.

    def try_peeling(U, V, W, T, max_iters=50):
        """
        Try to reduce rank by finding rank-1 components in the residual
        and re-optimizing.
        """
        r = U.shape[1]
        if r <= 1:
            return U, V, W, r

        # Flatten T
        T_flat = T.reshape(-1)
        dim_a, dim_b, dim_c = T.shape

        # Try to find a rank-1 component that best fits T
        # Using power iteration on the tensor
        # T as a 3-tensor: we want to find u, v, w such that
        # <T, u⊗v⊗w> is maximized, then subtract

        # Power iteration for best rank-1 approximation
        u = np.random.randn(dim_a)
        v = np.random.randn(dim_b)
        w = np.random.randn(dim_c)
        u /= np.linalg.norm(u)
        v /= np.linalg.norm(v)
        w /= np.linalg.norm(w)

        for it in range(max_iters):
            # Update u: T contracted with v, w
            u_new = np.einsum('abc,b,c->a', T, v, w)
            if np.linalg.norm(u_new) < 1e-15:
                break
            u_new /= np.linalg.norm(u_new)
            # Update v
            v_new = np.einsum('abc,a,c->b', T, u_new, w)
            if np.linalg.norm(v_new) < 1e-15:
                break
            v_new /= np.linalg.norm(v_new)
            # Update w
            w_new = np.einsum('abc,a,b->c', T, u, v_new)
            if np.linalg.norm(w_new) < 1e-15:
                break
            w_new /= np.linalg.norm(w_new)

            u, v, w = u_new, v_new, w_new

        # Now try to use this as a component
        # But this is the best rank-1 approx to T, not to the residual
        # Better approach: use alternating minimization on the full problem

        return U, V, W, r

    # Strategy 2: Direct alternating minimization for rank r
    def try_alternating_min(T, r, n_iter=200, n_restarts=5, seed=42):
        """
        Alternating minimization to find rank-r decomposition of T.
        T[a,b,c] = sum_i U[a,i] * V[b,i] * W[c,i]
        """
        dim_a, dim_b, dim_c = T.shape
        rng = np.random.RandomState(seed)
        T_flat = T.reshape(-1)

        best_err = np.inf
        best_U, best_V, best_W = None, None, None

        for restart in range(n_restarts):
            # Initialize
            U = rng.randn(dim_a, r)
            V = rng.randn(dim_b, r)
            W = rng.randn(dim_c, r)

            # Normalize to avoid scale issues
            for i in range(r):
                norm_u = np.linalg.norm(U[:, i])
                norm_v = np.linalg.norm(V[:, i])
                norm_w = np.linalg.norm(W[:, i])
                if norm_u > 1e-15 and norm_v > 1e-15 and norm_w > 1e-15:
                    # Redistribute scale
                    total_norm = (norm_u * norm_v * norm_w) ** (1.0 / 3.0)
                    U[:, i] /= norm_u
                    V[:, i] /= norm_v
                    W[:, i] /= norm_w
                    # Scale one of them
                    U[:, i] *= total_norm

            err = np.inf
            for it in range(n_iter):
                # Update U: for each column i, U[:,i] = (T contracted with V[:,i] and W[:,i]) / (||V[:,i]||^2 * ||W[:,i]||^2)
                for i in range(r):
                    # Contract T with V[:,i] and W[:,i]
                    tvw = np.einsum('abc,b,c->a', T, V[:, i], W[:, i])
                    vv = np.dot(V[:, i], V[:, i])
                    ww = np.dot(W[:, i], W[:, i])
                    if vv * ww > 1e-15:
                        U[:, i] = tvw / (vv * ww)

                # Update V
                for i in range(r):
                    tvw = np.einsum('abc,a,c->b', T, U[:, i], W[:, i])
                    uu = np.dot(U[:, i], U[:, i])
                    ww = np.dot(W[:, i], W[:, i])
                    if uu * ww > 1e-15:
                        V[:, i] = tvw / (uu * ww)

                # Update W
                for i in range(r):
                    tvw = np.einsum('abc,a,b->c', T, U[:, i], V[:, i])
                    uu = np.dot(U[:, i], U[:, i])
                    vv = np.dot(V[:, i], V[:, i])
                    if uu * vv > 1e-15:
                        W[:, i] = tvw / (uu * vv)

                # Compute error
                recon = np.zeros_like(T)
                for i in range(r):
                    recon += np.einsum('a,b,c->abc', U[:, i], V[:, i], W[:, i])
                err = np.linalg.norm(recon - T) / np.linalg.norm(T)

                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

                if err < 1e-10:
                    break

                # Time check
                if time.time() - _t0 > _DEADLINE - 10:
                    return best_U, best_V, best_W, r, best_err

        return best_U, best_V, best_W, r, best_err

    # Strategy 3: Use the existing decomposition as initialization and try to reduce rank
    # by finding linear dependencies among the columns

    def try_linear_reduction(U, V, W, T, target_rank):
        """
        Try to express the tensor as a sum of target_rank rank-1 terms
        by solving a linear system for the coefficients, then re-optimizing.
        """
        r = U.shape[1]
        if target_rank >= r:
            return U, V, W, r

        # The current decomposition: T = sum_{i=0}^{r-1} U[:,i] ⊗ V[:,i] ⊗ W[:,i]
        # We want to find new U', V', W' with target_rank columns.
        # This is nonlinear, so use alternating minimization initialized from the current solution.

        dim_a, dim_b, dim_c = T.shape
        rng = np.random.RandomState(123)

        # Initialize new decomposition from the old one
        # Take a random subset or use SVD-like approach
        U_new = np.zeros((dim_a, target_rank))
        V_new = np.zeros((dim_b, target_rank))
        W_new = np.zeros((dim_c, target_rank))

        # Simple init: take first target_rank columns and add noise
        for i in range(target_rank):
            U_new[:, i] = U[:, i] + 0.1 * rng.randn(dim_a)
            V_new[:, i] = V[:, i] + 0.1 * rng.randn(dim_b)
            W_new[:, i] = W[:, i] + 0.1 * rng.randn(dim_c)

        # Alternating minimization
        n_iter = 300
        for it in range(n_iter):
            # Update U
            for i in range(target_rank):
                tvw = np.einsum('abc,b,c->a', T, V_new[:, i], W_new[:, i])
                vv = np.dot(V_new[:, i], V_new[:, i])
                ww = np.dot(W_new[:, i], W_new[:, i])
                if vv * ww > 1e-15:
                    U_new[:, i] = tvw / (vv * ww)

            # Update V
            for i in range(target_rank):
                tvw = np.einsum('abc,a,c->b', T, U_new[:, i], W_new[:, i])
                uu = np.dot(U_new[:, i], U_new[:, i])
                ww = np.dot(W_new[:, i], W_new[:, i])
                if uu * ww > 1e-15:
                    V_new[:, i] = tvw / (uu * ww)

            # Update W
            for i in range(target_rank):
                tvw = np.einsum('abc,a,b->c', T, U_new[:, i], V_new[:, i])
                uu = np.dot(U_new[:, i], U_new[:, i])
                vv = np.dot(V_new[:, i], V_new[:, i])
                if uu * vv > 1e-15:
                    W_new[:, i] = tvw / (uu * vv)

            # Compute error
            recon = np.zeros_like(T)
            for i in range(target_rank):
                recon += np.einsum('a,b,c->abc', U_new[:, i], V_new[:, i], W_new[:, i])
            err = np.linalg.norm(recon - T) / np.linalg.norm(T)

            if time.time() - _t0 > _DEADLINE - 5:
                break

        return U_new, V_new, W_new, target_rank, err

    # Try various rank reductions
    # The Strassen decomposition has rank 66. Try to find lower ranks.
    # Known bounds: for 3x4x6, rank is at most 66, and conjectured to be around 55-60.

    # First, try a few ranks below 66
    for target_rank in [65, 64, 63, 62, 60, 58, 56, 54, 52, 50]:
        if time.time() - _t0 > _DEADLINE - 10:
            break

        # Use alternating minimization with the Strassen init
        U_try, V_try, W_try, r_try, err = try_linear_reduction(U0, V0, W0, T, target_rank)

        if err < 1e-8:
            # Found a valid decomposition!
            # Try to round to rational/half-integer values
            best_rank = r_try
            best_U, best_V, best_W = U_try, V_try, W_try
            # Continue trying lower ranks
            continue
        else:
            # Not converged, but keep the best found so far
            if err < 0.01:
                # Close but not exact, try with more iterations
                pass
            break  # If we can't even get close to rank 65, stop

    # If we found a valid decomposition, try to make it exact (rational)
    if best_rank < 66:
        # Try to round to half-integers
        def round_to_half_integer(M):
            """Round matrix entries to nearest half-integer."""
            M_rounded = np.round(M * 2) / 2.0
            return M_rounded

        U_rounded = round_to_half_integer(best_U)
        V_rounded = round_to_half_integer(best_V)
        W_rounded = round_to_half_integer(best_W)

        # Verify
        if verify(U_rounded, V_rounded, W_rounded, T, tol=1e-10):
            best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
        else:
            # Keep the floating point version
            pass

    # Final verification
    if not verify(best_U, best_V, best_W, T, tol=1e-8):
        # Fallback to the original Strassen decomposition
        best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
        best_rank = U0.shape[1]

    # Convert to lists for the return interface
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()

    # Convert to proper format: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # The interface expects: U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    # But the return format is {'U', 'V', 'W'} with these shapes

    # Ensure we return the correct shapes
    assert len(U_list) == n * m, f"U has {len(U_list)} rows, expected {n*m}"
    assert len(V_list) == m * p, f"V has {len(V_list)} rows, expected {m*p}"
    assert len(W_list) == p * n, f"W has {len(W_list)} rows, expected {p*n}"

    return {"U": U_list, "V": V_list, "W": W_list}
