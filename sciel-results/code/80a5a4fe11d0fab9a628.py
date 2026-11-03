"""Combined: structural Strassen construction + ALS refinement + factor-merging search."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 135.0  # 15s safety margin before 150s kill
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Precompute the 60 non-zero tensor entries: (U_idx, V_idx, W_idx)
    entries = [(i * m + j, j * p + k, k * n + i) for i in range(n) for j in range(m) for k in range(p)]
    ENTRIES = entries  # capture for closures
    _entry_set = set(ENTRIES)

    def elapsed():
        return time.time() - start_time

    def verify_fast(U, V, W, r, tol=1e-9):
        """Fast verification: check the 60 non-zero entries only."""
        for (ui, vi, wi) in ENTRIES:
            val = np.dot(U[ui, :r], V[vi, :r] * W[wi, :r])
            if abs(val - 1.0) > tol:
                return False
        # Check all other entries are ~0
        for i in range(nm):
            for j in range(mp):
                for k in range(pn):
                    if (i, j, k) in _entry_set:
                        continue
                    val = np.dot(U[i, :r], V[j, :r] * W[k, :r])
                    if abs(val) > tol:
                        return False
        return True

    def verify_error(U, V, W, r, tol=1e-10):
        """Return max error over all entries."""
        max_err = 0.0
        for (ui, vi, wi) in ENTRIES:
            val = np.dot(U[ui, :r], V[vi, :r] * W[wi, :r])
            err = abs(val - 1.0)
            if err > max_err:
                max_err = err
        for i in range(nm):
            for j in range(mp):
                for k in range(pn):
                    if (i, j, k) in _entry_set:
                        continue
                    val = np.dot(U[i, :r], V[j, :r] * W[k, :r])
                    err = abs(val)
                    if err > max_err:
                        max_err = err
        return max_err

    def build_rank55():
        """Rank-55 Strassen decomposition: j=0 trivial (20) + 5 Strassen blocks (7 each)."""
        rank = 55
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0

        # Part A: j=0 trivial (2 * 10 = 20 factors)
        for i in range(n):
            for k in range(p):
                U[i * m + 0, col] = 1
                V[0 * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

        # Part B: j=1,2 with Strassen on 5 blocks of (k0, k1)
        for b in range(5):
            k0 = 2 * b
            k1 = 2 * b + 1

            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1

            u_11 = 0 * m + 1
            u_12 = 0 * m + 2
            u_21 = 1 * m + 1
            u_22 = 1 * m + 2

            v_11 = 1 * p + k0
            v_12 = 1 * p + k1
            v_21 = 2 * p + k0
            v_22 = 2 * p + k1

            # M1 = (a11 + a22)(b11 + b22)
            U[u_11, col] += 1; U[u_22, col] += 1
            V[v_11, col] += 1; V[v_22, col] += 1
            W[w_00, col] += 1; W[w_11, col] += 1
            col += 1
            # M2 = (a21 + a22)b11
            U[u_21, col] += 1; U[u_22, col] += 1
            V[v_11, col] += 1
            W[w_10, col] += 1; W[w_11, col] += -1
            col += 1
            # M3 = a11(b12 - b22)
            U[u_11, col] += 1
            V[v_12, col] += 1; V[v_22, col] += -1
            W[w_01, col] += 1; W[w_11, col] += 1
            col += 1
            # M4 = a22(b21 - b11)
            U[u_22, col] += 1
            V[v_21, col] += 1; V[v_11, col] += -1
            W[w_00, col] += 1; W[w_10, col] += 1
            col += 1
            # M5 = (a11 + a12)b22
            U[u_11, col] += 1; U[u_12, col] += 1
            V[v_22, col] += 1
            W[w_00, col] += -1; W[w_01, col] += 1
            col += 1
            # M6 = (a21 - a11)(b11 + b12)
            U[u_21, col] += 1; U[u_11, col] += -1
            V[v_11, col] += 1; V[v_12, col] += 1
            W[w_11, col] += 1
            col += 1
            # M7 = (a12 - a22)(b21 + b22)
            U[u_12, col] += 1; U[u_22, col] += -1
            V[v_21, col] += 1; V[v_22, col] += 1
            W[w_00, col] += 1
            col += 1

        assert col == rank
        return U, V, W, rank

    def build_schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for (ui, vi, wi) in ENTRIES:
            U[ui, col] = 1
            V[vi, col] = 1
            W[wi, col] = 1
            col += 1
        return U, V, W, rank

    def remove_redundant_factors(U, V, W, r, max_passes=3):
        """Iteratively remove factors whose tensor contribution is zero (redundant)."""
        for _ in range(max_passes):
            if elapsed() > DEADLINE:
                break
            changed = False
            for c in range(r - 1, -1, -1):
                if elapsed() > DEADLINE:
                    break
                uc = U[:, c]; vc = V[:, c]; wc = W[:, c]
                max_contrib = 0.0
                for (ui, vi, wi) in ENTRIES:
                    contrib = uc[ui] * vc[vi] * wc[wi]
                    if abs(contrib) > max_contrib:
                        max_contrib = abs(contrib)
                if max_contrib < 1e-12:
                    mask = [i for i in range(r) if i != c]
                    U = U[:, mask]
                    V = V[:, mask]
                    W = W[:, mask]
                    r = len(mask)
                    changed = True
                    break
            if not changed:
                break
        return U, V, W, r

    def als_refine(U, V, W, r, target_rank, max_restarts=10, max_iters=20, tol=1e-10):
        """ALS refinement to try to find a lower-rank decomposition."""
        if elapsed() > DEADLINE:
            return U, V, W, r

        # Build target tensor
        T = np.zeros((nm, mp, pn))
        for (ui, vi, wi) in ENTRIES:
            T[ui, vi, wi] = 1.0

        best = (U, V, W, r)
        best_err = verify_error(U, V, W, r)

        for trial in range(max_restarts):
            if elapsed() > DEADLINE:
                break

            np.random.seed(trial * 1000 + target_rank)

            # Initialize near the current solution
            U_c = U[:, :target_rank].copy() + 0.1 * np.random.randn(nm, target_rank)
            V_c = V[:, :target_rank].copy() + 0.1 * np.random.randn(mp, target_rank)
            W_c = W[:, :target_rank].copy() + 0.1 * np.random.randn(pn, target_rank)

            for it in range(max_iters):
                if elapsed() > DEADLINE:
                    break

                # Solve for U
                M = np.einsum('jr,kr->jkr', V_c, W_c).reshape(mp * pn, target_rank)
                try:
                    U_c = T.reshape(nm, mp * pn) @ np.linalg.pinv(M.T)
                except:
                    break

                # Solve for V
                N = np.einsum('ir,kr->ikr', U_c, W_c).reshape(nm * pn, target_rank)
                try:
                    V_c = T.reshape(mp, nm * pn) @ np.linalg.pinv(N.T)
                except:
                    break

                # Solve for W
                P = np.einsum('ir,jr->ijr', U_c, V_c).reshape(nm * mp, target_rank)
                try:
                    W_c = T.reshape(pn, nm * mp) @ np.linalg.pinv(P.T)
                except:
                    break

            # Check convergence
            T_check = np.einsum('ir,jr,kr->ijk', U_c, V_c, W_c)
            error = np.max(np.abs(T_check - T))

            if error < 1e-8:
                # Try to round to half-integers
                U_r = np.round(U_c * 2) / 2
                V_r = np.round(V_c * 2) / 2
                W_r = np.round(W_c * 2) / 2

                T_check2 = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
                if np.max(np.abs(T_check2 - T)) < 1e-10:
                    best = (U_r, V_r, W_r, target_rank)
                    best_err = 0.0
                    break
                else:
                    # Use unrounded if close enough
                    if error < 1e-10:
                        best = (U_c, V_c, W_c, target_rank)
                        best_err = error
                        break

        return best

    def try_merge_factors(U, V, W, r, max_attempts=50):
        """Try to merge pairs of factors by solving for a combined factor."""
        if elapsed() > DEADLINE:
            return U, V, W, r

        best = (U, V, W, r)
        best_err = verify_error(U, V, W, r)

        for attempt in range(max_attempts):
            if elapsed() > DEADLINE:
                break

            # Pick two random factors to merge
            c1, c2 = np.random.choice(r, size=2, replace=False)
            if c1 == c2:
                continue

            # Try to find a single factor that approximates both
            # This is a heuristic: we try to find U', V', W' such that
            # U'[i] * V'[j] * W'[k] ≈ U[i,c1]*V[j,c1]*W[k,c1] + U[i,c2]*V[j,c2]*W[k,c2]
            # This is generally not possible with a single rank-1 term, but we can try
            # to find the best approximation via ALS.

            # Create a target tensor for the sum of these two factors
            T_target = np.zeros((nm, mp, pn))
            for (ui, vi, wi) in ENTRIES:
                T_target[ui, vi, wi] = U[ui, c1] * V[vi, c1] * W[wi, c1] + U[ui, c2] * V[vi, c2] * W[wi, c2]

            # Try to find a rank-1 approximation
            # Use ALS with 1 iteration
            U_c = (U[:, c1] + U[:, c2]) / 2 + 0.1 * np.random.randn(nm)
            V_c = (V[:, c1] + V[:, c2]) / 2 + 0.1 * np.random.randn(mp)
            W_c = (W[:, c1] + W[:, c2]) / 2 + 0.1 * np.random.randn(pn)

            for it in range(5):
                # Solve for U
                M = np.einsum('j,k->jk', V_c, W_c).reshape(mp * pn)
                if np.sum(M**2) < 1e-12:
                    break
                U_c = T_target.reshape(nm, mp * pn) / M[np.newaxis, :]

                # Solve for V
                N = np.einsum('i,k->ik', U_c, W_c).reshape(nm * pn)
                if np.sum(N**2) < 1e-12:
                    break
                V_c = T_target.reshape(mp, nm * pn) / N[np.newaxis, :]

                # Solve for W
                P = np.einsum('i,j->ij', U_c, V_c).reshape(nm * mp)
                if np.sum(P**2) < 1e-12:
                    break
                W_c = T_target.reshape(pn, nm * mp) / P[np.newaxis, :]

            # Check if the approximation is good
            T_check = np.einsum('i,j,k->ijk', U_c, V_c, W_c)
            error = np.max(np.abs(T_check - T_target))

            if error < 1e-8:
                # Try to merge: replace factors c1 and c2 with the new factor
                mask = [i for i in range(r) if i != c1 and i != c2]
                U_new = np.column_stack([U[:, mask], U_c])
                V_new = np.column_stack([V[:, mask], V_c])
                W_new = np.column_stack([W[:, mask], W_c])
                r_new = r - 1

                # Verify the new decomposition
                T_full = np.einsum('ir,jr,kr->ijk', U_new, V_new, W_new)
                T_full_err = np.max(np.abs(T_full - T_target))  # This is not the full tensor, just the merged part

                # We need to verify the full tensor
                T_full_check = np.zeros((nm, mp, pn))
                for c in range(r_new):
                    T_full_check += np.einsum('i,j,k->ijk', U_new[:, c], V_new[:, c], W_new[:, c])

                # This is expensive, so we only do it if the local approximation is good
                if error < 1e-8:
                    # Verify the full tensor
                    full_err = 0.0
                    for (ui, vi, wi) in ENTRIES:
                        val = np.dot(U_new[ui, :], V_new[vi, :] * W_new[wi, :])
                        err = abs(val - 1.0)
                        if err > full_err:
                            full_err = err
                    for i in range(nm):
                        for j in range(mp):
                            for k in range(pn):
                                if (i, j, k) in _entry_set:
                                    continue
                                val = np.dot(U_new[i, :], V_new[j, :] * W_new[k, :])
                                err = abs(val)
                                if err > full_err:
                                    full_err = err

                    if full_err < 1e-10:
                        best = (U_new, V_new, W_new, r_new)
                        best_err = full_err
                        U, V, W, r = U_new, V_new, W_new, r_new
                        break

        return best

    def polish(U, V, W, r):
        """Round to nearest half-integer and verify."""
        U2 = np.round(U * 2) / 2
        V2 = np.round(V * 2) / 2
        W2 = np.round(W * 2) / 2
        if verify_fast(U2, V2, W2, r):
            return U2, V2, W2, r
        return U, V, W, r

    def randomized_search(U, V, W, r, iterations=100):
        """Random perturbations + rounding + removal."""
        best = (U, V, W, r)
        for _ in range(iterations):
            if elapsed() > DEADLINE:
                break
            U_p = U + np.random.randn(nm, r) * 0.05
            V_p = V + np.random.randn(mp, r) * 0.05
            W_p = W + np.random.randn(pn, r) * 0.05
            U_r = np.round(U_p * 2) / 2
            V_r = np.round(V_p * 2) / 2
            W_r = np.round(W_p * 2) / 2
            if verify_fast(U_r, V_r, W_r, r):
                U_r, V_r, W_r, r_r = remove_redundant_factors(U_r, V_r, W_r, r)
                if r_r < best[3]:
                    best = (U_r, V_r, W_r, r_r)
        return best

    # Build initial decomposition
    U, V, W, r = build_rank55()
    if not verify_fast(U, V, W, r):
        U, V, W, r = build_schoolbook()

    # Remove redundant factors
    U, V, W, r = remove_redundant_factors(U, V, W, r)

    # Try ALS refinement to find a lower-rank decomposition
    if elapsed() < DEADLINE - 10:
        for target_rank in range(r - 1, max(nm, mp, pn) - 1, -1):
            if elapsed() > DEADLINE - 5:
                break
            U, V, W, r = als_refine(U, V, W, r, target_rank, max_restarts=5, max_iters=10)
            if r < target_rank + 1:
                break

    # Try to merge factors
    if elapsed() < DEADLINE - 5:
        U, V, W, r = try_merge_factors(U, V, W, r, max_attempts=20)

    # Randomized search
    if elapsed() < DEADLINE - 5:
        U, V, W, r = randomized_search(U, V, W, r, iterations=50)

    # Final polish
    U, V, W, r = polish(U, V, W, r)

    # Final verification; fallback to schoolbook if broken
    if not verify_fast(U, V, W, r):
        U, V, W, r = build_schoolbook()

    # Convert to lists of lists for the interface
    U_out = U.tolist()
    V_out = V.tolist()
    W_out = W.tolist()

    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
