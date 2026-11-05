"""Matrix multiplication 3x3 by 3x7 - rank minimization.

Strategy: fix U and V to a generic, full-rank basis (Chebyshev nodes + random
integers), solve for W via linear regression, then prune linearly-dependent /
negligible W columns to identify a minimal-rank support. A smooth epigraph
polish (log-sum-exp / soft-min) refines the support, and a final exact local
solve tightens the half-integer solution.
"""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]
TOL = 1e-9
DEADLINE = 145.0  # safety margin under 150s wall-clock


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    np.random.seed(42)

    # ---- Schoolbook baseline (rank 63) ----
    r_sb = n * m * p
    U_sb = np.zeros((n, r_sb))
    V_sb = np.zeros((m, r_sb))
    W_sb = np.zeros((p, r_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i, col] = 1
                V_sb[j, col] = 1
                W_sb[k, col] = 1
                col += 1

    def to_interface(U, V, W):
        r = U.shape[1]
        U_list = [[0.0] * r for _ in range(n * m)]
        V_list = [[0.0] * r for _ in range(m * p)]
        W_list = [[0.0] * r for _ in range(p * n)]
        for i in range(n):
            for j in range(m):
                for c in range(r):
                    U_list[i * m + j][c] = U[i, c]
        for j in range(m):
            for k in range(p):
                for c in range(r):
                    V_list[j * p + k][c] = V[j, c]
        for k in range(p):
            for i in range(n):
                for c in range(r):
                    W_list[k * n + i][c] = W[k, c]
        return U_list, V_list, W_list

    def verify(U, V, W):
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = float(np.dot(U[i], V[j] * W[k]))
                    if abs(val - 1.0) > TOL:
                        return False
        return True

    # ---- Matrix pencil: fix U,V to a generic full-rank basis ----
    def cheb_nodes(count, offset=0.0):
        # Chebyshev nodes in [-1,1], shifted to be positive and distinct
        t = np.cos(np.pi * (np.arange(count) + 0.5) / count)
        return t + 1.0 + offset  # in (1,3)

    def build_basis(r):
        # U basis: n x r, V basis: m x r. Use Chebyshev + random integer mix
        # so columns are generically full-rank and distinct.
        U = np.zeros((n, r))
        V = np.zeros((m, r))
        # Primary Chebyshev structure
        U[0, :] = cheb_nodes(r)
        V[0, :] = cheb_nodes(r, offset=0.7)
        # Fill remaining rows with random integers (small) for genericity
        for i in range(1, n):
            U[i, :] = np.random.randint(-2, 3, size=r)
        for j in range(1, m):
            V[j, :] = np.random.randint(-2, 3, size=r)
        return U, V

    def solve_W(U, V):
        # A[i*m+j, c] = U[i,c]*V[j,c];  A @ W[k,:]^T = 1  (for each k)
        r = U.shape[1]
        A = np.zeros((n * m, r))
        for i in range(n):
            for j in range(m):
                A[i * m + j, :] = U[i, :] * V[j, :]
        ones = np.ones(n * m)
        W = np.zeros((p, r))
        for k in range(p):
            W[k, :], residuals, rank, sv = np.linalg.lstsq(A, ones, rcond=None)
        return W, A

    def prune_support(W, A, max_iter=200):
        """Greedy prune: remove W columns with negligible coefficient or
        linear dependence, re-solve, repeat until stable. Returns minimal
        support column indices."""
        r = W.shape[1]
        active = list(range(r))
        for _ in range(max_iter):
            if time.time() > DEADLINE - 3:
                break
            # Solve restricted least squares on active columns
            A_act = A[:, active]
            if A_act.shape[1] == 0:
                break
            # Check linear dependence via rank
            _, s, _ = np.linalg.svd(A_act, full_matrices=False)
            rank = int(np.sum(s > 1e-8))
            # Remove columns whose W-row norm is tiny (negligible coefficient)
            new_active = []
            for idx in active:
                if np.max(np.abs(W[:, idx])) > 1e-6:
                    new_active.append(idx)
            # Also drop columns beyond numerical rank (linear dependence)
            if len(new_active) > rank:
                # Keep the 'rank' columns with largest W norm
                norms = [np.max(np.abs(W[:, idx])) for idx in new_active]
                order = np.argsort(norms)[::-1]
                new_active = [new_active[o] for o in order[:rank]]
            if sorted(new_active) == sorted(active):
                break
            active = new_active
            # Re-solve W on active columns
            A_act = A[:, active]
            ones = np.ones(n * m)
            W_new = np.zeros((p, len(active)))
            for k in range(p):
                W_new[k, :], _, _, _ = np.linalg.lstsq(A_act, ones, rcond=None)
            # Map back
            W_full = np.zeros((p, r))
            for a, c in zip(active, range(len(active))):
                W_full[:, a] = W_new[:, c]
            W = W_full
        return active

    def soft_polish(U, V, W, support, iters=80):
        """Smooth epigraph / soft-min polish: minimize soft log-sum-exp of
        squared residuals over the active support, with auxiliary t (epigraph).
        Uses gradient-free Nelder-Mead on a smoothed objective."""
        from scipy.optimize import minimize
        r = U.shape[1]
        idx = support
        A = np.zeros((n * m, len(idx)))
        for i in range(n):
            for j in range(m):
                A[i * m + j, :] = U[i, idx] * V[j, idx]
        ones = np.ones(n * m)

        def residuals(x):
            # x: p*len(idx) flattened W rows
            Wx = x.reshape(p, len(idx))
            res = np.zeros(n * m)
            for k in range(p):
                res = res + 0  # placeholder
            # For each (i,j): sum_c U[i,c]V[j,c]W[k,c] should equal 1 for all k
            # Residual vector over (k, i*m+j)
            R = np.zeros((p, n * m))
            for k in range(p):
                R[k, :] = A @ Wx[k, :] - ones
            return R

        def objective(x):
            R = residuals(x)
            # soft log-sum-exp of squared residuals (smooth min/max surrogate)
            sq = R * R
            M = np.max(sq)
            if M > 1e-6:
                lse = M + np.log(np.sum(np.exp(sq - M)))
            else:
                lse = np.sum(sq)
            return lse

        x0 = W[:, idx].reshape(-1)
        res = minimize(objective, x0, method='Nelder-Mead',
                       options={'maxiter': iters, 'xatol': 1e-9, 'fatol': 1e-12})
        W_new = np.zeros((p, r))
        W_new[:, idx] = res.x.reshape(p, len(idx))
        return W_new

    def round_half(mat):
        return np.round(mat * 2.0) / 2.0

    def exact_local_solve(U, V, W, support, max_iter=300):
        """Final high-precision polish: exact local solve on the support,
        trying half-integer rounding of the least-squares solution."""
        idx = support
        A = np.zeros((n * m, len(idx)))
        for i in range(n):
            for j in range(m):
                A[i * m + j, :] = U[i, idx] * V[j, idx]
        ones = np.ones(n * m)
        # Least squares per k
        W_ls = np.zeros((p, len(idx)))
        for k in range(p):
            W_ls[k, :], _, _, _ = np.linalg.lstsq(A, ones, rcond=None)
        # Try rounding to half-integers with local search
        best = W_ls.copy()
        best_err = np.inf
        for _ in range(max_iter):
            if time.time() > DEADLINE - 2:
                break
            R = np.zeros((p, n * m))
            for k in range(p):
                R[k, :] = A @ best[k, :] - ones
            err = np.max(R * R)
            if err < best_err:
                best_err = err
                if best_err < TOL:
                    break
            # Perturb a random entry
            k = np.random.randint(p)
            c = np.random.randint(len(idx))
            old = best[k, c]
            best[k, c] = round_half(old + np.random.choice([-1.0, -0.5, 0.5, 1.0]))
        # Map back to full W
        W_full = np.zeros((p, U.shape[1]))
        W_full[:, idx] = best
        return W_full

    # ---- Main search: increasing rank, matrix-pencil + prune + polish ----
    best_rank = r_sb
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()

    lower = max(n * m, m * p, p * n)  # 21
    for r in range(lower, r_sb):
        if time.time() > DEADLINE - 15:
            break
        for attempt in range(6):
            if time.time() > DEADLINE - 10:
                break
            U, V = build_basis(r)
            W, A = solve_W(U, V)
            support = prune_support(W, A)
            if len(support) < r:
                # Re-solve exactly on support, then polish
                W = soft_polish(U, V, W, support)
                W = exact_local_solve(U, V, W, support)
                # Verify on the support-only submatrices
                Us = U[:, support]
                Vs = V[:, support]
                Ws = W[:, support]
                if verify(Us, Vs, Ws):
                    best_rank = len(support)
                    best_U = Us
                    best_V = Vs
                    best_W = Ws
                    break
            else:
                # No pruning; try direct polish on full support
                W = soft_polish(U, V, W, list(range(r)))
                W = exact_local_solve(U, V, W, list(range(r)))
                if verify(U, V, W):
                    best_rank = r
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    break
        if best_rank <= r:
            break

    # Guarantee a valid fallback
    if not verify(best_U, best_V, best_W):
        best_rank = r_sb
        best_U, best_V, best_W = U_sb, V_sb, W_sb

    U_list, V_list, W_list = to_interface(best_U, best_V, best_W)
    return {"U": U_list, "V": V_list, "W": W_list}
