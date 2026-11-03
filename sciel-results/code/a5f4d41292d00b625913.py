"""Matrix multiplication 2x3x6 - hybrid: sparse-W linear solve + Gauss-Newton fallback."""
import time
import numpy as np
import scipy.optimize as opt
from itertools import product

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    n, m, p = 2, 3, 6

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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_result = schoolbook()
    best_rank = n * m * p

    def verify_full(U, V, W, n_s, m_s, p_s):
        r = U.shape[1]
        for i in range(n_s):
            for j in range(m_s):
                for k in range(p_s):
                    val = 0.0
                    for c in range(r):
                        val += U[i * m_s + j, c] * V[j * p_s + k, c] * W[k * n_s + i, c]
                    if abs(val - 1.0) > 1e-8:
                        return False
        return True

    def compute_residuals(U, V, W, n_s, m_s, p_s):
        r = U.shape[1]
        residuals = np.zeros(n_s * m_s * p_s)
        idx = 0
        for i in range(n_s):
            for j in range(m_s):
                for k in range(p_s):
                    val = 0.0
                    for c in range(r):
                        val += U[i * m_s + j, c] * V[j * p_s + k, c] * W[k * n_s + i, c]
                    residuals[idx] = val - 1.0
                    idx += 1
        return residuals

    def snap_and_verify(U, V, W, rank):
        """Snap to half-integers and verify exactness."""
        for factor in (2.0, 1.0):
            U_snap = np.round(U * factor) / factor
            V_snap = np.round(V * factor) / factor
            W_snap = np.round(W * factor) / factor
            if verify_full(U_snap, V_snap, W_snap, n, m, p):
                return U_snap, V_snap, W_snap
        return None

    # ---- STRUCTURAL IDEA: fix sparse W, solve linear system for U,V ----
    # For a fixed W (pn x r), the equations sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # are multilinear. But if we treat one factor as fixed and solve the others,
    # we can reduce to a linear system by flattening U and V and using the
    # structure. Simpler: for fixed W and fixed V, U is linear; for fixed W,U, V linear.
    # We use alternating linear solves: fix W, solve for U,V jointly via
    # a linear system built from the tensor equations.

    def solve_linear_for_UV(W, rank):
        """Given W (pn x rank), solve linear system for U (nm x rank), V (mp x rank).
        The equations are bilinear in U,V given W, so not linear jointly.
        Instead we do: for each column c, define a_c = W[:,c]. The contribution
        is U[i*m+j,c]*V[j*p+k,c]*a_c[k*n+i]. Sum over c = 1.
        This is still multilinear. We handle it via alternating linearization:
        fix V, solve linear for U; fix U, solve linear for V; iterate.
        """
        # Initialize U, V randomly
        np.random.seed(7)
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)

        # Alternating linear solves
        for it in range(60):
            # Solve for U given V, W: for each (i,j), sum_c U[i*m+j,c] * (V[j*p+*,c]*W[*,i]) = 1
            # For fixed (i,j): sum_c U[i*m+j,c] * g_{i,j,c} = 1 where g_{i,j,c} = sum_k V[j*p+k,c]*W[k*n+i,c]
            # This is a single linear equation per (i,j) with rank unknowns U[i*m+j,:].
            # Under-determined (rank unknowns, 1 equation) -> use least squares / min norm.
            for i in range(n):
                for j in range(m):
                    g = np.zeros(rank)
                    for k in range(p):
                        g += V[j * p + k, :] * W[k * n + i, :]
                    # min ||U_row|| s.t. U_row . g = 1
                    if np.linalg.norm(g) < 1e-12:
                        continue
                    U[i * m + j, :] = g / (g @ g)
            # Solve for V given U, W: for each (j,k): sum_c V[j*p+k,c] * h_{j,k,c} = 1 where h = sum_i U[i*m+j,c]*W[k*n+i,c]
            for j in range(m):
                for k in range(p):
                    h = np.zeros(rank)
                    for i in range(n):
                        h += U[i * m + j, :] * W[k * n + i, :]
                    if np.linalg.norm(h) < 1e-12:
                        continue
                    V[j * p + k, :] = h / (h @ h)

        # Verify
        if verify_full(U, V, W, n, m, p):
            return U, V
        return None

    def try_sparse_W(rank):
        """Try sparse half-integer W patterns and solve for U,V."""
        # Candidate W entries: small half-integers
        cand = [-1.0, -0.5, 0.0, 0.5, 1.0]
        pn = p * n  # 12
        # We'll do a randomized search over W patterns (sparse)
        for trial in range(2000):
            if time.time() > deadline - 15:
                break
            # Random sparse W: most entries 0, some from cand
            W = np.zeros((pn, rank))
            nnz = max(1, rank * 2)
            for c in range(rank):
                # choose a few nonzero rows
                nz = np.random.choice(pn, size=min(nnz, pn), replace=False)
                for ridx in nz:
                    W[ridx, c] = np.random.choice(cand)
            # Ensure W is not degenerate
            if np.linalg.matrix_rank(W) < 1:
                continue
            sol = solve_linear_for_UV(W, rank)
            if sol:
                U, V = sol
                snapped = snap_and_verify(U, V, W, rank)
                if snapped:
                    return snapped
        return None

    # ---- Gauss-Newton fallback (from current program) ----
    def solve_gauss_newton(rank, seed=0, max_iter=1000, tol=1e-12):
        size_U = n * m * rank
        size_V = m * p * rank
        size_W = p * n * rank
        total_size = size_U + size_V + size_W

        np.random.seed(42 + rank + seed)
        x0 = np.random.randn(total_size)

        def fun(x):
            U = x[:size_U].reshape(n * m, rank)
            V = x[size_U:size_U + size_V].reshape(m * p, rank)
            W = x[size_U + size_V:].reshape(p * n, rank)
            return compute_residuals(U, V, W, n, m, p)

        try:
            res = opt.least_squares(fun, x0, method='lm', max_nfev=400)
            x_opt = res.x
            err = np.sum(res.fun ** 2)
            if err < 1e-4:
                U_opt = x_opt[:size_U].reshape(n * m, rank)
                V_opt = x_opt[size_U:size_U + size_V].reshape(m * p, rank)
                W_opt = x_opt[size_U + size_V:].reshape(p * n, rank)
                return U_opt, V_opt, W_opt, err
        except Exception:
            pass
        return None

    # ---- Search ----
    # Known rank for 2x3x6 is 20. Try from 20 upward.
    for rank in range(20, 36):
        if time.time() > deadline - 10:
            break

        found = False

        # Strategy 1: sparse-W linear solve (structural idea)
        if time.time() < deadline - 20:
            res = try_sparse_W(rank)
            if res:
                U_f, V_f, W_f = res
                if rank < best_rank:
                    best_rank = rank
                    best_result = {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
                    found = True
                    if rank == 20:
                        break

        # Strategy 2: Gauss-Newton fallback
        if not found:
            for seed in range(3):
                if time.time() > deadline - 5:
                    break
                result = solve_gauss_newton(rank, seed=seed)
                if result:
                    U_sol, V_sol, W_sol, err = result
                    snapped = snap_and_verify(U_sol, V_sol, W_sol, rank)
                    if snapped:
                        U_final, V_final, W_final = snapped
                        if rank < best_rank:
                            best_rank = rank
                            best_result = {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
                            found = True
                            if rank == 20:
                                break
        if found and rank == 20:
            break

    return best_result
# EVOLVE-BLOCK-END
