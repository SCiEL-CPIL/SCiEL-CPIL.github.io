# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar
import sys

def arithmetic_kakeya_distribution() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin before 600s kill

    n = 10
    ys = np.arange(n)
    xs = np.arange(n)
    X, Y = np.meshgrid(xs, ys)  # X[y,x] = x, Y[y,x] = y

    # Precompute the index arrays for each projection
    XY_diff = (X - Y + (n - 1)).flatten()  # 0 to 2*(n-1)
    X_flat = X.flatten()  # 0 to n-1
    XY_sum = (X + Y).flatten()  # 0 to 2*(n-1)
    X2Y = (X + 2 * Y).flatten()  # 0 to 3*(n-1)

    # Precompute inverse mappings: for each bin, which cells map to it
    # This enables efficient gradient computation
    n_bins_diff = 2 * n - 1  # 19
    n_bins_x = n  # 10
    n_bins_sum = 2 * n - 1  # 19
    n_bins_2y = 3 * n - 1  # 29

    # Build sparse inverse mapping matrices (cell -> bin)
    # For gradient: dH/dbin = -log2(p_bin) - 1/ln(2)
    # dH/dp_cell = dH/dbin * (1/bin_weight) where bin_weight = sum of p in that bin

    def _get_proj_probs(P_flat):
        """Get pooled probabilities for all four projections."""
        p_diff = np.bincount(XY_diff, weights=P_flat, minlength=n_bins_diff)
        p_x = np.bincount(X_flat, weights=P_flat, minlength=n_bins_x)
        p_sum = np.bincount(XY_sum, weights=P_flat, minlength=n_bins_sum)
        p_2y = np.bincount(X2Y, weights=P_flat, minlength=n_bins_2y)
        return p_diff, p_x, p_sum, p_2y

    def _entropy_and_grad(p):
        """Compute entropy and gradient w.r.t. p (ignoring zeros)."""
        mask = p > 1e-15
        if not np.any(mask):
            return 0.0, np.zeros_like(p)
        p_m = p[mask]
        H = -np.sum(p_m * np.log2(p_m))
        # dH/dp = -log2(p) - 1/ln(2) for p > 0
        grad = np.zeros_like(p)
        grad[mask] = -np.log2(p_m) - 1.0 / np.log(2.0)
        return H, grad

    def compute_ratio(P):
        """Compute D/M ratio for a given distribution P."""
        P = np.maximum(P, 0)
        total = P.sum()
        if total <= 1e-9:
            return 0.0
        P = P / total
        P_flat = P.flatten()
        p_diff, p_x, p_sum, p_2y = _get_proj_probs(P_flat)
        H_diff, _ = _entropy_and_grad(p_diff)
        H_x, _ = _entropy_and_grad(p_x)
        H_sum, _ = _entropy_and_grad(p_sum)
        H_2y, _ = _entropy_and_grad(p_2y)
        M = max(H_x, H_sum, H_2y)
        if M <= 1e-9:
            return 0.0
        return H_diff / M

    def _objective_and_grad(params):
        """Compute objective and gradient for log-parameterized P."""
        P = np.exp(params).reshape(n, n)
        total = P.sum()
        if total <= 1e-9:
            return 1e10, np.zeros_like(params)
        P = P / total
        P_flat = P.flatten()

        p_diff, p_x, p_sum, p_2y = _get_proj_probs(P_flat)
        H_diff, g_diff = _entropy_and_grad(p_diff)
        H_x, g_x = _entropy_and_grad(p_x)
        H_sum, g_sum = _entropy_and_grad(p_sum)
        H_2y, g_2y = _entropy_and_grad(p_2y)

        M = max(H_x, H_sum, H_2y)
        if M <= 1e-9:
            return 1e10, np.zeros_like(params)

        ratio = H_diff / M
        # Gradient of ratio = (g_diff * M - H_diff * g_M) / M^2
        # g_M = gradient of max, which is the gradient of the argmax component
        # For exact max, we need to handle the non-smooth case
        # Use smooth approximation: weight by softmax of entropies
        temps = np.array([H_x, H_sum, H_2y])
        t_max = np.max(temps)
        exp_vals = np.exp((temps - t_max) / 0.01)
        weights = exp_vals / np.sum(exp_vals)
        g_M = weights[0] * g_x + weights[1] * g_sum + weights[2] * g_2y

        # Gradient w.r.t. P_flat
        grad_P_diff = np.zeros_like(P_flat)
        np.add.at(grad_P_diff, XY_diff, g_diff)
        grad_P_x = np.zeros_like(P_flat)
        np.add.at(grad_P_x, X_flat, g_x)
        grad_P_sum = np.zeros_like(P_flat)
        np.add.at(grad_P_sum, XY_sum, g_sum)
        grad_P_2y = np.zeros_like(P_flat)
        np.add.at(grad_P_2y, X2Y, g_2y)

        # dH_diff/dP_flat = grad_P_diff
        # dM/dP_flat = grad_P_x * w_x + grad_P_sum * w_sum + grad_P_2y * w_2y
        g_M_flat = weights[0] * grad_P_x + weights[1] * grad_P_sum + weights[2] * grad_P_2y

        # d(ratio)/dP_flat = (g_diff_flat * M - H_diff * g_M_flat) / M^2
        grad_ratio_flat = (grad_P_diff * M - H_diff * g_M_flat) / (M * M)

        # Chain rule: dP_flat/dparams = P_flat (since P = exp(params)/sum(exp(params)))
        # dP_flat/dparams_i = P_flat_i * (delta_ij - sum_j P_flat_j)
        # For simplicity, approximate with P_flat_i * delta_ij (ignoring normalization effect)
        # More precisely: dP_i/dtheta_j = P_i * (delta_ij - P_j)
        grad_params = grad_ratio_flat * P_flat  # approximate, good enough for optimization

        return -ratio, -grad_params

    def _pairwise_descent(P, max_rounds=50, tol=1e-12):
        """Pairwise coordinate descent: adjust pairs of cells to improve ratio."""
        P = P.copy()
        P = np.maximum(P, 1e-15)
        P /= P.sum()
        best_ratio = compute_ratio(P)

        for round_num in range(max_rounds):
            if time.time() > deadline - 30:
                break
            improved = False
            for y in range(n):
                for x in range(n):
                    for y2 in range(y + 1, n):
                        for x2 in range(n):
                            if (y, x) == (y2, x2):
                                continue
                            p1 = P[y, x]
                            p2 = P[y2, x2]
                            if p1 < 1e-14 and p2 < 1e-14:
                                continue

                            # Try moving probability mass between these two cells
                            for frac in [0.1, 0.3, 0.5, 0.7, 0.9]:
                                # Move frac * p1 to p2
                                P_trial = P.copy()
                                delta = frac * p1
                                P_trial[y, x] -= delta
                                P_trial[y2, x2] += delta
                                P_trial = np.maximum(P_trial, 0)
                                total = P_trial.sum()
                                if total <= 1e-9:
                                    continue
                                P_trial /= total
                                trial_ratio = compute_ratio(P_trial)
                                if trial_ratio > best_ratio + tol:
                                    P = P_trial
                                    best_ratio = trial_ratio
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break

        return P, best_ratio

    # Start with structured initializations
    np.random.seed(42)

    # Initial point 1: uniform on anti-diagonals (maximizes H(X-Y))
    P_best = np.zeros((n, n), dtype=np.float64)
    for y in range(n):
        for x in range(n):
            k = abs(x - y)
            P_best[y, x] = 1.0 / (19.0 * (n - k))
    P_best /= P_best.sum()
    best_ratio = compute_ratio(P_best)

    # Generate diverse starting points
    starting_points = []

    # 1. Anti-diagonal uniform (current best)
    starting_points.append(P_best.copy())

    # 2. Concentrate on cells where x-y has high entropy but x, x+y, x+2y are concentrated
    for subset_size in [3, 4, 5, 6, 7]:
        for offset in range(n):
            P_candidate = np.zeros((n, n), dtype=np.float64)
            for y in range(n):
                for x in range(n):
                    d = (x - y + n) % (2 * n - 1)
                    if d in range(offset, offset + subset_size) or d in range(2 * n - 1 - subset_size, 2 * n - 1):
                        P_candidate[y, x] = 1.0
            if P_candidate.sum() > 0:
                P_candidate /= P_candidate.sum()
                starting_points.append(P_candidate.copy())

    # 3. Random Dirichlet samples
    for _ in range(15):
        P_rand = np.random.dirichlet(np.ones(n * n))
        P_rand = P_rand.reshape(n, n)
        starting_points.append(P_rand)

    # 4. Structured: emphasize certain rows/columns
    for alpha in [0.3, 0.5, 0.7, 0.9]:
        P_struct = np.zeros((n, n), dtype=np.float64)
        for y in range(n):
            for x in range(n):
                P_struct[y, x] = np.exp(-alpha * (x - y) ** 2 / (2 * n))
        P_struct /= P_struct.sum()
        starting_points.append(P_struct)

    # 5. Sparse patterns
    for trial in range(10):
        P_sparse = np.zeros((n, n), dtype=np.float64)
        for _ in range(20):
            ry = np.random.randint(0, n)
            rx = np.random.randint(0, n)
            P_sparse[ry, rx] = 1.0
        P_sparse /= P_sparse.sum()
        starting_points.append(P_sparse)

    # 6. More structured starts
    for alpha in [0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        P_a = np.zeros((n, n))
        for y in range(n):
            for x in range(n):
                P_a[y, x] = np.exp(-alpha * abs(x - y))
        P_a /= P_a.sum()
        starting_points.append(P_a.copy())

    for beta in [0.1, 0.3, 0.5, 1.0]:
        P_b = np.zeros((n, n))
        for y in range(n):
            for x in range(n):
                P_b[y, x] = np.exp(-beta * (x + y) ** 2)
        P_b /= P_b.sum()
        starting_points.append(P_b.copy())

    for gamma in [0.1, 0.3, 0.5, 1.0, 2.0]:
        P_c = np.zeros((n, n))
        for y in range(n):
            for x in range(n):
                P_c[y, x] = np.exp(-gamma * (x - y) ** 2)
        P_c /= P_c.sum()
        starting_points.append(P_c.copy())

    for delta in [0.1, 0.3, 0.5, 1.0, 2.0]:
        P_d = np.zeros((n, n))
        for y in range(n):
            for x in range(n):
                P_d[y, x] = np.exp(-delta * (x + y - (n - 1)) ** 2)
        P_d /= P_d.sum()
        starting_points.append(P_d.copy())

    for d_set in [[0], [0, 1], [-1, 0, 1], [0, 2], [-2, 0, 2], [0, 3, -3]]:
        P_line = np.zeros((n, n))
        for y in range(n):
            for x in range(n):
                if (x - y) in d_set:
                    P_line[y, x] = 1.0
        if P_line.sum() > 0:
            P_line /= P_line.sum()
            starting_points.append(P_line.copy())

    for d_set, width in [([0], 1.0), ([0], 2.0), ([0, 1], 1.5), ([-1, 0, 1], 2.0)]:
        P_hybrid = np.zeros((n, n))
        for y in range(n):
            for x in range(n):
                d = x - y
                min_dist = min(abs(d - dd) for dd in d_set)
                P_hybrid[y, x] = np.exp(-min_dist ** 2 / (2 * width ** 2))
        if P_hybrid.sum() > 0:
            P_hybrid /= P_hybrid.sum()
            starting_points.append(P_hybrid.copy())

    for phase in [0, 1]:
        P_check = np.zeros((n, n))
        for y in range(n):
            for x in range(n):
                if (x + y + phase) % 2 == 0:
                    P_check[y, x] = 1.0
        if P_check.sum() > 0:
            P_check /= P_check.sum()
            starting_points.append(P_check.copy())

    for alpha in [0.5, 1.0, 2.0]:
        P_prod = np.zeros((n, n))
        for y in range(n):
            for x in range(n):
                P_prod[y, x] = np.exp(-alpha * abs(x - y) / n) * (
                    1 + 0.1 * np.sin(x * np.pi / n) * np.sin(y * np.pi / n))
        if P_prod.sum() > 0:
            P_prod /= P_prod.sum()
            starting_points.append(P_prod.copy())

    # Optimize from each starting point
    for i, P_init in enumerate(starting_points):
        if time.time() > deadline - 60:
            break

        P_init_safe = np.maximum(P_init, 1e-12)
        params_init = np.log(P_init_safe.flatten())

        try:
            # Phase 1: L-BFGS-B with gradient
            result = minimize(
                _objective_and_grad, params_init, method='L-BFGS-B',
                jac=True,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 5000}
            )
            P_opt = np.exp(result.x).reshape(n, n)
            P_opt = np.maximum(P_opt, 0)
            total = P_opt.sum()
            if total > 1e-9:
                P_opt /= total
                ratio = compute_ratio(P_opt)
                if ratio > best_ratio:
                    best_ratio = ratio
                    P_best = P_opt.copy()
        except Exception:
            pass

        # Phase 2: Pairwise descent refinement
        if time.time() > deadline - 60:
            break
        try:
            P_pw, ratio_pw = _pairwise_descent(P_best, max_rounds=20, tol=1e-12)
            if ratio_pw > best_ratio:
                best_ratio = ratio_pw
                P_best = P_pw.copy()
        except Exception:
            pass

    # Final high-precision polish with Nelder-Mead on the best candidate
    if time.time() < deadline - 30:
        try:
            def nm_objective(params):
                P = np.exp(params).reshape(n, n)
                P = np.maximum(P, 0)
                total = P.sum()
                if total <= 1e-9:
                    return 1e10
                P = P / total
                return -compute_ratio(P)

            P_init_nm = np.maximum(P_best, 1e-12)
            params_init_nm = np.log(P_init_nm.flatten())

            res_nm = minimize(
                nm_objective, params_init_nm,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )

            P_nm = np.exp(res_nm.x).reshape(n, n)
            P_nm = np.maximum(P_nm, 0)
            total_nm = P_nm.sum()
            if total_nm > 1e-9:
                P_nm /= total_nm
                ratio_nm = compute_ratio(P_nm)
                if ratio_nm > best_ratio:
                    best_ratio = ratio_nm
                    P_best = P_nm.copy()
        except Exception:
            pass

    # Final pairwise descent polish
    if time.time() < deadline - 10:
        try:
            P_final, ratio_final = _pairwise_descent(P_best, max_rounds=30, tol=1e-13)
            if ratio_final > best_ratio:
                best_ratio = ratio_final
                P_best = P_final.copy()
        except Exception:
            pass

    # Final normalization and clipping
    P_best = np.maximum(P_best, 0)
    total = P_best.sum()
    if total <= 1e-9:
        P_best = np.ones((n, n)) / (n * n)
    else:
        P_best /= total

    return P_best
# EVOLVE-BLOCK-END
