# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.optimize import linear_sum_assignment
from scipy.special import digamma

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

    def compute_ratio(P):
        """Compute D/M ratio for a given distribution P (vectorized)."""
        P = np.maximum(P, 0)
        total = P.sum()
        if total <= 1e-9:
            return 0.0
        P = P / total
        P_flat = P.flatten()

        # H(X-Y)
        XY_dist = np.bincount(XY_diff, weights=P_flat, minlength=2 * n - 1)
        XY_nonzero = XY_dist[XY_dist > 1e-15]
        H_XY = -np.sum(XY_nonzero * np.log2(XY_nonzero)) if len(XY_nonzero) > 0 else 0.0

        # H(X)
        X_dist = np.bincount(X_flat, weights=P_flat, minlength=n)
        X_nonzero = X_dist[X_dist > 1e-15]
        H_X = -np.sum(X_nonzero * np.log2(X_nonzero)) if len(X_nonzero) > 0 else 0.0

        # H(X+Y)
        XY_sum_dist = np.bincount(XY_sum, weights=P_flat, minlength=2 * n - 1)
        XY_sum_nonzero = XY_sum_dist[XY_sum_dist > 1e-15]
        H_XY_sum = -np.sum(XY_sum_nonzero * np.log2(XY_sum_nonzero)) if len(XY_sum_nonzero) > 0 else 0.0

        # H(X+2Y)
        X2Y_dist = np.bincount(X2Y, weights=P_flat, minlength=3 * n - 1)
        X2Y_nonzero = X2Y_dist[X2Y_dist > 1e-15]
        H_X2Y = -np.sum(X2Y_nonzero * np.log2(X2Y_nonzero)) if len(X2Y_nonzero) > 0 else 0.0

        M = max(H_X, H_XY_sum, H_X2Y)
        if M <= 1e-9:
            return 0.0
        return H_XY / M

    def soft_max(vals, temperature=0.1):
        """Smooth approximation of max using log-sum-exp."""
        vals = np.array(vals)
        v_max = np.max(vals)
        if v_max <= 0:
            return 0.0
        exp_vals = np.exp((vals - v_max) / temperature)
        return v_max + temperature * np.log(np.sum(exp_vals))

    def objective(params):
        """Negative of the ratio for minimization."""
        P = np.exp(params).reshape(n, n)
        total = P.sum()
        if total <= 1e-9:
            return 1e10
        P = P / total
        return -compute_ratio(P)

    def objective_smooth(params, temperature=0.05):
        """Smooth version with soft-max for better gradient info."""
        P = np.exp(params).reshape(n, n)
        total = P.sum()
        if total <= 1e-9:
            return 1e10
        P = P / total
        P_flat = P.flatten()

        XY_dist = np.bincount(XY_diff, weights=P_flat, minlength=2 * n - 1)
        XY_nonzero = XY_dist[XY_dist > 1e-15]
        H_XY = -np.sum(XY_nonzero * np.log2(XY_nonzero)) if len(XY_nonzero) > 0 else 0.0

        X_dist = np.bincount(X_flat, weights=P_flat, minlength=n)
        X_nonzero = X_dist[X_dist > 1e-15]
        H_X = -np.sum(X_nonzero * np.log2(X_nonzero)) if len(X_nonzero) > 0 else 0.0

        XY_sum_dist = np.bincount(XY_sum, weights=P_flat, minlength=2 * n - 1)
        XY_sum_nonzero = XY_sum_dist[XY_sum_dist > 1e-15]
        H_XY_sum = -np.sum(XY_sum_nonzero * np.log2(XY_sum_nonzero)) if len(XY_sum_nonzero) > 0 else 0.0

        X2Y_dist = np.bincount(X2Y, weights=P_flat, minlength=3 * n - 1)
        X2Y_nonzero = X2Y_dist[X2Y_dist > 1e-15]
        H_X2Y = -np.sum(X2Y_nonzero * np.log2(X2Y_nonzero)) if len(X2Y_nonzero) > 0 else 0.0

        M_smooth = soft_max([H_X, H_XY_sum, H_X2Y], temperature)
        if M_smooth <= 1e-9:
            return 1e10
        return -H_XY / M_smooth

    # Build diverse starting points
    starting_points = []
    np.random.seed(42)

    ys_grid, xs_grid = np.meshgrid(ys, xs, indexing='ij')
    diff_grid = xs_grid - ys_grid

    # 1. Uniform on each anti-diagonal (maximizes H(X-Y) structure)
    P_uni = np.zeros((n, n))
    for y in range(n):
        for x in range(n):
            k = abs(x - y)
            P_uni[y, x] = 1.0 / (n - k)
    P_uni /= P_uni.sum()
    starting_points.append(P_uni.copy())

    # 2. Gaussian concentration on main diagonal
    for alpha in [0.1, 0.3, 0.5, 1.0, 2.0, 3.0]:
        P_g = np.exp(-alpha * (xs_grid - ys_grid) ** 2)
        P_g /= P_g.sum()
        starting_points.append(P_g.copy())

    # 3. Concentrate on specific x-y values
    for d_set in [[0], [0, 1], [-1, 0, 1], [0, 2], [-2, 0, 2], [0, 3, -3],
                  [1, 2, 3], [-3, -2, -1], [0, 4, -4], [-1, 1], [2, -2, 4, -4]]:
        mask = np.isin(diff_grid, d_set).astype(np.float64)
        if mask.sum() > 0:
            P_line = mask / mask.sum()
            starting_points.append(P_line.copy())

    # 4. Smooth concentration on lines
    for d_set, width in [([0], 0.8), ([0], 1.5), ([0], 2.5), ([0, 1], 1.0), ([-1, 0, 1], 1.5),
                         ([0], 0.5), ([0, 2], 1.2), ([-2, 0, 2], 2.0), ([0, 3], 1.5)]:
        min_dist = np.min([np.abs(diff_grid - dd) for dd in d_set], axis=0)
        P_hybrid = np.exp(-min_dist ** 2 / (2 * width ** 2))
        if P_hybrid.sum() > 0:
            P_hybrid /= P_hybrid.sum()
            starting_points.append(P_hybrid.copy())

    # 5. Anti-diagonal concentration (x+y = constant)
    for delta in [0.1, 0.3, 0.5, 1.0, 2.0]:
        P_anti = np.exp(-delta * (xs_grid + ys_grid - (n - 1)) ** 2)
        P_anti /= P_anti.sum()
        starting_points.append(P_anti.copy())

    # 6. Corner concentration
    for beta in [0.1, 0.3, 0.5, 1.0, 2.0]:
        P_corner = np.exp(-beta * (xs_grid + ys_grid) ** 2)
        P_corner /= P_corner.sum()
        starting_points.append(P_corner.copy())

    # 7. Checkerboard patterns
    for phase in [0, 1]:
        mask = ((xs_grid + ys_grid + phase) % 2 == 0).astype(np.float64)
        if mask.sum() > 0:
            P_check = mask / mask.sum()
            starting_points.append(P_check.copy())

    # 8. Random Dirichlet
    for _ in range(10):
        P_rand = np.random.dirichlet(np.ones(n * n)).reshape(n, n)
        starting_points.append(P_rand.copy())

    # 9. Sparse random
    for _ in range(8):
        P_sparse = np.zeros((n, n))
        for _ in range(15):
            ry = np.random.randint(0, n)
            rx = np.random.randint(0, n)
            P_sparse[ry, rx] = 1.0
        P_sparse /= P_sparse.sum()
        starting_points.append(P_sparse.copy())

    # 10. Product form: concentrate x-y, spread x and x+y
    for alpha in [0.3, 0.5, 0.8, 1.2]:
        P_prod = np.exp(-alpha * diff_grid ** 2) * np.ones((n, n))
        P_prod /= P_prod.sum()
        starting_points.append(P_prod.copy())

    # 11. Fourier-based patterns
    for kx in range(3):
        for ky in range(3):
            if kx == 0 and ky == 0:
                continue
            P_fourier = 0.5 + 0.5 * np.cos(2 * np.pi * kx * xs_grid / n) * np.cos(2 * np.pi * ky * ys_grid / n)
            P_fourier = np.maximum(P_fourier, 0)
            if P_fourier.sum() > 0:
                P_fourier /= P_fourier.sum()
                starting_points.append(P_fourier.copy())

    # Evaluate all starting points
    best_ratio = 0.0
    P_best = np.ones((n, n)) / (n * n)
    for P_init in starting_points:
        ratio_init = compute_ratio(P_init)
        if ratio_init > best_ratio:
            best_ratio = ratio_init
            P_best = P_init.copy()

    # Optimize from each diverse starting point
    for i, P_init in enumerate(starting_points):
        if time.time() > deadline - 60:
            break

        P_init_safe = np.maximum(P_init, 1e-12)
        params_init = np.log(P_init_safe.flatten())

        # Phase 1: Smooth L-BFGS-B
        try:
            res_smooth = minimize(objective_smooth, params_init, method='L-BFGS-B',
                                  args=(0.05,),
                                  options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 5000})
            P_cand = np.exp(res_smooth.x).reshape(n, n)
            P_cand = np.maximum(P_cand, 0)
            if P_cand.sum() > 1e-9:
                P_cand /= P_cand.sum()
                ratio_c = compute_ratio(P_cand)
                if ratio_c > best_ratio:
                    best_ratio = ratio_c
                    P_best = P_cand.copy()
        except Exception:
            pass

        # Phase 2: Exact L-BFGS-B polish
        if time.time() > deadline - 60:
            break
        try:
            P_polish = np.maximum(P_best, 1e-12)
            params_polish = np.log(P_polish.flatten())
            res_exact = minimize(objective, params_polish, method='L-BFGS-B',
                                 options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 3000})
            P_pol = np.exp(res_exact.x).reshape(n, n)
            P_pol = np.maximum(P_pol, 0)
            if P_pol.sum() > 1e-9:
                P_pol /= P_pol.sum()
                ratio_pol = compute_ratio(P_pol)
                if ratio_pol > best_ratio:
                    best_ratio = ratio_pol
                    P_best = P_pol.copy()
        except Exception:
            pass

    # SLSQP refinement on best candidate
    if time.time() < deadline - 60:
        try:
            P_slsqp_init = np.maximum(P_best, 1e-12)
            params_slsqp = np.log(P_slsqp_init.flatten())

            res_slsqp = minimize(
                objective,
                params_slsqp,
                method='SLSQP',
                bounds=[(-50, 50)] * (n * n),
                options={'maxiter': 400, 'ftol': 1e-12, 'disp': False}
            )
            P_slsqp = np.exp(res_slsqp.x).reshape(n, n)
            P_slsqp = np.maximum(P_slsqp, 0)
            if P_slsqp.sum() > 1e-9:
                P_slsqp /= P_slsqp.sum()
                ratio_slsqp = compute_ratio(P_slsqp)
                if ratio_slsqp > best_ratio:
                    best_ratio = ratio_slsqp
                    P_best = P_slsqp.copy()
        except Exception:
            pass

    # Nelder-Mead high-precision polish
    if time.time() < deadline - 30:
        try:
            P_nm_init = np.maximum(P_best, 1e-12)
            params_nm = np.log(P_nm_init.flatten())

            res_nm = minimize(
                objective,
                params_nm,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-15, 'adaptive': True, 'maxfev': 8000}
            )
            P_nm = np.exp(res_nm.x).reshape(n, n)
            P_nm = np.maximum(P_nm, 0)
            if P_nm.sum() > 1e-9:
                P_nm /= P_nm.sum()
                ratio_nm = compute_ratio(P_nm)
                if ratio_nm > best_ratio:
                    best_ratio = ratio_nm
                    P_best = P_nm.copy()
        except Exception:
            pass

    # Final SLSQP on the very best
    if time.time() < deadline - 15:
        try:
            P_final_init = np.maximum(P_best, 1e-12)
            params_final = np.log(P_final_init.flatten())

            res_final = minimize(
                objective,
                params_final,
                method='SLSQP',
                bounds=[(-50, 50)] * (n * n),
                options={'maxiter': 500, 'ftol': 1e-14, 'disp': False}
            )
            P_final = np.exp(res_final.x).reshape(n, n)
            P_final = np.maximum(P_final, 0)
            if P_final.sum() > 1e-9:
                P_final /= P_final.sum()
                ratio_final = compute_ratio(P_final)
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
