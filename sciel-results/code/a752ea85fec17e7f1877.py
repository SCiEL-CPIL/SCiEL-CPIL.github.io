"""Sparse Grid Interpolation + Adaptive Refinement Optimizer."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import qmc


class Optimizer:
    """Kriging-based surrogate with adaptive uncertainty minimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        n = dim

        # --- Phase 1: Coarse Sobol initialization ---
        n_init = 50
        sobol = qmc.Sobol(d=n, scramble=True, seed=42)
        points_unit = sobol.random(n_init)
        X = -5.0 + 10.0 * points_unit
        Y = np.array([func(x) for x in X])
        
        calls_made = n_init
        best_idx = np.argmin(Y)
        best_x = X[best_idx].copy()
        best_y = Y[best_idx]

        # Pre-generate candidate pool for acquisition
        sobol_pool = qmc.Sobol(d=n, scramble=True, seed=123)
        pool_size = 2000
        pool_unit = sobol_pool.random(pool_size)
        pool = -5.0 + 10.0 * pool_unit
        pool_idx = 0

        # Kriging hyperparameters
        nu = 1.5  # Matern smoothness
        # Estimate initial range and variance from data
        y_mean = np.mean(Y)
        y_var = np.var(Y)
        if y_var < 1e-10:
            y_var = 1e-10

        # Estimate length scale from pairwise distances
        if len(X) > 1:
            # Use median distance as initial range estimate
            X2 = np.sum(X**2, axis=1)
            D2 = X2[:, None] + X2[None, :] - 2 * X @ X.T
            D2 = np.maximum(D2, 0)
            D = np.sqrt(D2)
            np.fill_diagonal(D, np.inf)
            median_dist = np.median(D)
            if median_dist > 1e-6:
                l0 = median_dist / 3.0  # Heuristic: range ~ 3x median dist
            else:
                l0 = 1.0
        else:
            l0 = 1.0

        l0 = np.clip(l0, 0.1, 5.0)

        # --- Main Loop ---
        while calls_made < budget:
            # Update kriging parameters based on current data
            y_mean = np.mean(Y)
            y_var = np.var(Y)
            if y_var < 1e-10:
                y_var = 1e-10

            # Recompute length scale
            if len(X) > 2:
                X2 = np.sum(X**2, axis=1)
                D2 = X2[:, None] + X2[None, :] - 2 * X @ X.T
                D2 = np.maximum(D2, 0)
                D = np.sqrt(D2)
                np.fill_diagonal(D, np.inf)
                # Use a quantile to be robust
                median_dist = np.median(D)
                if median_dist > 1e-6:
                    l0 = median_dist / 3.0
                else:
                    l0 = 1.0
            l0 = np.clip(l0, 0.1, 5.0)

            # Build kriging covariance matrix
            n_train = len(X)
            X2 = np.sum(X**2, axis=1)
            D2_train = X2[:, None] + X2[None, :] - 2 * X @ X.T
            D2_train = np.maximum(D2_train, 0)
            D_train = np.sqrt(D2_train)

            # Matern kernel
            def matern_kernel(dist, nu=nu, l=l0, v=y_var):
                """Matern covariance function."""
                dist = np.maximum(dist, 1e-10)
                if nu == 1.5:
                    # Closed form for nu=1.5
                    z = np.sqrt(3.0) * dist / l
                    return v * (1.0 + z) * np.exp(-z)
                elif nu == 0.5:
                    return v * np.exp(-dist / l)
                else:
                    # General Matern using scipy
                    from scipy.special import kv
                    z = np.sqrt(2 * nu) * dist / l
                    with np.errstate(invalid='ignore', divide='ignore'):
                        kern = (1.0 / kv(nu, 0)) * (z ** nu) * kv(nu, z)
                    kern = np.nan_to_num(kern, nan=0.0)
                    return v * kern

            C = matern_kernel(D_train, nu=nu, l=l0, v=y_var)
            # Add small jitter for numerical stability
            C[np.diag_indices_from(C)] += 1e-8

            # Solve kriging system: K * beta = y_centered
            y_centered = Y - y_mean
            try:
                beta = np.linalg.solve(C, y_centered)
            except np.linalg.LinAlgError:
                beta = np.linalg.lstsq(C, y_centered, rcond=None)[0]

            # Generate candidates: mix of global pool and local around best
            progress = calls_made / budget
            n_global = 80
            n_local = 70

            # Global candidates from pool
            if pool_idx + n_global > pool_size:
                new_pool_unit = sobol_pool.random(pool_size)
                pool = -5.0 + 10.0 * new_pool_unit
                pool_idx = 0

            g_cands = pool[pool_idx:pool_idx + n_global]
            pool_idx += n_global

            # Local candidates around best point
            step_size = 1.5 * (1.0 - progress) + 0.3
            rng = np.random.default_rng(calls_made)
            l_cands = best_x + step_size * rng.standard_normal((n_local, n))
            l_cands = np.clip(l_cands, -5, 5)

            candidates = np.vstack([g_cands, l_cands])
            n_cands = len(candidates)

            # Compute kriging prediction and variance for candidates
            C2 = np.sum(candidates**2, axis=1)
            D2_cand = C2[:, None] + X2[None, :] - 2 * candidates @ X.T
            D2_cand = np.maximum(D2_cand, 0)
            D_cand = np.sqrt(D2_cand)

            # Cross-covariance vector for each candidate
            f = matern_kernel(D_cand, nu=nu, l=l0, v=y_var)  # (n_cands, n_train)

            # Kriging mean: mu = y_mean + f @ beta
            mu = y_mean + f @ beta

            # Kriging variance: sigma2 = v - f @ solve(C, f.T)
            # Compute C_inv_f = solve(C, f.T) => shape (n_train, n_cands)
            try:
                C_inv_f = np.linalg.solve(C, f.T)  # (n_train, n_cands)
            except np.linalg.LinAlgError:
                C_inv_f = np.linalg.lstsq(C, f.T, rcond=None)[0]

            var_krig = y_var - np.sum(f * C_inv_f.T, axis=1)
            var_krig = np.maximum(var_krig, 1e-10)
            std_krig = np.sqrt(var_krig)

            # Acquisition: minimize expected improvement (EI) for minimization
            # EI = (best_y - mu) * Phi((best_y - mu)/std) + std * phi((best_y - mu)/std)
            # For minimization, we want points where mu is low AND std is high
            # Use LCB with adaptive beta
            beta_acq = 2.5 * (1.0 - progress) + 0.5
            acq = mu - beta_acq * std_krig

            # Select best candidate
            best_cand_idx = np.argmin(acq)
            x_new = candidates[best_cand_idx]

            # Evaluate
            y_new = func(x_new)
            calls_made += 1

            # Update data
            X = np.vstack([X, x_new])
            Y = np.append(Y, y_new)

            if y_new < best_y:
                best_y = y_new
                best_x = x_new.copy()

        return best_x, best_y
# EVOLVE-BLOCK-END
