import numpy as np
import time
from scipy.stats import qmc
from scipy.optimize import minimize
from scipy.interpolate import RBFInterpolator

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        call_count = [0]
        best_f = float('inf')
        best_x = None
        lower = np.full(dim, -5.0)
        upper = np.full(dim, 5.0)
        bounds = [(-5.0, 5.0)] * dim

        # Store all evaluated points
        X_all = []
        F_all = []

        def wrapped_func(x):
            nonlocal best_f, best_x
            if call_count[0] >= budget or time.time() > deadline:
                return 1e10
            x = np.asarray(x, dtype=np.float64).reshape(-1)
            if len(x) != dim:
                x = x[:dim]
            x = np.clip(x, lower, upper)

            val = func(x)
            if not np.isfinite(val):
                val = 1e10

            call_count[0] += 1
            X_all.append(x.copy())
            F_all.append(val)

            if val < best_f:
                best_f = val
                best_x = x.copy()

            return val

        def polish(x_start, maxfev=30):
            nonlocal best_f, best_x
            if call_count[0] >= budget or time.time() > deadline:
                return
            try:
                res = minimize(
                    wrapped_func,
                    x_start.copy(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 20,
                        'maxfev': maxfev,
                        'ftol': 1e-15,
                        'gtol': 1e-12,
                        'disp': False
                    }
                )
            except Exception:
                pass

        # Phase 1: Initial exploration with Sobol + random
        n_sobol = 32
        n_random = 18
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        sobol_pts = sampler.random(n_sobol) * 10.0 - 5.0
        random_pts = rng.uniform(-5, 5, (n_random, dim))
        center_pt = np.zeros(dim).reshape(1, -1)

        all_starts = np.vstack([sobol_pts, random_pts, center_pt])
        for pt in all_starts:
            if call_count[0] >= budget or time.time() > deadline:
                break
            wrapped_func(pt)

        if best_x is None:
            return np.zeros(dim), best_f

        # Phase 2: GP-based Bayesian Optimization loop
        # Use RBF interpolation as a lightweight surrogate (avoiding heavy GP fitting)
        # We'll use a simple Kriging-like approach with RBF + local regression

        n_bo_iter = 20  # Number of BO iterations
        n_candidates_per_iter = 8  # Candidates to evaluate per BO iteration

        for iter_i in range(n_bo_iter):
            if call_count[0] >= budget or time.time() > deadline:
                break

            if len(X_all) < 3:
                # Not enough data for surrogate, do random
                for _ in range(n_candidates_per_iter):
                    if call_count[0] >= budget or time.time() > deadline:
                        break
                    x_try = rng.uniform(-5, 5, dim)
                    wrapped_func(x_try)
                continue

            X = np.array(X_all)
            F = np.array(F_all)

            # Normalize F for better conditioning
            f_min = F.min()
            f_range = F.max() - f_min
            if f_range < 1e-10:
                f_range = 1.0
            F_norm = (F - f_min) / f_range

            # Build RBF surrogate
            try:
                # Use RBF interpolation with thin plate spline kernel
                rbf = RBFInterpolator(X, F_norm, kernel='thin_plate_spline',
                                       smoothing=0.1)
            except Exception:
                try:
                    rbf = RBFInterpolator(X, F_norm, kernel='gaussian',
                                           smoothing=0.1)
                except Exception:
                    # Fallback: just pick random points near best
                    for _ in range(n_candidates_per_iter):
                        if call_count[0] >= budget or time.time() > deadline:
                            break
                        step = 1.0 * (1.0 - iter_i / n_bo_iter)
                        x_try = best_x + step * rng.standard_normal(dim)
                        x_try = np.clip(x_try, lower, upper)
                        wrapped_func(x_try)
                    continue

            # Generate candidate points using EI acquisition
            # We'll sample candidates from a mixture:
            # 1. Random in the box
            # 2. Points near current best
            # 3. Points in promising regions (low predicted F)

            n_candidates = n_candidates_per_iter * 3  # Generate more, pick best

            cand_list = []

            # Random candidates
            n_rand = n_candidates // 3
            cand_rand = rng.uniform(-5, 5, (n_rand, dim))
            cand_list.append(cand_rand)

            # Candidates near best point with decreasing radius
            n_near = n_candidates // 3
            radii = np.linspace(3.0, 0.3, n_near)
            cand_near = []
            for r in radii:
                direction = rng.standard_normal(dim)
                direction /= np.linalg.norm(direction)
                x_cand = best_x + r * direction
                x_cand = np.clip(x_cand, lower, upper)
                cand_near.append(x_cand)
            cand_list.append(np.array(cand_near))

            # Candidates in regions where surrogate predicts low values
            # Sample from a grid and pick low-prediction areas
            n_grid = n_candidates // 3
            # Use Latin hypercube for diverse sampling
            sampler2 = qmc.LatinHypercube(d=dim, seed=rng.integers(10000))
            cand_lh = sampler2.random(n_grid) * 10.0 - 5.0
            cand_list.append(cand_lh)

            all_cands = np.vstack(cand_list)
            all_cands = np.clip(all_cands, lower, upper)

            # Evaluate surrogate predictions and compute EI
            # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
            # For RBF, we don't have explicit sigma, so we approximate
            # using prediction variance from residual analysis

            try:
                pred_F = rbf(all_cands)
                # Approximate uncertainty: use distance to nearest training points
                # Closer points = lower uncertainty
                from scipy.spatial import cKDTree
                tree = cKDTree(X)
                dists, _ = tree.query(all_cands, k=1)
                # Normalize distances
                max_dist = np.sqrt(dim) * 10.0  # max possible distance in box
                dist_norm = np.clip(dists / max_dist, 0, 1)
                # Uncertainty estimate: higher when far from data
                sigma_est = 0.5 * dist_norm + 0.1

                f_best_norm = (best_f - f_min) / f_range
                z = (f_best_norm - pred_F) / np.maximum(sigma_est, 1e-8)

                # Expected Improvement
                from scipy.stats import norm
                EI = (f_best_norm - pred_F) * norm.cdf(z) + sigma_est * norm.pdf(z)
            except Exception:
                # Fallback: just use distance-based exploration
                EI = -np.random.randn(len(all_cands)) * 0.1

            # Select top candidates by EI
            top_idx = np.argsort(-EI)[:n_candidates_per_iter]
            selected_cands = all_cands[top_idx]

            # Evaluate selected candidates
            for x_cand in selected_cands:
                if call_count[0] >= budget or time.time() > deadline:
                    break
                wrapped_func(x_cand)

            # After each BO iteration, do a quick polish on the new best
            if call_count[0] < budget and time.time() < deadline and best_x is not None:
                polish(best_x, maxfev=min(15, budget - call_count[0]))

        # Phase 3: Final intensive polish
        if call_count[0] < budget and time.time() < deadline and best_x is not None:
            polish(best_x, maxfev=budget - call_count[0])

        if best_x is None:
            best_x = np.zeros(dim)
            if call_count[0] < budget:
                wrapped_func(best_x)

        return best_x, best_f
