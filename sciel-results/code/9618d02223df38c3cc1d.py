"""Multi-objective optimizer on ZDT6 using Bayesian Optimization on the penalty term."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.utils import check_array


class Optimizer:
    """Bayesian Optimization on ZDT6 penalty g(x) to find low-g points."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim

        # Track all evaluated points and objectives
        X_all = []
        F_all = []
        eval_count = 0

        def evaluate(x):
            nonlocal eval_count
            if eval_count >= self.budget:
                return None
            x = np.clip(np.asarray(x, dtype=float), lo, hi)
            f = np.asarray(problem(x), dtype=float)
            eval_count += 1
            X_all.append(x.copy())
            F_all.append(f.copy())
            return f

        # Extract g from f1, f2
        def extract_g(f1, f2):
            if abs(f1) < 1e-8:
                return f2 / 2.0
            sin_f1 = np.sin(f1)
            if abs(sin_f1) < 1e-8:
                return f2
            ratio = f1 / sin_f1
            denom = 1.0 + ratio
            if abs(denom) < 1e-8:
                return f2
            return f2 / denom

        # Phase 1: Initial diverse sampling
        n_init = min(30, self.budget // 4)
        rng = np.random.default_rng(0)
        for _ in range(n_init):
            x = lo + (hi - lo) * rng.random(n)
            evaluate(x)

        # Prepare GP data
        X_g = np.array(X_all)
        F_g = np.array(F_all)
        g_vals = np.array([extract_g(f[0], f[1]) for f in F_g])

        # GP setup
        kernel = ConstantKernel(1.0) * Matern(length_scale=0.5, nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=3, normalize_y=True)

        def expected_improvement(X_new, y_min):
            X_new = check_array(X_new)
            mu, sigma = gp.predict(X_new, return_std=True)
            sigma = np.maximum(sigma, 1e-8)
            Z = (y_min - mu) / sigma
            EI = (y_min - mu) * norm.cdf(Z) + sigma * norm.pdf(Z)
            return EI

        def local_refine(x0, f0):
            best_x = x0.copy()
            best_g = extract_g(f0[0], f0[1])
            step_sizes = [0.05, 0.02, 0.01]
            for ss in step_sizes:
                for _ in range(5):
                    if eval_count >= self.budget or time.time() > deadline - 10:
                        return best_x, best_g
                    pert = rng.standard_normal(n) * ss
                    x_new = np.clip(best_x + pert, lo, hi)
                    f_new = evaluate(x_new)
                    if f_new is None:
                        return best_x, best_g
                    g_new = extract_g(f_new[0], f_new[1])
                    if g_new < best_g:
                        best_g = g_new
                        best_x = x_new.copy()
            return best_x, best_g

        # BO loop
        while eval_count < self.budget:
            if time.time() > deadline - 5:
                break

            # Refit GP
            if len(X_g) >= 5:
                try:
                    gp.fit(X_g, g_vals)
                except Exception:
                    break

            # Generate candidates
            n_cand = 50
            X_cand = lo + (hi - lo) * rng.random((n_cand, n))
            y_min = np.min(g_vals)
            EI = expected_improvement(X_cand, y_min)
            best_idx = int(np.argmax(EI))
            x_proposed = X_cand[best_idx]

            # Evaluate
            f_proposed = evaluate(x_proposed)
            if f_proposed is None:
                break
            g_proposed = extract_g(f_proposed[0], f_proposed[1])

            # Update GP data
            X_g = np.vstack([X_g, x_proposed])
            F_g = np.vstack([F_g, f_proposed])
            g_vals = np.append(g_vals, g_proposed)

            # Local refinement if promising
            if g_proposed < np.mean(g_vals) * 1.5:
                local_refine(x_proposed, f_proposed)

        return X_all
# EVOLVE-BLOCK-END
