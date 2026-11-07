"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Derivative-Free Trust-Region with global random restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 300.0  # 300s wall-clock limit with margin

        best_x = None
        best_f = np.inf

        def safe_func(x):
            nonlocal best_x, best_f
            x = np.clip(x, -5.0, 5.0)
            try:
                fx = float(func(x))
            except Exception:
                return np.inf
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def trust_region_search(x0, f0, max_evals):
            """Local DFTR search around x0."""
            nonlocal budget, deadline
            x = x0.copy()
            fx = f0
            dim = self.dim
            rho = 1.0  # trust region radius
            min_rho = 1e-6
            max_rho = 2.0

            for _ in range(max_evals):
                if time.time() > deadline - 5.0:
                    break

                # Build local quadratic model using sparse finite differences
                # Gradient estimation
                grad = np.zeros(dim)
                for i in range(dim):
                    e_i = np.zeros(dim)
                    e_i[i] = rho
                    x_plus = np.clip(x + e_i, -5, 5)
                    x_minus = np.clip(x - e_i, -5, 5)
                    fp = safe_func(x_plus)
                    fm = safe_func(x_minus)
                    if fp == np.inf or fm == np.inf:
                        grad[i] = 0.0
                    else:
                        grad[i] = (fp - fm) / (2.0 * rho)

                # Hessian estimation (diagonal only for efficiency)
                hess_diag = np.zeros(dim)
                for i in range(dim):
                    e_i = np.zeros(dim)
                    e_i[i] = rho
                    x_plus = np.clip(x + e_i, -5, 5)
                    x_minus = np.clip(x - e_i, -5, 5)
                    # We already have fp and fm from above, but let's recompute for clarity
                    # Actually, let's use a smaller step for Hessian
                    h_step = max(rho * 0.1, 1e-4)
                    e_h = np.zeros(dim)
                    e_h[i] = h_step
                    x_plus_h = np.clip(x + e_h, -5, 5)
                    x_minus_h = np.clip(x - e_h, -5, 5)
                    f_plus_h = safe_func(x_plus_h)
                    f_minus_h = safe_func(x_minus_h)
                    if f_plus_h == np.inf or f_minus_h == np.inf:
                        hess_diag[i] = 1.0  # default
                    else:
                        hess_diag[i] = (f_plus_h - 2.0 * fx + f_minus_h) / (h_step ** 2)
                        if hess_diag[i] < 1e-8:
                            hess_diag[i] = 1e-8

                # Solve trust region subproblem: min 0.5*d'H*d + g'd s.t. ||d|| <= rho
                # Using diagonal Hessian approximation
                H = np.diag(hess_diag)
                # Regularize to ensure positive definiteness
                H += 1e-4 * np.eye(dim)

                # Solve (H + lambda*I) d = -g for optimal lambda such that ||d|| <= rho
                # Use bisection on lambda
                g = grad
                if np.linalg.norm(g) < 1e-10:
                    # At a critical point, expand trust region
                    rho = min(rho * 2.0, max_rho)
                    continue

                lo, hi = 0.0, 1e6
                for _ in range(50):
                    lam = (lo + hi) / 2.0
                    A = H + lam * np.eye(dim)
                    try:
                        d = np.linalg.solve(A, -g)
                    except np.linalg.LinAlgError:
                        d = -g / (np.linalg.norm(g) + 1e-10)
                    if np.linalg.norm(d) > rho:
                        lo = lam
                    else:
                        hi = lam

                lam = (lo + hi) / 2.0
                A = H + lam * np.eye(dim)
                try:
                    d = np.linalg.solve(A, -g)
                except np.linalg.LinAlgError:
                    d = -g / (np.linalg.norm(g) + 1e-10)

                # Ensure step is within trust region
                d_norm = np.linalg.norm(d)
                if d_norm > rho:
                    d = d * (rho / d_norm)

                x_new = np.clip(x + d, -5, 5)
                f_new = safe_func(x_new)

                if f_new == np.inf:
                    rho = max(rho * 0.5, min_rho)
                    continue

                # Predicted reduction
                pred_reduction = -0.5 * d @ H @ d - g @ d
                actual_reduction = fx - f_new

                if pred_reduction > 1e-12:
                    ratio = actual_reduction / pred_reduction
                else:
                    ratio = 0.0

                if ratio > 0.75:
                    # Good step, expand trust region
                    rho = min(rho * 1.5, max_rho)
                    x = x_new
                    fx = f_new
                elif ratio > 0.25:
                    # Accept step, keep trust region
                    x = x_new
                    fx = f_new
                else:
                    # Reject step, shrink trust region
                    rho = max(rho * 0.5, min_rho)

            return x, fx

        # Global search strategy with restarts
        n_restarts = 8
        evals_per_restart = budget // n_restarts
        evals_per_restart = max(evals_per_restart, 10)

        for restart in range(n_restarts):
            if time.time() > deadline - 10.0:
                break

            # Initialize with random point, biased towards center
            if restart == 0:
                x0 = rng.uniform(-3, 3, dim)
            else:
                # Mix of random and perturbation from best
                if rng.random() < 0.3 and best_x is not None:
                    x0 = best_x + rng.normal(0, 1.0, dim)
                else:
                    x0 = rng.uniform(-5, 5, dim)
            x0 = np.clip(x0, -5, 5)
            f0 = safe_func(x0)

            x_local, f_local = trust_region_search(x0, f0, evals_per_restart)

        return best_x, best_f
# EVOLVE-BLOCK-END
