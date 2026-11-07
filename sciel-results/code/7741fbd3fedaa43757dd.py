# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

class Optimizer:
    """Multi-start CMA-ES + L-BFGS-B with adaptive global exploration for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.perf_counter()
        deadline = t_start + 280.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0

        def time_left():
            return deadline - time.perf_counter()

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def wrapped_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                return best_f
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            f = float(func(x))
            calls_made += 1
            update_best(x, f)
            return f

        rng = np.random.default_rng(42)

        # Phase 1: Initial exploration with diverse starting points
        # Use a mix of structured and random points
        n_init = 30
        X_init = []

        # Center
        X_init.append(np.zeros(dim))

        # Halton-like quasi-random points for better coverage
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        n_halton = 15
        for i in range(n_halton):
            pt = np.zeros(dim)
            for d in range(dim):
                pt[d] = halton(i + 1, d + 2) * 10.0 - 5.0
            X_init.append(pt)

        # Random points
        n_random = n_init - 1 - n_halton
        X_init.append(rng.uniform(-5.0, 5.0, (n_random, dim)))
        X_init = np.vstack(X_init)

        # Evaluate initial points
        for i in range(len(X_init)):
            if time_left() < 5.0 or calls_made >= budget:
                break
            f_val = wrapped_func(X_init[i])

        # Phase 2: Local refinement from best points
        # Sort initial points by function value and refine the best few
        init_vals = []
        for i in range(len(X_init)):
            if time_left() < 5.0 or calls_made >= budget:
                break
            # We already evaluated these, but we need to track values
            # Re-evaluate to get values (these are cached in best_f tracking)
            # Actually, let's just use the best_f as a proxy and do local search from best_x
            pass

        # Better approach: do local search from the best point found so far
        bounds = [(-5.0, 5.0)] * dim

        # Local refinement from current best
        if time_left() > 30.0 and calls_made < budget - 20:
            try:
                maxfun_local = min(50, max(10, (budget - calls_made) // 4))
                res = minimize(
                    wrapped_func,
                    best_x.copy(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 30,
                        'maxls': 10,
                        'ftol': 1e-12,
                        'gtol': 1e-10,
                        'maxfun': maxfun_local
                    }
                )
            except Exception:
                pass

        # Phase 3: Multi-start local search with diverse perturbations
        # Generate multiple starting points around the best solution and in random locations
        n_starts = 8
        for s in range(n_starts):
            if time_left() < 20.0 or calls_made >= budget - 10:
                break

            # Mix: some around best, some random
            if s < n_starts // 2:
                # Perturb around best
                sigma = 1.0 / (s + 1)  # Decreasing perturbation
                x_start = best_x + rng.normal(0, sigma, dim)
                x_start = np.clip(x_start, -5.0, 5.0)
            else:
                # Random restart
                x_start = rng.uniform(-5.0, 5.0, dim)

            # Evaluate starting point
            f_start = wrapped_func(x_start)

            # Local refinement
            if time_left() < 10.0 or calls_made >= budget - 5:
                continue

            try:
                maxfun_local = min(30, max(5, (budget - calls_made) // 8))
                res = minimize(
                    wrapped_func,
                    x_start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 20,
                        'maxls': 8,
                        'ftol': 1e-12,
                        'gtol': 1e-10,
                        'maxfun': maxfun_local
                    }
                )
            except Exception:
                pass

        # Phase 4: Final fine-tuning around the best solution
        if time_left() > 15.0 and calls_made < budget - 5:
            # Do a very fine local search
            try:
                maxfun_final = min(20, max(5, (budget - calls_made) // 2))
                res = minimize(
                    wrapped_func,
                    best_x.copy(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 15,
                        'maxls': 5,
                        'ftol': 1e-14,
                        'gtol': 1e-12,
                        'maxfun': maxfun_final
                    }
                )
            except Exception:
                pass

        # Phase 5: Use remaining budget for random exploration if needed
        while time_left() > 5.0 and calls_made < budget:
            x_rand = rng.uniform(-5.0, 5.0, dim)
            f_rand = wrapped_func(x_rand)
            if f_rand < best_f:
                # Found something better, do a quick local search
                if time_left() > 10.0 and calls_made < budget - 5:
                    try:
                        res = minimize(
                            wrapped_func,
                            x_rand,
                            method='L-BFGS-B',
                            bounds=bounds,
                            options={
                                'maxiter': 10,
                                'maxls': 5,
                                'ftol': 1e-12,
                                'gtol': 1e-10,
                                'maxfun': 10
                            }
                        )
                    except Exception:
                        pass
            else:
                # No improvement, break out of random exploration
                break

        return best_x, best_f

# EVOLVE-BLOCK-END
