"""BBOB f19 (Griewank-Rosenbrock) optimizer.

Strategy:
- Structured initialization informed by the Griewank-Rosenbrock structure.
- Powell's method for robust derivative-free local optimization.
- Nelder-Mead for fast convergence in smooth basins.
- Coordinate descent for fine-grained refinement.
- L-BFGS-B for high-precision final polish.
"""
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 275.0
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        call_count = 0

        def eval_point(x):
            nonlocal best_x, best_f, call_count
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            f = float(func(x))
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(12345)

        # Structured initialization exploiting the Griewank-Rosenbrock structure
        init_points = []

        # Core structured points
        for val in [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]:
            init_points.append(np.ones(dim) * val)

        # Diverse points around key centers
        for center in [0.3, 0.5, 0.7, 1.0]:
            for _ in range(3):
                offset = rng.normal(0, 0.3, dim)
                init_points.append(center * np.ones(dim) + offset)

        # Focused random points
        for _ in range(6):
            init_points.append(rng.uniform(-1, 3, dim))

        # A few fully random for coverage
        for _ in range(3):
            init_points.append(rng.uniform(-5, 5, dim))

        # Evaluate initial points
        for p in init_points:
            if call_count >= budget or time.time() >= deadline:
                break
            eval_point(p)

        if call_count >= budget or time.time() >= deadline:
            return best_x

        # Powell's method for robust local search
        def powell_search(x0, max_iter=50):
            nonlocal best_x, best_f, call_count
            if call_count >= budget or time.time() >= deadline:
                return

            try:
                def objective(x):
                    if call_count >= budget or time.time() >= deadline:
                        return best_f
                    f = float(func(x))
                    call_count += 1
                    if f < best_f:
                        best_f = f
                        best_x = np.clip(x, -5, 5).copy()
                    return f

                res = minimize(
                    objective,
                    np.clip(x0, -5, 5),
                    method='Powell',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': max_iter, 'ftol': 1e-10, 'xtol': 1e-8}
                )
            except Exception:
                pass

        # Run Powell from best point first
        powell_search(best_x.copy(), max_iter=80)

        # Run Powell from other promising starting points
        candidate_starts = [best_x.copy()]
        for center in [0.3, 0.5, 0.7, 1.0]:
            candidate_starts.append(center * np.ones(dim))

        for start in candidate_starts[1:]:
            if call_count >= budget or time.time() >= deadline:
                break
            if np.linalg.norm(start - best_x) > 0.5:
                powell_search(start, max_iter=40)

        # Multi-restart L-BFGS-B for exploiting gradient information in smooth landscape
        if call_count < budget and time.time() < deadline:
            # Generate diverse restart points around current best and known centers
            lbfgs_starts = [best_x.copy()]
            for center in [0.3, 0.5, 0.7, 1.0]:
                c = center * np.ones(dim)
                if np.linalg.norm(c - best_x) > 0.3:
                    lbfgs_starts.append(c)
            
            # Add some perturbed versions of best_x for diversity
            for scale in [0.5, 1.0, 2.0]:
                for _ in range(2):
                    perturbed = np.clip(best_x + scale * rng.standard_normal(dim), -5, 5)
                    if np.linalg.norm(perturbed - best_x) > 0.2:
                        lbfgs_starts.append(perturbed)

            for start in lbfgs_starts:
                if call_count >= budget or time.time() >= deadline:
                    break
                try:
                    def lbfgs_objective(x):
                        if call_count >= budget or time.time() >= deadline:
                            return best_f
                        x = np.clip(x, -5, 5)
                        f = float(func(x))
                        call_count += 1
                        if f < best_f:
                            best_f = f
                            best_x = x.copy()
                        return f

                    res = minimize(
                        lbfgs_objective,
                        np.clip(start, -5, 5),
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
                    )
                except Exception:
                    pass

        # Nelder-Mead for additional local refinement
        def nm_search(x0, max_iter=100):
            nonlocal best_x, best_f, call_count
            if call_count >= budget or time.time() >= deadline:
                return

            try:
                def objective(x):
                    if call_count >= budget or time.time() >= deadline:
                        return best_f
                    x = np.clip(x, -5, 5)
                    f = float(func(x))
                    call_count += 1
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                    return f

                res = minimize(
                    objective,
                    np.clip(x0, -5, 5),
                    method='Nelder-Mead',
                    options={'maxiter': max_iter, 'xatol': 0.001, 'fatol': 1e-8}
                )
            except Exception:
                pass

        if call_count < budget and time.time() < deadline:
            nm_search(best_x.copy(), max_iter=150)

        # Adaptive coordinate descent for final convergence
        if call_count < budget and time.time() < deadline:
            x = best_x.copy()
            f = best_f
            step_sizes = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001]

            for step in step_sizes:
                improved = True
                while improved and call_count < budget and time.time() < deadline:
                    improved = False
                    for c in range(dim):
                        if call_count >= budget or time.time() >= deadline:
                            break
                        for direction in [1, -1]:
                            x_trial = x.copy()
                            x_trial[c] += direction * step
                            x_trial = np.clip(x_trial, -5, 5)
                            f_trial = float(func(x_trial))
                            call_count += 1
                            if f_trial < f - 1e-10:
                                x = x_trial.copy()
                                f = f_trial
                                if f < best_f:
                                    best_f = f
                                    best_x = x.copy()
                                improved = True
                            elif f_trial < best_f:
                                best_f = f_trial
                                best_x = np.clip(x_trial, -5, 5).copy()

        # Final high-precision polish with L-BFGS-B - tighter tolerances for maximum precision
        if call_count < budget and time.time() < deadline:
            try:
                def polish_func(x):
                    if call_count >= budget or time.time() >= deadline:
                        return best_f
                    f = float(func(x))
                    call_count += 1
                    if f < best_f:
                        best_f = f
                        best_x = np.clip(x, -5, 5).copy()
                    return f

                res = minimize(
                    polish_func,
                    best_x.copy(),
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 200}
                )
            except Exception:
                pass

        # Last few random perturbations if budget remains
        if call_count < budget and time.time() < deadline:
            for scale in [0.1, 0.05, 0.01]:
                if call_count >= budget or time.time() >= deadline:
                    break
                x_try = np.clip(best_x + scale * rng.standard_normal(dim), -5, 5)
                eval_point(x_try)

        return best_x
