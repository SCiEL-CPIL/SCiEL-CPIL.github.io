# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B with diverse starting points and adaptive refinement for BBOB f14."""

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

        # Generate diverse starting points
        rng = np.random.default_rng(42)

        # Strategy: mix of structured and random starts
        starts = []

        # 1. Center
        starts.append(np.zeros(dim))

        # 2. Halton-like quasi-random points (low-discrepancy)
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        n_halton = 25
        for i in range(1, n_halton + 1):
            pt = np.array([halton(i, b) for b in range(2, dim + 2)])
            pt = pt * 10.0 - 5.0  # Map [0,1] to [-5,5]
            starts.append(pt)

        # 3. Random points
        n_random = 40
        for _ in range(n_random):
            starts.append(rng.uniform(-5.0, 5.0, dim))

        # 4. Axis-aligned points (for potential coordinate-wise structure)
        for i in range(dim):
            for val in [-4.0, -2.0, 0.0, 2.0, 4.0]:
                pt = np.zeros(dim)
                pt[i] = val
                starts.append(pt)

        # 5. Corner points
        for sign in [-1, 1]:
            for i in range(1, dim + 1):
                pt = np.zeros(dim)
                for j in range(i):
                    pt[j] = sign * 4.5
                starts.append(pt)

        # Shuffle to mix strategies
        rng.shuffle(starts)

        # Phase 1: Quick evaluation of all starts, store results
        evaluated_starts = []
        for x0 in starts:
            if time_left() < 5.0 or calls_made >= budget:
                break
            x0 = np.clip(x0, -5.0, 5.0)
            f0 = wrapped_func(x0)
            evaluated_starts.append((f0, x0.copy()))

        # Sort by function value (best first)
        evaluated_starts.sort(key=lambda item: item[0])

        # Phase 2: L-BFGS-B from the best starting points with aggressive settings
        bounds = [(-5.0, 5.0)] * dim
        n_refine = min(15, len(evaluated_starts))

        for idx, (f_cand, x_cand) in enumerate(evaluated_starts[:n_refine]):
            if time_left() < 10.0 or calls_made >= budget:
                break

            x_cand = np.clip(x_cand, -5.0, 5.0)

            # Allocate more iterations for better candidates
            if idx < 5:
                max_iter = 100
                maxfun = max(15, (budget - calls_made) // 4)
                ftol = 1e-14
                gtol = 1e-12
            elif idx < 10:
                max_iter = 70
                maxfun = max(10, (budget - calls_made) // 5)
                ftol = 1e-12
                gtol = 1e-10
            else:
                max_iter = 50
                maxfun = max(8, (budget - calls_made) // 6)
                ftol = 1e-10
                gtol = 1e-8

            try:
                res = minimize(
                    wrapped_func,
                    x_cand,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': max_iter,
                        'maxls': 20,
                        'ftol': ftol,
                        'gtol': gtol,
                        'maxfun': maxfun
                    }
                )
                if res.fun < best_f:
                    update_best(res.x, res.fun)
            except Exception:
                pass

        # Phase 3: If we still have budget, do more random restarts with L-BFGS-B
        restart_count = 0
        while time_left() > 20.0 and calls_made < budget - 30 and restart_count < 50:
            x0 = rng.uniform(-5.0, 5.0, dim)
            try:
                res = minimize(
                    wrapped_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 60,
                        'maxls': 15,
                        'ftol': 1e-12,
                        'gtol': 1e-10,
                        'maxfun': max(10, (budget - calls_made) // 8)
                    }
                )
                if res.fun < best_f:
                    update_best(res.x, res.fun)
            except Exception:
                pass
            restart_count += 1

        # Phase 4: Local search around the best point with small perturbations
        if calls_made < budget - 15 and time_left() > 10.0:
            # Try several small perturbations around the best point
            for scale in [0.5, 0.2, 0.1, 0.05]:
                if calls_made >= budget - 5 or time_left() < 5.0:
                    break
                for _ in range(5):
                    if calls_made >= budget - 5 or time_left() < 5.0:
                        break
                    perturbation = rng.normal(0, scale, dim)
                    x_pert = np.clip(best_x + perturbation, -5.0, 5.0)
                    try:
                        res = minimize(
                            wrapped_func,
                            x_pert,
                            method='L-BFGS-B',
                            bounds=bounds,
                            options={
                                'maxiter': 30,
                                'maxls': 10,
                                'ftol': 1e-13,
                                'gtol': 1e-11,
                                'maxfun': max(5, min(20, budget - calls_made - 5))
                            }
                        )
                        if res.fun < best_f:
                            update_best(res.x, res.fun)
                    except Exception:
                        pass

        # Final refinement: one more L-BFGS-B from the best point found
        if calls_made < budget - 10 and time_left() > 5.0:
            try:
                res = minimize(
                    wrapped_func,
                    best_x.copy(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 50,
                        'maxls': 20,
                        'ftol': 1e-15,
                        'gtol': 1e-13,
                        'maxfun': max(5, budget - calls_made - 5)
                    }
                )
                if res.fun < best_f:
                    update_best(res.x, res.fun)
            except Exception:
                pass

        return best_x, best_f

# EVOLVE-BLOCK-END
