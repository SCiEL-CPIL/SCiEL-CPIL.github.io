"""Multi-start Powell's method with dense stratified LHS for Lunacek bi-Rastrigin f24."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Powell with dense stratified Latin Hypercube initialization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._calls = 0
        self._deadline = None

    def __call__(self, func):
        self._calls = 0
        self._deadline = time.time() + 280.0  # safety margin under 300s

        dim = self.dim
        budget = self.budget
        calls = [0]

        def f(x):
            if calls[0] >= budget:
                return 1e12
            if time.time() > self._deadline:
                return 1e12
            calls[0] += 1
            val = func(np.asarray(x, dtype=float))
            return float(val)

        best_x = None
        best_f = np.inf

        # Dense stratified Latin Hypercube: 25 points for broad coverage
        n_lhs = 25
        rng = np.random.default_rng(42)
        lhs_points = []
        for i in range(n_lhs):
            point = np.zeros(dim)
            for d in range(dim):
                perm = rng.permutation(n_lhs)
                interval = perm[i]
                point[d] = -5.0 + (interval + rng.random()) * (10.0 / n_lhs)
            lhs_points.append(point)

        # Strategic points targeting known structure of Lunacek bi-Rastrigin
        # The function has two basins; one near origin, one shifted
        strategic_points = [
            np.zeros(dim),
            np.full(dim, 1.0),
            np.full(dim, -1.0),
            np.full(dim, 2.0),
            np.full(dim, -2.0),
            np.full(dim, 3.0),
            np.full(dim, -3.0),
            np.full(dim, 0.5),
            np.full(dim, -0.5),
            np.full(dim, 4.0),
            np.full(dim, -4.0),
        ]

        all_starts = lhs_points + strategic_points

        # Phase 1: Evaluate all starting points to find the best basin
        for x0 in all_starts:
            if calls[0] >= budget or time.time() > self._deadline:
                break
            fx0 = f(x0)
            if fx0 < best_f:
                best_f = fx0
                best_x = x0.copy()

        # Phase 2: Multi-start Powell from top-k starting points
        # Powell is efficient for smooth, moderately separable functions like Lunacek
        # Run Powell from the best few starts, allocating budget based on remaining
        if best_x is not None and calls[0] < budget and time.time() < self._deadline:
            # Sort starts by their initial evaluation (we only stored the best,
            # so we re-evaluate a subset to rank them)
            # Actually, let's just run Powell from the best point first,
            # then from a few other promising points if budget allows
            
            # First: Powell from the best point with generous budget
            remaining = budget - calls[0]
            powell_budget = min(100, remaining)
            try:
                res = minimize(
                    f,
                    best_x,
                    method='Powell',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 50, 'maxfev': powell_budget, 'xtol': 1e-6, 'ftol': 1e-8}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 3: Powell from other promising starting points
        # Use a few more starts with smaller budgets each
        if calls[0] < budget and time.time() < self._deadline:
            # Pick a few diverse starting points
            extra_starts = []
            # Add some points near the current best but perturbed
            if best_x is not None:
                for scale in [0.5, 1.0, 2.0]:
                    perturbation = rng.normal(0, scale, dim)
                    x_pert = np.clip(best_x + perturbation, -5, 5)
                    extra_starts.append(x_pert)
            
            # Also add some from the original LHS that weren't the best
            for pt in lhs_points[:5]:
                extra_starts.append(pt)
            
            for x0 in extra_starts:
                if calls[0] >= budget or time.time() > self._deadline:
                    break
                # Small budget for each extra start
                small_budget = min(20, budget - calls[0])
                if small_budget < 5:
                    break
                try:
                    res = minimize(
                        f,
                        x0,
                        method='Powell',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 15, 'maxfev': small_budget, 'xtol': 1e-5, 'ftol': 1e-7}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass

        # Phase 4: Final polish with L-BFGS-B if budget remains
        if calls[0] < budget and time.time() < self._deadline and best_x is not None:
            remaining = budget - calls[0]
            if remaining > 10:
                try:
                    res = minimize(
                        f,
                        best_x,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 30, 'maxfun': min(30, remaining), 'ftol': 1e-10}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass

        # Phase 5: If significant budget remains, do a few more random restarts with Powell
        remaining_budget = budget - calls[0]
        if remaining_budget > 20 and time.time() < self._deadline:
            n_extra = min(3, remaining_budget // 20)
            for i in range(n_extra):
                if calls[0] >= budget or time.time() > self._deadline:
                    break
                x_rand = rng.uniform(-4, 4, dim)
                fx_rand = f(x_rand)
                if fx_rand < best_f:
                    best_f = fx_rand
                    best_x = x_rand.copy()
                if calls[0] < budget and time.time() < self._deadline:
                    try:
                        res = minimize(
                            f,
                            x_rand,
                            method='Powell',
                            bounds=[(-5, 5)] * dim,
                            options={'maxiter': 10, 'maxfev': min(15, budget - calls[0]), 'xtol': 1e-5, 'ftol': 1e-7}
                        )
                        if res.fun < best_f:
                            best_f = res.fun
                            best_x = res.x.copy()
                    except Exception:
                        pass

        return best_x, best_f
# EVOLVE-BLOCK-END
