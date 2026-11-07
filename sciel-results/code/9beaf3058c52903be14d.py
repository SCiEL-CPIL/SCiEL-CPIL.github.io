"""Rewritten optimizer: LHS global basin search + Nelder-Mead local exploit."""
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Latin Hypercube global search followed by Nelder-Mead local exploitation."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        global_x = None
        global_fx = np.inf
        calls_used = 0

        def check_deadline():
            return time.time() > deadline

        def count_call():
            nonlocal calls_used
            calls_used += 1

        # --- Phase 1: Dense Latin Hypercube Sampling to identify basins ---
        # Use a good fraction of budget for global exploration
        n_lhs = min(budget // 3, 80)  # up to 80 LHS points, but not more than 1/3 budget
        if n_lhs < 10:
            n_lhs = 10

        # Generate Latin Hypercube points in [-5, 5]^dim
        rng = np.random.default_rng(12345)
        lhs_points = np.zeros((n_lhs, dim))
        for d in range(dim):
            # Create permuted stratified samples
            strata = np.linspace(0, 1, n_lhs, endpoint=False)
            perm = rng.permutation(n_lhs)
            # Within each stratum, uniform random
            offsets = rng.uniform(0, 1, n_lhs)
            lhs_points[:, d] = -5 + 10 * (strata[perm] + offsets[perm]) / n_lhs * n_lhs
            # Simplified: just use stratified random
            lhs_points[:, d] = -5 + 10 * (perm + offsets) / n_lhs

        # Clip to valid range
        lhs_points = np.clip(lhs_points, -5, 5)

        # Evaluate all LHS points
        for i in range(n_lhs):
            if calls_used >= budget or check_deadline():
                break
            x = lhs_points[i]
            fx = func(x)
            count_call()
            if fx < global_fx:
                global_fx = fx
                global_x = x.copy()

        # --- Phase 2: Nelder-Mead from the best LHS point ---
        if global_x is not None and not check_deadline() and calls_used < budget:
            # Try Nelder-Mead from the best point
            nm_start = global_x.copy()
            nm_fx = global_fx

            # Nelder-Mead with tight tolerances
            nm_result = minimize(
                func,
                nm_start,
                method='Nelder-Mead',
                options={
                    'maxiter': min(budget - calls_used, 200),
                    'maxfev': min(budget - calls_used, 200),
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True,
                }
            )

            # Count the function calls made by Nelder-Mead
            nm_calls = nm_result.get('nfev', 0)
            calls_used += nm_calls

            if nm_result.fun < global_fx:
                global_fx = nm_result.fun
                global_x = nm_result.x.copy()

        # --- Phase 3: If budget remains, try Nelder-Mead from second-best LHS point ---
        if not check_deadline() and calls_used < budget and global_x is not None:
            # Find second best LHS point (not the same as global best)
            second_best_idx = None
            second_best_fx = np.inf
            for i in range(n_lhs):
                if calls_used >= budget or check_deadline():
                    break
                # Skip if this is essentially the same point as global best
                if np.linalg.norm(lhs_points[i] - global_x) < 0.1:
                    continue
                # Re-evaluate to get current value (or use stored)
                # We didn't store all values, so re-evaluate only top candidates
                # Actually, let's just do another NM from a perturbed global best
                break

            # Alternative: do a second Nelder-Mead from a perturbed version of global best
            if calls_used < budget and not check_deadline():
                rng2 = np.random.default_rng(999)
                perturbation = 0.5 * rng2.standard_normal(dim)
                nm_start2 = np.clip(global_x + perturbation, -5, 5)

                nm_result2 = minimize(
                    func,
                    nm_start2,
                    method='Nelder-Mead',
                    options={
                        'maxiter': min(budget - calls_used, 100),
                        'maxfev': min(budget - calls_used, 100),
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True,
                    }
                )

                nm_calls2 = nm_result2.get('nfev', 0)
                calls_used += nm_calls2

                if nm_result2.fun < global_fx:
                    global_fx = nm_result2.fun
                    global_x = nm_result2.x.copy()

        # --- Phase 4: Fine polish with small random steps if budget remains ---
        if not check_deadline() and calls_used < budget and global_x is not None:
            rng_polish = np.random.default_rng(77777)
            x = global_x.copy()
            fx = global_fx
            sigma = 0.1
            max_polish = min(budget - calls_used, 50)

            for _ in range(max_polish):
                if calls_used >= budget or check_deadline():
                    break

                y = np.clip(x + sigma * rng_polish.standard_normal(dim), -5, 5)
                fy = func(y)
                count_call()

                if fy < fx:
                    x, fx = y, fy
                    if fx < global_fx:
                        global_fx = fx
                        global_x = x.copy()
                        sigma *= 0.8  # shrink step on improvement
                else:
                    sigma *= 0.95  # gentle shrink on failure

                sigma = max(sigma, 0.001)

        return global_x, global_fx
