"""Optimizer using CMA-ES with multi-start strategy for Katsuura f23."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start CMA-ES with adaptive budget allocation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Track best solution found
        best_x = None
        best_f = np.inf

        # We'll use a multi-start strategy with CMA-ES
        # Allocate budget: use ~150 evaluations for exploration, rest for exploitation

        # Phase 1: Random sampling to get initial diversity (50 evals)
        n_random = min(50, budget)
        X_samples = []
        F_samples = []

        for i in range(n_random):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            X_samples.append(x)
            F_samples.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        X_samples = np.array(X_samples)
        F_samples = np.array(F_samples)

        # Phase 2: Multi-start CMA-ES from top diverse points
        # Select top-k candidates with diversity constraint
        k_starts = min(5, n_random)
        idx_sorted = np.argsort(F_samples)

        # Ensure diversity: pick from different regions
        selected_indices = []
        for idx in idx_sorted:
            x_candidate = X_samples[idx]
            if len(selected_indices) == 0:
                selected_indices.append(idx)
            else:
                # Check minimum distance from already selected
                min_dist = np.inf
                for sel_idx in selected_indices:
                    dist = np.linalg.norm(x_candidate - X_samples[sel_idx])
                    min_dist = min(min_dist, dist)
                if min_dist > 1.5:  # Require some diversity
                    selected_indices.append(idx)
            if len(selected_indices) >= k_starts:
                break

        # If not enough diverse points, add more from sorted list
        for idx in idx_sorted:
            if len(selected_indices) >= k_starts:
                break
            if idx not in selected_indices:
                selected_indices.append(idx)

        # Allocate remaining budget among starts
        remaining_budget = budget - n_random
        evals_per_start = max(10, remaining_budget // len(selected_indices))

        # For each start, run CMA-ES with limited budget
        for start_idx in selected_indices:
            if time.time() > deadline - 30:  # Check time
                break

            x0 = X_samples[start_idx].copy()
            f0 = F_samples[start_idx]

            # Use scipy's CMA-ES via minimize with method='CMA' if available,
            # otherwise implement a simple CMA-ES
            try:
                # Try scipy CMA-ES
                from scipy.optimize import _minimize_cmaes

                # Define callback to count evaluations and check time
                eval_count = [0]
                stop_flag = [False]

                def callback(xk):
                    eval_count[0] += 1
                    fx = func(xk)
                    if fx < best_f:
                        nonlocal best_f, best_x
                        best_f = fx
                        best_x = xk.copy()
                    if eval_count[0] >= evals_per_start:
                        stop_flag[0] = True
                    if time.time() > deadline - 10:
                        stop_flag[0] = True
                    return stop_flag[0]

                # CMA-ES options
                options = {
                    'maxiter': evals_per_start,
                    'popsize': max(4, dim + 2),
                    'seed': start_idx,
                }

                res = _minimize_cmaes(
                    lambda x: func(x),
                    x0,
                    method='CMA',
                    options=options,
                    callback=callback
                )

                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()

            except Exception:
                # Fallback: simple (1+1)-ES from this start
                x = x0.copy()
                fx = f0
                sigma = 1.0
                for i in range(evals_per_start):
                    if time.time() > deadline - 5:
                        break
                    y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                    fy = func(y)
                    if fy < fx:
                        x, fx = y, fy
                        sigma *= 1.3
                        if fy < best_f:
                            best_f = fy
                            best_x = x.copy()
                    else:
                        sigma *= 0.85

        # Phase 3: Final polish with Nelder-Mead or L-BFGS from best point
        if best_x is not None and time.time() < deadline - 20:
            try:
                # Nelder-Mead polish
                result = minimize(
                    func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

            # Try L-BFGS-B as well
            if time.time() < deadline - 10:
                try:
                    result2 = minimize(
                        func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 30, 'ftol': 1e-10}
                    )
                    if result2.fun < best_f:
                        best_f = result2.fun
                        best_x = result2.x.copy()
                except Exception:
                    pass

        # Ensure we have a valid result
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
