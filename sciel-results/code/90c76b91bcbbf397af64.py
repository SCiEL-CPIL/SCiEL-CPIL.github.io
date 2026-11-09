"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start CMA-ES / Nelder-Mead hybrid for BBOB f22 (Gallagher 21 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        budget = self.budget
        dim = self.dim

        best_x = np.zeros(dim)
        best_f = float('inf')
        calls_made = 0

        def elapsed_ok():
            return (time.time() - start_time) < deadline

        def eval_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return None
            x_clipped = np.clip(x, -5.0, 5.0)
            calls_made += 1
            fx = func(x_clipped)
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        def try_improve(x, fx):
            nonlocal best_x, best_f
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 1: Dense random sampling to find promising basins
        rng = np.random.default_rng(12345)
        n_init = min(60, budget // 8)
        for i in range(n_init):
            if not elapsed_ok() or calls_made >= budget:
                break
            candidate = rng.uniform(-5, 5, dim)
            fc = eval_func(candidate)
            if fc is None:
                break
            try_improve(candidate, fc)

        # Phase 2: For each of the top diverse points, run a local optimizer
        # We collect diverse candidates from the initial sampling
        candidates = []
        for i in range(n_init):
            if not elapsed_ok() or calls_made >= budget:
                break
            candidate = rng.uniform(-5, 5, dim)
            fc = eval_func(candidate)
            if fc is None:
                break
            candidates.append((candidate, fc))
            try_improve(candidate, fc)

        # Sort by function value
        candidates.sort(key=lambda t: t[1])

        # Pick diverse top points
        diverse_starts = []
        min_dist = 2.0
        for cand, fval in candidates:
            if len(diverse_starts) >= 5:
                break
            is_diverse = True
            for ds in diverse_starts:
                if np.linalg.norm(cand - ds) < min_dist:
                    is_diverse = False
                    break
            if is_diverse:
                diverse_starts.append(cand)

        # If we don't have enough diverse starts, add more from the sorted list
        if len(diverse_starts) < 5:
            for cand, fval in candidates:
                if len(diverse_starts) >= 5:
                    break
                if cand not in diverse_starts:
                    diverse_starts.append(cand)

        # Ensure we have at least one start
        if not diverse_starts:
            diverse_starts = [best_x.copy()]

        # Phase 3: Local refinement with Nelder-Mead for each diverse start
        for start in diverse_starts:
            if not elapsed_ok() or calls_made >= budget - 5:
                break

            # Use scipy's Nelder-Mead with our budget-aware wrapper
            def local_func(x):
                if not elapsed_ok() or calls_made >= budget:
                    return 1e10  # large value to stop
                return eval_func(x)

            try:
                result = minimize(
                    local_func,
                    start,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'xatol': 1e-6,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
                if result.x is not None and result.fun < best_f:
                    try_improve(result.x, result.fun)
            except Exception:
                pass

            # Additional fine-tuning with small random steps
            if elapsed_ok() and calls_made < budget - 5:
                x_local = best_x.copy()
                f_local = best_f
                step_size = 0.1
                for _ in range(20):
                    if not elapsed_ok() or calls_made >= budget:
                        break
                    step = rng.normal(0, step_size, dim)
                    trial = np.clip(x_local + step, -5, 5)
                    ft = eval_func(trial)
                    if ft is None:
                        break
                    if ft < f_local:
                        x_local = trial
                        f_local = ft
                        step_size *= 0.7
                    else:
                        step_size *= 0.9

        # Phase 4: Final exploitation - many small steps around the best point
        if elapsed_ok() and calls_made < budget:
            x_local = best_x.copy()
            f_local = best_f
            step_size = 0.05
            n_final = min(100, budget - calls_made)
            for _ in range(n_final):
                if not elapsed_ok() or calls_made >= budget:
                    break
                step = rng.normal(0, step_size, dim)
                trial = np.clip(x_local + step, -5, 5)
                ft = eval_func(trial)
                if ft is None:
                    break
                if ft < f_local:
                    x_local = trial
                    f_local = ft
                    step_size *= 0.85
                else:
                    step_size *= 0.95

        return best_x, best_f
# EVOLVE-BLOCK-END
