"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid multi-start scout + Powell/Nelder-Mead polish for Rosenbrock-style f8."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit

        call_count = 0
        best_x = None
        best_f = np.inf

        def wrapped_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= self.budget:
                return best_f
            x = np.asarray(x, dtype=float)
            if x.shape != (self.dim,):
                x = x.reshape(self.dim)
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def time_check():
            return time.time() < deadline

        rng = np.random.default_rng(42)

        # Phase 1: Multi-start scout to find promising basins
        # Use a mix of strategies to explore the landscape
        n_scouts = 20
        scout_budget_per = max(1, (self.budget - 100) // n_scouts)

        # Generate diverse starting points
        starts = []
        # Center
        starts.append(np.zeros(self.dim))
        # Corners of the box
        for i in range(min(8, 2 ** self.dim)):
            corner = np.array([(i >> j) & 1 for j in range(self.dim)], dtype=float)
            corner = corner * 10.0 - 5.0
            starts.append(corner)
        # Random points
        while len(starts) < n_scouts:
            starts.append(rng.uniform(-5, 5, self.dim))

        best_scout_x = None
        best_scout_f = np.inf

        for sx in starts[:n_scouts]:
            if not time_check() or call_count >= self.budget - 50:
                break

            # Quick local search from this start using a few Powell iterations
            try:
                res = minimize(
                    wrapped_func,
                    sx,
                    method='Powell',
                    options={'maxiter': 5, 'maxfev': 10}
                )
                if res.fun < best_scout_f:
                    best_scout_f = res.fun
                    best_scout_x = res.x.copy()
            except Exception:
                pass

        if best_scout_x is None:
            best_scout_x = np.zeros(self.dim)
            best_scout_f = wrapped_func(best_scout_x)

        # Phase 2: Intensive polish from the best scout result
        # Use Powell's method for the curved valley structure
        remaining_budget = self.budget - call_count - 5

        if remaining_budget > 10 and time_check():
            try:
                res = minimize(
                    wrapped_func,
                    best_scout_x,
                    method='Powell',
                    options={'maxiter': 100, 'maxfev': max(10, remaining_budget)}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 3: Nelder-Mead polish for fine-tuning if budget remains
        remaining_budget = self.budget - call_count - 5

        if remaining_budget > 10 and time_check() and best_x is not None:
            try:
                res = minimize(
                    wrapped_func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'maxfev': max(10, remaining_budget),
                             'xatol': 1e-8, 'fatol': 1e-10}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 4: If still have budget, do a few more Powell restarts from top scouts
        if call_count < self.budget - 10 and time_check():
            # Try Powell from a few other good points
            for sx in starts[:5]:
                if call_count >= self.budget - 10 or not time_check():
                    break
                if np.linalg.norm(sx - best_x) < 1e-6:
                    continue
                try:
                    res = minimize(
                        wrapped_func,
                        sx,
                        method='Powell',
                        options={'maxiter': 20, 'maxfev': max(5, self.budget - call_count - 10)}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass

        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = wrapped_func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
