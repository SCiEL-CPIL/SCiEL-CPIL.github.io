"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with adaptive step, success-ratio tuning, and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _run_local_search(self, func, x_start, sigma_start, rng, budget_remaining, deadline, best_state):
        n = self.dim
        x = x_start.copy()
        fx = func(x)
        calls = 1
        sigma = sigma_start
        best_x = x.copy()
        best_f = fx

        success_window = []
        window_size = 10
        max_no_improve = 40
        no_improve = 0

        for _ in range(budget_remaining - 1):
            if time.time() > deadline:
                break
            if calls >= budget_remaining:
                break

            y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
            fy = func(y)
            calls += 1

            success = fy < fx
            if success:
                x, fx = y, fy
                no_improve = 0
                if len(success_window) >= window_size:
                    success_window.pop(0)
                success_window.append(1)
                if len(success_window) == window_size:
                    rate = sum(success_window) / window_size
                    if rate > 0.55:
                        sigma *= 1.3
                    elif rate < 0.35:
                        sigma *= 0.75
            else:
                no_improve += 1
                if len(success_window) >= window_size:
                    success_window.pop(0)
                success_window.append(0)
                if len(success_window) == window_size:
                    rate = sum(success_window) / window_size
                    if rate > 0.55:
                        sigma *= 1.3
                    elif rate < 0.35:
                        sigma *= 0.75

            if fy < best_f:
                best_f = fy
                best_x = y.copy()

            if no_improve >= max_no_improve:
                break

        if best_f < best_state[0]:
            best_state[0] = best_f
            best_state[1] = best_x.copy()

        return calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285
        n = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        # Global best tracking
        best_f = np.inf
        best_x = np.zeros(n)

        # Phase 1: Initial exploration with several random starts
        # Use 80% of budget for multi-start, 20% for refinement
        phase1_budget = int(budget * 0.75)
        n_starts = 8
        calls_used = 0
        best_state = [best_f, best_x]

        # First, evaluate origin
        x0 = np.zeros(n)
        f0 = func(x0)
        calls_used += 1
        if f0 < best_f:
            best_f = f0
            best_x = x0.copy()

        # Multi-start local searches
        start_points = [x0.copy()]
        for _ in range(n_starts - 1):
            sp = rng.uniform(-4, 4, n)
            start_points.append(sp)

        # Shuffle and try them
        for sp in start_points:
            if calls_used >= budget:
                break
            if time.time() > deadline:
                break
            remaining = budget - calls_used
            sigma_start = 1.5
            local_calls = self._run_local_search(func, sp, sigma_start, rng, remaining, deadline, best_state)
            calls_used += local_calls

        # Phase 2: Refinement around best found point with smaller steps
        if calls_used < budget and time.time() < deadline:
            remaining = budget - calls_used
            # Try several perturbations around best
            n_refine_starts = 5
            for i in range(n_refine_starts):
                if calls_used >= budget:
                    break
                if time.time() > deadline:
                    break
                if i == 0:
                    rp = best_x.copy()
                else:
                    rp = np.clip(best_x + 0.3 * rng.standard_normal(n), -5, 5)
                remaining = budget - calls_used
                local_calls = self._run_local_search(func, rp, 0.5, rng, remaining, deadline, best_state)
                calls_used += local_calls

        # Phase 3: Final fine-tuning with very small steps if budget remains
        if calls_used < budget and time.time() < deadline:
            remaining = budget - calls_used
            local_calls = self._run_local_search(func, best_x, 0.1, rng, remaining, deadline, best_state)
            calls_used += local_calls

        return best_x, best_f
# EVOLVE-BLOCK-END
