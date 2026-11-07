"""Portfolio optimizer for BBOB f3 (Rastrigin) with multi-scale exploration and local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-strategy optimizer for Rastrigin-like functions:
    1. Coarse random sampling to identify promising basins.
    2. Adaptive (1+1)-ES with success rule for local refinement.
    3. Multi-restart strategy with varying step sizes to escape local minima.
    4. Time-budgeted to stay within wall-clock limits.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 40s safety margin before 330s kill

        # Global state
        best_x = None
        best_f = float('inf')
        calls_made = 0
        max_calls = self.budget

        def check_deadline():
            return (time.time() - start_time) > deadline or calls_made >= max_calls

        def update_best(x, f):
            nonlocal best_x, best_f, calls_made
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Coarse exploration
        # Sample diverse points across the domain to find promising basins
        rng = np.random.default_rng(12345)
        n_explore = min(30, max_calls // 15)
        
        # Include origin and some structured points
        candidates = [np.zeros(self.dim)]
        # Add random points
        for _ in range(n_explore - 1):
            candidates.append(rng.uniform(-4.5, 4.5, self.dim))

        for x in candidates:
            if check_deadline():
                break
            x_clipped = np.clip(x, -5, 5)
            f = func(x_clipped)
            update_best(x_clipped, f)

        if best_x is None:
            x = np.zeros(self.dim)
            f = func(x)
            update_best(x, f)
            return best_x, best_f

        # Phase 2: Local refinement with adaptive step size
        # We'll run multiple independent local searches from different starting points
        # Each search uses (1+1)-ES with success rule for sigma adaptation

        def local_search(x_start, f_start, budget_left, rng_local, sigma_init=0.5):
            """
            Runs a (1+1)-ES local search from x_start.
            Uses success rule: if 1/5 of last 15 steps successful, sigma unchanged.
            """
            nonlocal calls_made, best_x, best_f
            x = x_start.copy()
            f = f_start
            sigma = sigma_init
            success_history = []
            max_history = 15

            for i in range(budget_left):
                if check_deadline():
                    break

                # Generate candidate
                step = sigma * rng_local.standard_normal(self.dim)
                y = np.clip(x + step, -5, 5)
                fy = func(y)
                calls_made += 1

                # Update success history
                if fy < f:
                    success_history.append(1)
                    x = y
                    f = fy
                    # Update global best
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                else:
                    success_history.append(0)

                # Trim history
                if len(success_history) > max_history:
                    success_history.pop(0)

                # Adapt sigma using success rule
                if len(success_history) == max_history:
                    successes = sum(success_history)
                    if successes > max_history / 5:
                        sigma *= 1.1
                    elif successes < max_history / 5:
                        sigma *= 0.9

                # Keep sigma in reasonable bounds
                sigma = np.clip(sigma, 0.01, 5.0)

            return x, f

        # Outer loop: Multiple restarts with different strategies
        remaining = max_calls - calls_made
        if remaining <= 0:
            return best_x, best_f

        # Strategy 1: Polish the current best with small sigma
        part1 = remaining // 4
        if part1 > 5:
            local_search(best_x, best_f, part1, rng, sigma_init=0.2)
            remaining = max_calls - calls_made

        # Strategy 2: Escape from local minimum with medium sigma
        if remaining > 20:
            part2 = remaining // 3
            # Start from a perturbation of the best
            sigma_escape = 0.8
            x_start2 = np.clip(best_x + sigma_escape * rng.standard_normal(self.dim), -5, 5)
            f_start2 = func(x_start2)
            calls_made += 1
            if f_start2 < best_f:
                best_f = f_start2
                best_x = x_start2.copy()
            
            local_search(x_start2, f_start2, part2, rng, sigma_init=0.5)
            remaining = max_calls - calls_made

        # Strategy 3: Re-explore with larger sigma
        if remaining > 20:
            part3 = remaining // 2
            sigma_reexplore = 1.5
            x_start3 = np.clip(best_x + sigma_reexplore * rng.standard_normal(self.dim), -5, 5)
            f_start3 = func(x_start3)
            calls_made += 1
            if f_start3 < best_f:
                best_f = f_start3
                best_x = x_start3.copy()
            
            local_search(x_start3, f_start3, part3, rng, sigma_init=1.0)
            remaining = max_calls - calls_made

        # Final polish with whatever budget is left
        if remaining > 5 and not check_deadline():
            local_search(best_x, best_f, remaining, rng, sigma_init=0.1)

        return best_x, best_f

# EVOLVE-BLOCK-END
