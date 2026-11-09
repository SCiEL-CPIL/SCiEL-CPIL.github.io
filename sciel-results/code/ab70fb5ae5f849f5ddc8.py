"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with LHS exploration and success-rate sigma adaptation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # 35s safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        call_count = 0

        def eval_point(x):
            nonlocal best_x, best_f, call_count
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: LHS exploration to find diverse good starting regions
        # Use ~15% of budget for exploration
        explore_budget = max(15, budget // 7)
        
        # Latin Hypercube sampling
        n_lhs = min(explore_budget, 25)
        for i in range(n_lhs):
            x = np.zeros(dim)
            for d in range(dim):
                lo = -5.0 + d * 10.0 / dim
                hi = lo + 10.0 / dim
                x[d] = lo + rng.random() * (hi - lo)
            eval_point(x)

        # Fill remaining explore budget with random points
        remaining_explore = explore_budget - n_lhs
        for _ in range(remaining_explore):
            x = rng.uniform(-5.0, 5.0, dim)
            eval_point(x)

        if best_x is None:
            x0 = np.zeros(dim)
            eval_point(x0)

        # Phase 2: Multi-restart (1+1)-ES from diverse starting points
        remaining_budget = budget - call_count
        
        # Number of restarts: balance between exploration and exploitation
        n_restarts = min(6, max(1, remaining_budget // 60))
        budget_per_restart = max(10, remaining_budget // n_restarts)

        for r in range(n_restarts):
            if call_count >= budget:
                break
            if time.time() > deadline:
                break

            # Starting point selection
            if r == 0:
                # Start from the best point found during exploration
                x = best_x.copy()
                sigma = 1.0
            else:
                # Perturb the best point for later restarts
                x = best_x + rng.normal(0, 2.0, dim)
                x = np.clip(x, -5.0, 5.0)
                sigma = 1.5

            # Evaluate starting point if it's new
            if r == 0:
                fx = best_f
            else:
                fx = eval_point(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

            # (1+1)-ES with success-rate adaptation
            successes = 0
            total_steps = 0
            window_size = 12

            for step in range(budget_per_restart):
                if call_count >= budget:
                    break
                if time.time() > deadline:
                    break

                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = eval_point(y)
                total_steps += 1

                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    sigma *= 1.25
                else:
                    sigma *= 0.85

                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

                # Adapt sigma based on success rate
                if total_steps % window_size == 0:
                    success_rate = successes / window_size
                    if success_rate > 0.5:
                        sigma *= 1.3
                    elif success_rate < 0.2:
                        sigma *= 0.7
                    successes = 0

                # Clamp sigma to reasonable range
                sigma = np.clip(sigma, 0.005, 5.0)

        # Phase 3: Fine-tuning around the best point with small steps
        if call_count < budget and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.05
            fine_tune_budget = min(budget - call_count, 80)
            
            for _ in range(fine_tune_budget):
                if call_count >= budget:
                    break
                if time.time() > deadline:
                    break

                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = eval_point(y)

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.15
                else:
                    sigma *= 0.9

                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
