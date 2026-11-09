"""Portfolio optimizer for BBOB f15 (Rastrigin rotated)."""
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
        deadline = start_time + 320.0  # Safety margin before 330s kill
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

        # ---- Phase 1: LHS + random diverse exploration to locate a good basin ----
        # Use a small portion of the budget for global exploration
        explore_budget = max(10, min(50, budget // 10))
        
        # Latin Hypercube Sampling for diversity
        n_lhs = min(explore_budget, 20)
        for i in range(n_lhs):
            x = np.zeros(dim)
            for d in range(dim):
                lo = -5.0 + d * 10.0 / dim
                hi = lo + 10.0 / dim
                x[d] = lo + rng.random() * (hi - lo)
            eval_point(x)
        
        # Fill remaining explore budget with random points
        for _ in range(explore_budget - n_lhs):
            x = rng.uniform(-5.0, 5.0, dim)
            eval_point(x)

        if best_x is None:
            eval_point(np.zeros(dim))

        # ---- Phase 2: Multi-restart (1+1)-ES with success-rate sigma adaptation ----
        # Define restart points:
        # 1. Best found so far
        # 2-5. Random points in [-4, 4]
        # 6-7. Perturbed best points
        restart_points = [best_x.copy()]
        for _ in range(4):
            restart_points.append(rng.uniform(-4.0, 4.0, dim))
        for _ in range(2):
            restart_points.append(np.clip(best_x + rng.normal(0, 1.5, dim), -5.0, 5.0))
        
        n_restarts = len(restart_points)
        
        # Allocate budget: 
        # First restart gets the majority (to converge fast on the best lead)
        # Others get smaller shares to explore other basins
        remaining = budget - call_count
        alloc = []
        for i in range(n_restarts):
            if i == 0:
                # Give 50% of remaining to the first restart
                alloc.append(int(remaining * 0.5))
            else:
                # Distribute the rest among other restarts
                alloc.append(int(remaining * 0.5 / (n_restarts - 1)))
        
        # Ensure we don't exceed budget
        total_alloc = sum(alloc)
        if total_alloc > remaining:
            scale = remaining / total_alloc
            alloc = [int(a * scale) for a in alloc]
            alloc[-1] = remaining - sum(alloc[:-1])

        for r, start_pt in enumerate(restart_points):
            if call_count >= budget:
                break
            if time.time() > deadline:
                break

            x = start_pt.copy()
            if r == 0:
                fx = best_f
            else:
                fx = eval_point(x)

            sigma = 1.5
            successes = 0
            total_steps = 0
            window_size = 10

            for _ in range(alloc[r]):
                if call_count >= budget:
                    break
                if time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = eval_point(y)
                total_steps += 1

                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                else:
                    pass # (1+1) ES keeps x if y is worse

                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

                # Success-rate based sigma adaptation
                if total_steps % window_size == 0:
                    success_rate = successes / window_size
                    if success_rate > 0.5:
                        sigma *= 1.2
                    elif success_rate < 0.2:
                        sigma *= 0.7
                    successes = 0
                sigma = np.clip(sigma, 0.01, 5.0)

        # ---- Phase 3: Fine polish around the best point ----
        # If we have budget left, do a final tight search around the best point
        if call_count < budget and best_x is not None and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            successes = 0
            total_steps = 0
            window_size = 8
            for _ in range(budget - call_count):
                if call_count >= budget:
                    break
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = eval_point(y)
                total_steps += 1
                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                
                if total_steps % window_size == 0:
                    success_rate = successes / window_size
                    if success_rate > 0.5:
                        sigma *= 1.1
                    elif success_rate < 0.2:
                        sigma *= 0.8
                    successes = 0
                sigma = np.clip(sigma, 0.001, 1.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
