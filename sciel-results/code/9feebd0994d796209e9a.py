import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with Hooke-Jeeves pattern search refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin before 330s kill

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

        # Phase 1: Diverse exploration with Latin Hypercube + random sampling
        explore_budget = max(15, budget // 10)  # ~50 calls

        # Latin Hypercube sampling for diversity
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

        # Phase 2: Multi-restart (1+1)-ES from best point and diverse neighbors
        remaining_budget = budget - call_count

        # Determine number of restarts based on remaining budget
        n_restarts = min(6, max(2, remaining_budget // 60))
        budget_per_restart = max(10, remaining_budget // n_restarts)

        for r in range(n_restarts):
            if call_count >= budget:
                break
            if time.time() > deadline:
                break

            # Starting point: best found so far, perturbed for later restarts
            if r == 0:
                x = best_x.copy()
                sigma = 1.0
            else:
                # Perturb the best point with increasing spread
                spread = 1.5 + 0.5 * r
                x = best_x + rng.normal(0, spread, dim)
                x = np.clip(x, -5.0, 5.0)
                sigma = 1.5

            fx = best_f if r == 0 else eval_point(x)

            # (1+1)-ES with adaptive sigma and success-rate adaptation
            successes = 0
            total_steps = 0
            window_size = 8

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
                    sigma *= 0.82

                # Adapt sigma based on success rate
                if total_steps % window_size == 0:
                    success_rate = successes / window_size
                    if success_rate > 0.6:
                        sigma *= 1.15
                    elif success_rate < 0.2:
                        sigma *= 0.7
                    successes = 0

                # Prevent sigma from becoming too small or too large
                sigma = np.clip(sigma, 0.02, 5.0)

            # Phase 3: Hooke-Jeeves pattern search refinement on best point
            if call_count < budget and best_x is not None:
                x = best_x.copy()
                fx = best_f
                step_size = 0.3

                hj_budget = min(budget - call_count, 40)

                for hj_step in range(hj_budget):
                    if call_count >= budget:
                        break
                    if time.time() > deadline:
                        break

                    improved = False
                    for d in range(dim):
                        if call_count >= budget:
                            break
                        if time.time() > deadline:
                            break

                        # Try positive step
                        y_pos = x.copy()
                        y_pos[d] += step_size
                        y_pos = np.clip(y_pos, -5.0, 5.0)
                        fy_pos = eval_point(y_pos)

                        if fy_pos < fx:
                            x = y_pos
                            fx = fy_pos
                            improved = True
                            continue

                        # Try negative step
                        y_neg = x.copy()
                        y_neg[d] -= step_size
                        y_neg = np.clip(y_neg, -5.0, 5.0)
                        fy_neg = eval_point(y_neg)

                        if fy_neg < fx:
                            x = y_neg
                            fx = fy_neg
                            improved = True

                    if improved:
                        step_size *= 1.3  # Expand step on success
                    else:
                        step_size *= 0.6  # Contract step on failure

                    step_size = np.clip(step_size, 0.005, 2.0)

                    if step_size < 0.01:
                        break

        return best_x, best_f
