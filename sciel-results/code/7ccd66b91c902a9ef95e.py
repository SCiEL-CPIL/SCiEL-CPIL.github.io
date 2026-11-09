"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with adaptive step size and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        # Fixed seed for reproducibility
        rng = np.random.default_rng(42)

        # Weierstrass f16 on [-5,5]^5: multi-scale oscillatory function.
        # Strategy: 
        #   1. Quick random sampling to find a good starting basin (small fraction of budget).
        #   2. (1+1)-ES with adaptive sigma (1/5 rule) for exploitation.
        #   3. If time permits, a second restart from a different region if stuck.

        # Phase 1: Random sampling to find a promising region
        # Use ~10% of budget for exploration
        explore_budget = max(10, budget // 10)
        best_x = None
        best_f = np.inf

        # Evaluate a few well-spread points plus random ones
        # Start with center and a few structured points
        candidates = [
            np.zeros(n),
            np.full(n, 0.5),
            np.full(n, -0.5),
        ]
        for _ in range(explore_budget - len(candidates)):
            candidates.append(rng.uniform(-4, 4, n))

        for c in candidates:
            if time.time() > deadline:
                break
            c = np.clip(c, -5, 5)
            fc = func(c)
            if fc < best_f:
                best_f = fc
                best_x = c.copy()

        if best_x is None:
            # Fallback: just do a single random eval
            best_x = rng.uniform(-5, 5, n)
            best_f = func(best_x)

        # Phase 2: (1+1)-ES with adaptive step size
        x = best_x.copy()
        fx = best_f
        sigma = 1.0  # start with moderate step
        success = 0
        total = 0
        success_rate = 0.2  # 1/5 rule threshold

        remaining = budget - explore_budget
        for i in range(remaining):
            if time.time() > deadline:
                break

            y = x + sigma * rng.standard_normal(n)
            y = np.clip(y, -5, 5)
            fy = func(y)

            total += 1
            if fy < fx:
                x = y
                fx = fy
                best_f = fx
                best_x = x.copy()
                success += 1
            else:
                # No improvement
                pass

            # Adapt sigma every 10 steps
            if total % 10 == 0:
                rate = success / total
                if rate > success_rate:
                    sigma *= 1.1
                elif rate < success_rate:
                    sigma *= 0.9
                # Clamp sigma
                sigma = np.clip(sigma, 1e-4, 10.0)
                success = 0
                total = 0

        # Phase 3: If we still have budget and time, try a few more restarts
        # from slightly perturbed versions of the best point
        if time.time() < deadline - 10:
            for _ in range(min(5, budget - (explore_budget + remaining))):
                if time.time() > deadline:
                    break
                # Perturb best solution
                x2 = best_x + 0.5 * rng.standard_normal(n)
                x2 = np.clip(x2, -5, 5)
                fx2 = func(x2)
                if fx2 < best_f:
                    best_x = x2.copy()
                    best_f = fx2

        return best_x, best_f
# EVOLVE-BLOCK-END
