"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """LHS multi-start (1+1)-ES with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(42)

        # Phase 1: LHS (Latin Hypercube Sampling) for broad exploration
        # Use 100 LHS samples to cover the space well
        n_lhs = 100
        lhs_samples = np.zeros((n_lhs, dim))
        for d in range(dim):
            perm = rng.permutation(n_lhs)
            lhs_samples[:, d] = -5 + 10 * (perm + 0.5) / n_lhs

        best_x = None
        best_f = np.inf
        calls_used = 0

        for i in range(n_lhs):
            if time.time() > deadline - 30:  # keep 30s for refinement
                break
            x = lhs_samples[i]
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Local refinement with (1+1)-ES from best point
        # Also try a few nearby points to escape local minima
        x = best_x.copy()
        fx = best_f
        sigma = 0.5

        # Try a few initial sigma values and pick the best
        for sigma_init in [0.5, 1.0, 2.0]:
            if time.time() > deadline - 10:
                break
            x_try = x.copy()
            fx_try = fx
            sigma = sigma_init
            # Run (1+1)-ES for a limited number of steps
            max_steps = min(150, budget - calls_used - 50)
            for _ in range(max_steps):
                if time.time() > deadline - 10:
                    break
                if calls_used >= budget - 50:
                    break
                y = np.clip(x_try + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1
                if fy < fx_try:
                    x_try, fx_try = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.8
            if fx_try < best_f:
                best_f = fx_try
                best_x = x_try.copy()
                x = x_try.copy()
                fx = fx_try

        # Phase 3: Final polish with very small sigma
        if time.time() < deadline - 5 and calls_used < budget:
            sigma = 0.1
            x = best_x.copy()
            fx = best_f
            remaining = budget - calls_used
            for _ in range(remaining):
                if time.time() > deadline - 2:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                else:
                    sigma *= 0.7
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
