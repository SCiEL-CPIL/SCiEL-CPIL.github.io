"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Derivative-free global search with Sobol coverage + adaptive local exploitation."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Use a Sobol-like quasi-random sequence for systematic coverage
        # We'll generate points using a stratified sampling approach
        rng = np.random.default_rng(42)

        # Phase 1: Systematic exploration with quasi-random points
        # Split budget: ~40% exploration, ~50% exploitation, ~10% final polish
        n_explore = int(budget * 0.4)
        n_exploit = int(budget * 0.5)
        n_polish = budget - n_explore - n_exploit

        # Generate quasi-random points using stratified sampling
        # Divide each dimension into bins and pick one point per stratum
        n_strata = 2
        points = []
        for i in range(n_explore):
            # Use a deterministic quasi-random sequence based on golden ratio
            t = i / max(1, n_explore - 1) if n_explore > 1 else 0.5
            # Halton-like sequence for each dimension
            x = np.zeros(dim)
            for d in range(dim):
                # Use different bases for different dimensions
                base = d + 2
                # Generate a point in [0,1) using a simple low-discrepancy method
                # For simplicity, use a combination of deterministic and random
                frac = (i * (base ** 0.5) + d * 7.3) % 1.0
                x[d] = -5.0 + 10.0 * frac
            # Add small random jitter to avoid exact duplicates
            x += rng.uniform(-0.1, 0.1, dim)
            x = np.clip(x, -5, 5)
            points.append(x)

        # Evaluate all exploration points
        best_x = None
        best_f = np.inf
        eval_count = 0

        for x in points:
            if time.time() > deadline:
                break
            fx = func(x)
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Adaptive local exploitation around the best point
        # Use a variant of (1+1)-ES with adaptive step size
        if best_x is not None and eval_count < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5  # Start with smaller step size for exploitation

            for _ in range(n_exploit):
                if time.time() > deadline:
                    break
                if eval_count >= budget:
                    break

                # Generate candidate
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                eval_count += 1

                if fy < fx:
                    x, fx = y, fy
                    best_x, best_f = x.copy(), fx
                    sigma *= 1.2
                else:
                    sigma *= 0.85

        # Phase 3: Final polish with very small steps
        if best_x is not None and eval_count < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.05

            for _ in range(n_polish):
                if time.time() > deadline:
                    break
                if eval_count >= budget:
                    break

                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                eval_count += 1

                if fy < fx:
                    x, fx = y, fy
                    best_x, best_f = x.copy(), fx
                    sigma *= 1.1
                else:
                    sigma *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
