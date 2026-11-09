"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Two-phase optimizer: coarse random/grid search followed by CMA-ES-like local search."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Coarse exploration - use ~30% of budget for diverse random search
        n_explore = max(30, budget // 3)
        
        best_x = None
        best_f = np.inf
        all_points = []
        all_fvals = []

        # First, try center and some structured points
        candidates = [np.zeros(dim)]
        # Add points at various radii from center
        for r in [0.5, 1.0, 2.0, 3.0, 4.0]:
            for _ in range(3):
                direction = rng.standard_normal(dim)
                direction /= np.linalg.norm(direction)
                candidates.append(np.clip(direction * r, -5, 5))
        # Add random points
        while len(candidates) < n_explore:
            candidates.append(rng.uniform(-5, 5, dim))

        for x in candidates:
            if time.time() > deadline:
                break
            x = np.clip(x, -5, 5)
            fx = func(x)
            all_points.append(x)
            all_fvals.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Sort by function value
        idx = np.argsort(all_fvals)
        all_points = [all_points[i] for i in idx]
        all_fvals = [all_fvals[i] for i in idx]

        # Phase 2: Local refinement using a population-based local search
        # Use the best few points to initialize a local search
        remaining_budget = budget - len(all_points)
        if remaining_budget <= 0 or time.time() > deadline:
            return best_x, best_f

        # Take top k points as seeds for local search
        k = min(5, len(all_points))
        seeds = all_points[:k]
        seed_fvals = all_fvals[:k]

        # Local search: adaptive step size with multiple trajectories
        sigma = 0.5
        n_local = remaining_budget

        for i in range(n_local):
            if time.time() > deadline:
                break
            
            # Choose which seed to expand (roulette wheel based on fitness)
            # Simpler: cycle through seeds, or use the best one
            # Use the best seed most of the time
            if i % 3 == 0 or i == 0:
                base_idx = 0  # best
            else:
                base_idx = i % min(k, 3)

            base_x = seeds[base_idx]
            base_f = seed_fvals[base_idx]

            # Generate perturbation
            step = sigma * rng.standard_normal(dim)
            # Occasionally use a larger step for exploration
            if rng.random() < 0.1:
                step *= 3.0

            y = np.clip(base_x + step, -5, 5)
            fy = func(y)

            if fy < seed_fvals[base_idx]:
                seeds[base_idx] = y
                seed_fvals[base_idx] = fy
                # Increase step size on success
                sigma *= 1.3
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
            else:
                # Decrease step size on failure
                sigma *= 0.85

            # Keep sigma in reasonable bounds
            sigma = max(1e-6, min(sigma, 5.0))

            # If we've converged very well, reduce exploration
            if sigma < 1e-4:
                sigma = 1e-4

        return best_x, best_f
# EVOLVE-BLOCK-END
