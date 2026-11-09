"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Global-scan then multi-restart (1+1)-ES with deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Phase 1: Global scan – sample diverse points to identify promising basins
        # Use a stratified sampling approach for better coverage
        n_scan = 150
        evaluated = []

        # Create a stratified grid with some jitter
        # Divide the space into a 5D grid
        # For 5D with 150 points, we can do roughly 3^5 = 243, so let's use fewer
        # Actually let's do a mix: some grid points and some random
        n_grid = 3  # 3^5 = 243, too many, let's do 2^5 = 32 grid points
        # Actually, let's use a different approach: space-filling curve or stratified random
        
        # Use stratified random sampling
        for i in range(n_scan):
            if time.time() > deadline - 5.0:
                break
            if len(evaluated) >= budget:
                break
            # Stratified: divide each dimension into n_scan^(1/dim) bins
            # For simplicity, use random but with stratification
            x = rng.uniform(-4.5, 4.5, dim)
            fx = func(x)
            evaluated.append((fx, x.copy()))

        # Sort by function value
        evaluated.sort(key=lambda t: t[0])
        best_x = evaluated[0][1].copy()
        best_fx = evaluated[0][0]

        # Phase 2: Focused refinement on top candidates
        # Take top K candidates and run (1+1)-ES on each
        n_refine = 4
        top_candidates = evaluated[:n_refine]
        remaining_budget = budget - len(evaluated)

        # Allocate refinement budget: give more to better candidates
        # Use proportional allocation based on rank
        weights = [1.0 / (k + 1) for k in range(n_refine)]
        total_w = sum(weights)
        refine_budgets = [max(10, int(remaining_budget * w / total_w)) for w in weights]

        for rank, (fx_init, x_init) in enumerate(top_candidates):
            if time.time() > deadline - 10.0:
                break
            if remaining_budget <= 0:
                break

            x = x_init.copy()
            fx = fx_init
            sigma = 1.0  # Start with moderate step size

            n_steps = min(refine_budgets[rank], remaining_budget)
            for step in range(n_steps):
                if time.time() > deadline - 2.0:
                    return best_x, best_fx
                if remaining_budget <= 0:
                    break

                # Adaptive sigma: increase on success, decrease on failure
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                remaining_budget -= 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                    if fx < best_fx:
                        best_x, best_fx = x.copy(), fx
                else:
                    sigma *= 0.8
                    # If sigma gets too small, try a larger jump
                    if sigma < 0.01:
                        sigma = 0.5

        return best_x, best_fx
# EVOLVE-BLOCK-END
