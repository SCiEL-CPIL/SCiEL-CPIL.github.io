"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Two-phase optimizer: structured exploration then (1+1)-ES exploitation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)

        best_x = None
        best_f = np.inf
        calls_used = 0

        def evaluate(x):
            nonlocal best_x, best_f, calls_used
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_used += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Structured exploration using stratified sampling
        # Allocate ~20% of budget to exploration
        explore_budget = max(20, budget // 5)
        
        # Stratified sampling: divide each dimension into bins, sample one point per cell
        # For dim=5, use 2 bins per dimension = 32 cells, sample from each
        bins_per_dim = 2
        num_cells = bins_per_dim ** dim
        points_per_cell = max(1, explore_budget // num_cells)
        
        for cell_idx in range(num_cells):
            if calls_used >= budget or time.time() >= deadline:
                break
            
            # Determine which cell this index corresponds to
            # Convert cell_idx to multi-dimensional index
            cell_coords = []
            temp = cell_idx
            for d in range(dim):
                cell_coords.append(temp % bins_per_dim)
                temp //= bins_per_dim
            cell_coords = cell_coords[::-1]  # reverse to get proper order
            
            # Sample points within this cell
            for _ in range(points_per_cell):
                if calls_used >= budget or time.time() >= deadline:
                    break
                x = np.zeros(dim)
                for d in range(dim):
                    # Uniform within the cell
                    lo = -5.0 + cell_coords[d] * (10.0 / bins_per_dim)
                    hi = -5.0 + (cell_coords[d] + 1) * (10.0 / bins_per_dim)
                    x[d] = rng.uniform(lo, hi)
                evaluate(x)
        
        # If no points were evaluated (shouldn't happen), fallback
        if best_x is None:
            evaluate(np.zeros(dim))
        
        # Phase 2: (1+1)-ES exploitation from best point
        x = best_x.copy()
        f_x = best_f
        sigma = 1.0
        min_sigma = 1e-6
        max_sigma = 5.0
        
        success_count = 0
        success_window = 0
        window_size = 10
        no_improve_calls = 0
        max_no_improve = 30
        restarts = 0
        max_restarts = 10
        
        remaining = budget - calls_used
        
        while remaining > 0 and time.time() < deadline:
            # Generate trial point
            step = sigma * rng.standard_normal(dim)
            trial_x = x + step
            trial_x = np.clip(trial_x, -5, 5)
            
            trial_f = evaluate(trial_x)
            remaining = budget - calls_used
            
            if trial_f < f_x:
                # Accept
                x = trial_x
                f_x = trial_f
                success_count += 1
                success_window += 1
                no_improve_calls = 0
            else:
                # Reject
                success_window += 1
                no_improve_calls += 1
            
            # Adapt sigma based on success rate over window
            if success_window >= window_size:
                rate = success_count / success_window
                if rate > 0.5:
                    sigma *= 1.4
                elif rate < 0.2:
                    sigma *= 0.7
                else:
                    sigma *= 1.05
                sigma = max(sigma, min_sigma)
                sigma = min(sigma, max_sigma)
                success_count = 0
                success_window = 0
            
            # Restart if stuck
            if no_improve_calls >= max_no_improve:
                restarts += 1
                if restarts > max_restarts:
                    break
                # Reset sigma and try a new location
                sigma = 1.0
                no_improve_calls = 0
                success_count = 0
                success_window = 0
                
                # Try a few random points to find a better basin
                for _ in range(3):
                    if calls_used >= budget or time.time() >= deadline:
                        break
                    candidate_x = rng.uniform(-4, 4, size=dim)
                    candidate_f = evaluate(candidate_x)
                    if candidate_f < best_f:
                        x = candidate_x
                        f_x = candidate_f
                    remaining = budget - calls_used
                
                # If we found a better point, continue from there; otherwise perturb current best
                if f_x >= best_f - 1e-10:
                    # Perturb around global best
                    x = best_x + 0.5 * rng.standard_normal(dim)
                    x = np.clip(x, -5, 5)
                    f_x = evaluate(x)
                    remaining = budget - calls_used

        return best_x, best_f
# EVOLVE-BLOCK-END
