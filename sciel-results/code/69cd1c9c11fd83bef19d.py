"""Optimizer for BBOB-style f15 (Rastrigin rotated) using a structured grid + local search hybrid."""
import numpy as np
import time
import math


class Optimizer:
    """Structured grid search with local refinement for Rastrigin f15."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285
        rng = np.random.default_rng(7)
        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf
        calls_used = 0

        def evaluate(x):
            nonlocal calls_used, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            fx = float(func(x))
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def time_ok():
            return time.time() < deadline and calls_used < budget

        # Phase 1: Coarse structured grid search
        # Rastrigin has period 2*pi ≈ 6.28 in each dimension.
        # Box is [-5, 5], so we have about 10/6.28 ≈ 1.6 periods per dimension.
        # A grid with spacing ~pi/2 ≈ 1.57 would give ~6-7 points per dimension.
        # Total: 6^5 = 7776 which is too many. Let's use a coarser grid first.
        
        # Start with a moderate grid: 4 points per dimension = 4^5 = 1024 points
        # That's too many for 500 budget. Let's use 3^5 = 243 points.
        
        # Actually, let's think differently. We have 500 evaluations.
        # Let's use a two-phase approach:
        # Phase 1: Coarse grid (e.g., 3 points per dim = 243 evals)
        # Phase 2: Local search around the best few points
        
        n_grid = 3
        grid_points = np.array(np.meshgrid(*[np.linspace(-5, 5, n_grid) for _ in range(dim)]))
        grid_points = grid_points.reshape(dim, -1).T
        
        # Shuffle to avoid systematic bias
        indices = rng.permutation(len(grid_points))
        
        for idx in indices:
            if not time_ok():
                break
            x = grid_points[idx]
            evaluate(x)
        
        if not time_ok():
            return (best_x, best_f)
        
        # Phase 2: Local search around the best points
        # Use a simple coordinate descent / Nelder-Mead style local search
        
        # Collect top candidates from grid
        # We don't have all values stored, so let's just use best_x and try perturbations
        
        # Actually, let's do a more systematic local search.
        # Try small perturbations around best_x in each direction
        
        local_budget = budget - calls_used
        if local_budget <= 0:
            return (best_x, best_f)
        
        # Coordinate descent with adaptive step
        step_sizes = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005]
        
        x_current = best_x.copy()
        f_current = best_f
        
        for step in step_sizes:
            if not time_ok():
                break
            improved = True
            while improved and time_ok():
                improved = False
                for d in range(dim):
                    # Try +step
                    x_try = x_current.copy()
                    x_try[d] += step
                    x_try = np.clip(x_try, -5, 5)
                    f_try = evaluate(x_try)
                    if f_try < f_current - 1e-12:
                        x_current = x_try
                        f_current = f_try
                        best_x = x_try.copy()
                        best_f = f_try
                        improved = True
                    
                    # Try -step
                    if time_ok():
                        x_try = x_current.copy()
                        x_try[d] -= step
                        x_try = np.clip(x_try, -5, 5)
                        f_try = evaluate(x_try)
                        if f_try < f_current - 1e-12:
                            x_current = x_try
                            f_current = f_try
                            best_x = x_try.copy()
                            best_f = f_try
                            improved = True
        
        # Phase 3: Fine-tuning with very small steps
        if time_ok():
            fine_steps = [0.002, 0.001, 0.0005]
            for step in fine_steps:
                if not time_ok():
                    break
                for _ in range(3):
                    if not time_ok():
                        break
                    for d in range(dim):
                        for sign in [1, -1]:
                            if not time_ok():
                                break
                            x_try = best_x.copy()
                            x_try[d] += sign * step
                            x_try = np.clip(x_try, -5, 5)
                            f_try = evaluate(x_try)
                            if f_try < best_f - 1e-12:
                                best_x = x_try.copy()
                                best_f = f_try
        
        return (best_x, best_f)
