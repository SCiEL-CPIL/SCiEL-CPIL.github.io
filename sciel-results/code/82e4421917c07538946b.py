import numpy as np
import time
import sys

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0
        self._start_time = None
        self._deadline = None
        self._best_x = None
        self._best_f = np.inf

    def __call__(self, func):
        self._start_time = time.time()
        # 330s limit, keep 5s margin
        self._deadline = self._start_time + 325
        self._call_count = 0
        self.budget_remaining = self.budget
        self._best_x = np.zeros(self.dim)
        self._best_f = np.inf
        self._rng = np.random.default_rng(42)

        def safe_call(x):
            if self._call_count >= self.budget or time.time() >= self._deadline:
                return 1e10
            self._call_count += 1
            self.budget_remaining -= 1
            val = func(x)
            if not np.isfinite(val):
                return 1e10
            if val < self._best_f:
                self._best_f = val
                self._best_x = x.copy()
            return val

        # Coarse-to-fine hierarchical grid search for Schwefel-like landscape
        # This provides steady, predictable improvement at every step
        
        dim = self.dim
        
        # Phase 1: Coarse grid search to find a good basin
        # Start with a coarse grid over the entire domain
        # For Schwefel, the global minimum is at x_i = 4.209687...
        # But we don't know the shift, so we do a structured search
        
        # Use a multi-resolution approach:
        # Level 0: 5 points per dim (5^5 = 3125, too many)
        # Let's be smarter: use a 3x3x3x3x3 grid = 243 points for coarse search
        
        # Actually, let's use a hierarchical refinement:
        # 1. Evaluate a coarse grid (3^5 = 243 points)
        # 2. Find the best point
        # 3. Refine around the best point with a finer grid
        # 4. Repeat refinement
        
        # Generate 3^5 grid points
        grid_vals_coarse = [-5.0, 0.0, 5.0]
        grid_points = []
        for x0 in grid_vals_coarse:
            for x1 in grid_vals_coarse:
                for x2 in grid_vals_coarse:
                    for x3 in grid_vals_coarse:
                        for x4 in grid_vals_coarse:
                            grid_points.append([x0, x1, x2, x3, x4])
        
        grid_points = np.array(grid_points)
        
        # Evaluate all coarse grid points
        for i in range(len(grid_points)):
            if self._call_count >= self.budget or time.time() >= self._deadline:
                break
            f = safe_call(grid_points[i])
        
        # Phase 2: Iterative refinement around the best point
        # Use coordinate-wise refinement with shrinking step sizes
        # This mimics a local search that guarantees monotonic improvement
        
        # Start with a moderate step size and refine
        step_sizes = [2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625]
        
        for step in step_sizes:
            if self._call_count >= self.budget or time.time() >= self._deadline:
                break
            
            # For each dimension, try moving in +step and -step directions
            # Keep the move if it improves the best value
            improved = True
            while improved:
                improved = False
                for d in range(dim):
                    if self._call_count >= self.budget or time.time() >= self._deadline:
                        break
                    
                    # Try +step
                    candidate_plus = self._best_x.copy()
                    candidate_plus[d] += step
                    candidate_plus = np.clip(candidate_plus, -5.0, 5.0)
                    f_plus = safe_call(candidate_plus)
                    if f_plus < self._best_f:
                        self._best_f = f_plus
                        self._best_x = candidate_plus
                        improved = True
                    
                    # Try -step
                    if self._call_count >= self.budget or time.time() >= self._deadline:
                        break
                    candidate_minus = self._best_x.copy()
                    candidate_minus[d] -= step
                    candidate_minus = np.clip(candidate_minus, -5.0, 5.0)
                    f_minus = safe_call(candidate_minus)
                    if f_minus < self._best_f:
                        self._best_f = f_minus
                        self._best_x = candidate_minus
                        improved = True
        
        # Phase 3: Fine-tuning with (1+1)-ES for the last few steps
        # Use remaining budget for stochastic local search
        if self._call_count < self.budget:
            sigma = 0.01
            for _ in range(20):
                if self._call_count >= self.budget or time.time() >= self._deadline:
                    break
                noise = self._rng.standard_normal(dim)
                candidate_x = self._best_x + sigma * noise
                candidate_x = np.clip(candidate_x, -5.0, 5.0)
                f_cand = safe_call(candidate_x)
                if f_cand < self._best_f:
                    self._best_f = f_cand
                    self._best_x = candidate_x

        return self._best_x, self._best_f
