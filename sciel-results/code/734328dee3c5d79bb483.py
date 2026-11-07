"""Optimizer for BBOB-style f11 (Discus) using multi-start L-BFGS-B."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-start L-BFGS-B optimizer with Sobol sampling for global coverage."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _sobol_sample(self, n, dim, seed=0):
        """Generate Sobol-like quasi-random points in [-5, 5]^dim."""
        rng = np.random.default_rng(seed)
        # Use a simple quasi-random sequence via Halton-like approach
        # For simplicity, use stratified random sampling
        points = np.zeros((n, dim))
        for d in range(dim):
            for i in range(n):
                # Stratified sampling: divide [0,1] into n equal intervals
                # and pick a random point within each
                interval = i / n
                points[i, d] = -5 + 10 * (interval + rng.random() / n)
        return points

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # Phase 1: Global exploration with multiple starts
        # Use Sobol-like sampling for diverse starting points
        n_starts = min(20, max(5, budget // 50))
        
        # Generate diverse starting points
        start_points = self._sobol_sample(n_starts, dim, seed=42)
        # Add center and some random points for extra diversity
        center = np.zeros(dim)
        rng = np.random.default_rng(123)
        extra_points = rng.uniform(-4, 4, size=(5, dim))
        
        all_starts = np.vstack([start_points, [center], extra_points])
        all_starts = np.clip(all_starts, -5, 5)
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        max_calls_per_local = max(30, budget // len(all_starts))
        
        # Track call count
        call_count = [0]
        
        def wrapped_func(x):
            call_count[0] += 1
            return func(x)
        
        # Evaluate all starting points
        for i, x0 in enumerate(all_starts):
            if call_count[0] >= budget:
                break
            if time.time() > deadline:
                break
            
            fx0 = wrapped_func(x0)
            if fx0 < best_f:
                best_f = fx0
                best_x = x0.copy()
        
        # Phase 2: Local refinement with L-BFGS-B from best starts
        # Sort starts by initial function value and refine top ones
        start_values = []
        for i, x0 in enumerate(all_starts):
            if call_count[0] >= budget:
                break
            if time.time() > deadline:
                break
            
            # Re-evaluate if not already done (simpler: just track)
            # Actually, let's just run L-BFGS-B from each start
            if call_count[0] >= budget - 10:
                break
            
            try:
                result = minimize(
                    wrapped_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 100, 'maxls': 20}
                )
                
                if result.success and result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
                    
            except Exception:
                continue
            
            if time.time() > deadline:
                break
        
        # Phase 3: If we have budget left, do a final polish from the best point
        if best_x is not None and call_count[0] < budget - 5:
            if time.time() < deadline:
                try:
                    result = minimize(
                        wrapped_func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 50, 'maxls': 10}
                    )
                    if result.success and result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                except Exception:
                    pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
