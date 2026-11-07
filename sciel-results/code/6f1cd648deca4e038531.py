import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        start_time = time.time()
        max_time = 320.0  # Safety margin before 330s kill
        
        def time_ok():
            return time.time() - start_time < max_time

        calls = 0
        best_x = None
        best_f = np.inf

        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            if not time_ok():
                return best_f
            x = np.clip(x, -5, 5)
            f_val = func(x)
            calls += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Strategy: Multi-start CMA-ES style or simple evolutionary strategy
        # Lunacek bi-Rastrigin is multimodal with a global and local minimum.
        # The global minimum is typically around 0 or shifted, local is at -1.
        # DE was good, but let's try a more robust approach:
        # 1. Coarse grid / random sampling to find promising basins.
        # 2. Adaptive Step Size (1+1)-ES or CMA-ES for refinement.
        
        # Given the hidden rotation and shift, we need robust global search first.
        # Let's use a population-based method with restarts if stagnation occurs.
        
        # Phase 1: Global Search using a simple Evolutionary Strategy (1+lambda)-ES
        # or Differential Evolution with more iterations but smaller population?
        # Actually, the previous DE approach scored 1.327.
        # Let's try a different structural change: 
        # Use a grid of initial points and refine each with a local optimizer,
        # but do it in a way that balances exploration and exploitation.
        
        # Alternative: Use scipy.optimize.differential_evolution but with 
        # more aggressive settings and then polish with Nelder-Mead.
        # The previous code did this. Let's tweak the budget split and parameters.
        
        # Maybe the issue is that DE gets stuck or doesn't converge fast enough.
        # Let's try a two-stage approach:
        # 1. Random sampling + best point selection (cheap global)
        # 2. CMA-ES or Nelder-Mead from multiple best points.
        
        # Let's implement a simple multi-start Nelder-Mead with random initialization.
        
        n_starts = 10
        evals_per_start = max(5, budget // n_starts)
        
        rng = np.random.default_rng(42)
        
        # Generate initial points
        # Mix of random and some structured points
        points = []
        # Center
        points.append(np.zeros(dim))
        # Random points
        for _ in range(n_starts - 1):
            points.append(rng.uniform(-5, 5, dim))
        
        # Evaluate all initial points
        local_best_x = None
        local_best_f = np.inf
        
        for p in points:
            if not time_ok() or calls >= budget:
                break
            f_val = wrapped_func(p)
            if f_val < local_best_f:
                local_best_f = f_val
                local_best_x = p.copy()
        
        # Now refine the best points found
        # Sort points by their function value (we need to store them)
        # Let's re-evaluate or store. Since we didn't store all, let's just use the best one found so far?
        # Better: Store all evaluated points and their values.
        
        # Let's restart the logic slightly to track all points.
        
        # Reset for a cleaner multi-start
        calls = 0
        best_x = None
        best_f = np.inf
        
        # Phase 1: Sample 20 points
        n_samples = 20
        samples = rng.uniform(-5, 5, (n_samples, dim))
        samples[0] = 0.0  # Ensure origin is sampled
        
        sample_vals = []
        for i in range(n_samples):
            if not time_ok() or calls >= budget:
                break
            f_val = wrapped_func(samples[i])
            sample_vals.append(f_val)
        
        # Sort samples by value
        sorted_indices = np.argsort(sample_vals)
        
        # Phase 2: Refine the top 5 points using Nelder-Mead
        n_refine = min(5, len(sample_vals))
        
        for idx in sorted_indices[:n_refine]:
            if not time_ok() or calls >= budget:
                break
            
            x0 = samples[idx]
            
            # Use Nelder-Mead with a limited budget
            remaining = budget - calls
            if remaining < 10:
                break
                
            # Simple manual Nelder-Mead or use scipy?
            # Using scipy.optimize.minimize is convenient.
            try:
                from scipy.optimize import minimize
                res = minimize(
                    wrapped_func,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': min(remaining, 100), 'xatol': 1e-6, 'fatol': 1e-8}
                )
            except Exception:
                pass
            
            # Check if we improved global best
            # Note: wrapped_func updates best_x and best_f internally
            # But we need to ensure best_x is valid
            
        # Phase 3: If we have budget left, do a global DE polish?
        # Or just return the best found.
        
        return best_x, best_f
