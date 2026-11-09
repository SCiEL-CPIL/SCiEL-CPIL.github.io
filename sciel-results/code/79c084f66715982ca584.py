import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc

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
        # Set deadline to 325s to be safe (limit is 330s)
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
            try:
                val = float(func(x))
            except Exception:
                return 1e10
            if not np.isfinite(val):
                return 1e10
            if val < self._best_f:
                self._best_f = val
                self._best_x = x.copy()
            return val

        # 1. Global Exploration: Sobol Sequence
        n_sobol = 30
        try:
            sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42)
            sobol_points = sampler.random(n_sobol)
            sobol_points = sobol_points * 10.0 - 5.0
        except Exception:
            sobol_points = self._rng.uniform(-5, 5, (n_sobol, self.dim))

        evaluated_points = []
        
        # Evaluate initial Sobol points
        for i in range(n_sobol):
            if self._call_count >= self.budget or time.time() >= self._deadline:
                break
            x = sobol_points[i]
            f = safe_call(x)
            evaluated_points.append((f, x))
            
        if not evaluated_points:
            return self._best_x, self._best_f

        # Sort by fitness
        evaluated_points.sort(key=lambda x: x[0])
        
        # 2. Coarse-to-Fine Grid Search around best points
        # We will refine the top few candidates using a hierarchical grid approach.
        # This ensures steady improvement and guarantees convergence to the local minimum.
        
        top_candidates = [p[1] for p in evaluated_points[:5]]
        
        # For each candidate, perform a grid refinement
        for start_x in top_candidates:
            if self._call_count >= self.budget or time.time() >= self._deadline:
                break
            
            current_best_x = start_x.copy()
            current_best_f = safe_call(start_x)
            
            # Start with a relatively large step size, then refine
            # Step sizes: 1.0, 0.5, 0.25, 0.125, 0.0625, ...
            # We'll do a few levels of refinement
            step_sizes = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]
            
            for step in step_sizes:
                if self._call_count >= self.budget or time.time() >= self._deadline:
                    break
                
                # Generate grid points around current best
                # For dim=5, we can't do a full grid (2^5 * 5^5 is too big)
                # Instead, we'll do a coordinate descent style grid or a sparse grid
                
                # Let's try a simple approach: for each dimension, try +/- step
                # This is 2*dim = 10 points per step level
                # Total points per candidate: 10 * 6 = 60
                # For 5 candidates: 300 points
                # We have 500 budget, 30 used for Sobol, so 470 left. This fits.
                
                for d in range(self.dim):
                    if self._call_count >= self.budget or time.time() >= self._deadline:
                        break
                    
                    for direction in [-1, 1]:
                        if self._call_count >= self.budget or time.time() >= self._deadline:
                            break
                        
                        x_new = current_best_x.copy()
                        x_new[d] += direction * step
                        # Clip to bounds
                        x_new = np.clip(x_new, -5, 5)
                        
                        f_new = safe_call(x_new)
                        
                        if f_new < current_best_f:
                            current_best_f = f_new
                            current_best_x = x_new.copy()
                            # Update global best if needed (safe_call already did this)
            
            # After grid refinement, we might have a better point.
            # Add it to our evaluated points if it's new? 
            # Actually, safe_call updates self._best_x/f automatically.
            # We don't need to manage a separate list for the grid phase.

        # 3. Final Polish: Nelder-Mead on the very best point found
        if self._call_count < self.budget and time.time() < self._deadline - 5:
            def obj_func(x_local):
                return safe_call(x_local)
            
            local_budget = min(50, self.budget_remaining)
            try:
                res = minimize(obj_func, self._best_x, method='Nelder-Mead', 
                               options={'maxiter': 200, 'maxfev': local_budget, 'adaptive': True})
            except Exception:
                pass

        return self._best_x, self._best_f
