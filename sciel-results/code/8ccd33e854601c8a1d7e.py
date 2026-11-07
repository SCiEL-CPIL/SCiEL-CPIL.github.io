import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.start_time = None

    def __call__(self, func):
        self.start_time = time.time()
        deadline = self.start_time + 280  # 300s limit, keep 20s margin
        
        # Wrap func to track calls and handle time/budget
        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')
        
        def wrapped_func(x):
            if self.calls >= self.budget:
                return self.best_f
            if time.time() > deadline:
                return self.best_f
            
            x = np.clip(x, -5, 5)
            self.calls += 1
            f_val = float(func(x))
            
            if f_val < self.best_f:
                self.best_f = f_val
                self.best_x = x.copy()
                
            return f_val

        rng = np.random.default_rng(42)
        
        # Phase 1: Diversification
        # Use a Latin Hypercube-like or just random points spread out
        # 20 initial points
        n_init = 20
        if self.budget < n_init:
             n_init = self.budget
             
        init_points = rng.uniform(-5, 5, (n_init, self.dim))
        
        # Evaluate initial points
        for x in init_points:
            if self.calls >= self.budget or time.time() > deadline:
                break
            wrapped_func(x)
        
        # Get fitnesses
        # Note: wrapped_func updates self.best_f, but we need individual fitnesses to sort
        # Let's re-evaluate or store them? 
        # Better: store the results from the initial phase.
        
        # Let's redo this slightly more carefully to get individual scores for sorting
        # We already called wrapped_func. We can just re-use the points and assume the best ones are among the initial set.
        # To sort, we need the f values. Let's just keep the points and re-evaluate the top candidates? 
        # No, that wastes budget.
        
        # Let's modify wrapped_func to store recent history? 
        # Simpler: Just run the initial evaluations and store (x, f) pairs.
        
        # Reset for proper tracking if needed, but let's just use the points we have.
        # We need to know which initial points were good.
        # Let's re-run the initial evaluation logic to capture values.
        
        # Actually, the simplest robust way:
        # 1. Generate points.
        # 2. Evaluate them one by one, storing (x, f).
        
        self.calls = 0
        self.best_f = float('inf')
        self.best_x = None
        
        # Re-implement wrapper to be safe or just use local vars
        evals = []
        
        for x in init_points:
             if self.calls >= self.budget or time.time() > deadline:
                 break
             x_clip = np.clip(x, -5, 5)
             self.calls += 1
             f_val = float(func(x_clip))
             evals.append((f_val, x_clip))
             if f_val < self.best_f:
                 self.best_f = f_val
                 self.best_x = x_clip.copy()
        
        # Sort by fitness
        evals.sort(key=lambda item: item[0])
        
        # Phase 2: Local Refinement on top candidates
        # Take top 5 candidates
        n_refine = min(5, len(evals))
        
        # Allocate remaining budget among them
        remaining_budget = self.budget - self.calls
        if remaining_budget <= 0:
            return self.best_x, self.best_f
            
        # We will run Nelder-Mead on the top n_refine points
        # Each gets a portion of the remaining budget
        # To ensure early progress, we run them sequentially. 
        # The first one (best initial) will likely improve quickly.
        
        # We need a wrapper that checks the global budget and time
        def local_wrapper(x):
            if self.calls >= self.budget or time.time() > deadline:
                return self.best_f
            x = np.clip(x, -5, 5)
            self.calls += 1
            f_val = float(func(x))
            if f_val < self.best_f:
                self.best_f = f_val
                self.best_x = x.copy()
            return f_val

        # Determine maxfev for each Nelder-Mead run
        # Leave some slack for overhead
        maxfev_per_run = max(10, (remaining_budget - n_refine) // n_refine)
        
        for i in range(n_refine):
            if self.calls >= self.budget or time.time() > deadline:
                break
                
            start_x = evals[i][1]
            
            # Nelder-Mead is derivative free
            # initial_simplex: default is based on start_x
            # We can try to make the simplex smaller if we are already good
            # Default step size is 0.05 * x_i + 0.00025 if x_i is 0. 
            # For [-5,5], this is reasonable.
            
            try:
                res = minimize(
                    local_wrapper, 
                    start_x, 
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'maxfev': maxfev_per_run, 'xatol': 1e-6, 'fatol': 1e-6}
                )
            except Exception:
                pass
                
            # If we found a better point, update evals list? 
            # Not strictly necessary if we just want the global best at the end,
            # but for subsequent runs, using the improved point as start could be better?
            # The loop structure runs them sequentially. 
            # If run 1 improves the best, run 2 starts from its *original* start_x.
            # This is fine for multi-start.
            
        return self.best_x, self.best_f
