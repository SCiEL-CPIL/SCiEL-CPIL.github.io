import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        self.call_count = 0
        self.start_time = time.time()
        self.func = func
        self.deadline = self.start_time + 290.0
        self.best_x = None
        self.best_f = float('inf')
        self.dim = self.dim
        self.budget = self.budget

        def wrapped_func(x):
            if self.call_count >= self.budget:
                return float('inf')
            if time.time() > self.deadline:
                return float('inf')
            
            x = np.asarray(x)
            if x.shape != (self.dim,):
                x = np.reshape(x, (self.dim,))
            
            self.call_count += 1
            f_val = float(func(x))
            
            if np.isfinite(f_val) and f_val < self.best_f:
                self.best_f = f_val
                self.best_x = x.copy()
            
            return f_val

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Strategy: Multi-restart with a (1+1)-ES core.
        # Gallagher 101 peaks is highly multimodal. A single global search followed by a single local search
        # often gets trapped in a suboptimal basin. By distributing the budget across multiple restarts,
        # we increase the probability of finding the global minimum.
        # We use a simple (1+1)-ES for local refinement as it is robust and requires no gradient.
        
        # Allocate budget to restarts.
        # 5 restarts seems like a good balance between exploration and exploitation.
        n_restarts = 5
        budget_per_restart = budget // n_restarts
        
        # First restart gets a slightly larger budget for global scan
        # Actually, let's just use uniform budget for simplicity and robustness.
        
        for r in range(n_restarts):
            if self.call_count >= budget or time.time() > self.deadline:
                break
            
            # Phase 1 of each restart: Small global scan to find a starting point
            # Use LHS for better coverage
            n_scan = min(20, budget - self.call_count)
            if n_scan <= 0:
                break
            
            points = np.zeros((n_scan, dim))
            for d in range(dim):
                perm = rng.permutation(n_scan)
                points[:, d] = (perm + rng.random(n_scan)) / n_scan
                points[:, d] = points[:, d] * 10.0 - 5.0 # Scale to [-5, 5]
            
            # Evaluate scan points
            for i in range(n_scan):
                if self.call_count >= budget or time.time() > self.deadline:
                    break
                wrapped_func(points[i])
            
            # Phase 2 of each restart: (1+1)-ES local refinement
            if self.call_count >= budget or time.time() > self.deadline:
                break
            
            if self.best_x is None:
                continue
            
            x = self.best_x.copy()
            f = self.best_f
            sigma = 1.0 # Initial step size
            remaining = budget - self.call_count
            if remaining <= 0:
                break
            
            # (1+1)-ES loop
            while self.call_count < budget and time.time() < self.deadline:
                # Generate mutation
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0) # Clip to box
                
                f_y = wrapped_func(y)
                
                if f_y < f:
                    x = y
                    f = f_y
                    sigma *= 1.2 # Increase step size on success
                else:
                    sigma *= 0.8 # Decrease step size on failure
                
                # Stop if sigma is too small
                if sigma < 1e-6:
                    break
        
        # Fallback if no calls were made
        if self.best_x is None:
            self.best_x = np.zeros(dim)
            if self.call_count < budget:
                self.best_f = wrapped_func(self.best_x)
            else:
                self.best_f = 0.0

        return self.best_x, self.best_f
