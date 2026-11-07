"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-phase optimizer: coarse exploration, (1+1)-ES, and local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget
        
        def time_ok():
            return time.time() < deadline

        # Phase 1: Coarse exploration with multiple random restarts
        # Use ~30% of budget for exploration
        n_explore = max(5, budget // 4)
        best_x = None
        best_f = np.inf
        
        for i in range(n_explore):
            if not time_ok():
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        if best_x is None:
            best_x = rng.uniform(-5, 5, dim)
            best_f = func(best_x)
        
        calls_made = n_explore
        
        # Phase 2: (1+1)-ES with adaptive step size
        # Use ~50% of budget for main optimization
        n_es = max(5, budget // 2)
        x = best_x.copy()
        fx = best_f
        sigma = 1.5
        
        for i in range(n_es):
            if not time_ok():
                break
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_made += 1
            
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.2
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                sigma *= 0.85
                
            if sigma < 1e-10:
                break
        
        # Phase 3: Local refinement with smaller steps
        # Use remaining budget for fine-tuning
        remaining = budget - calls_made
        if remaining > 0 and time_ok():
            sigma = max(1e-4, sigma * 0.5)
            for i in range(remaining):
                if not time_ok():
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9
                    
                if sigma < 1e-12:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
