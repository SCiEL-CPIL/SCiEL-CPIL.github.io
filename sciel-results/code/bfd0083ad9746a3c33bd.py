"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with coordinate descent polish and stagnation-triggered restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        deadline = time.time() + 280  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        
        # Phase 1: Multi-restart (1+1)-ES with stagnation detection
        # Reserve ~40% of budget for coordinate descent polish
        es_budget = int(self.budget * 0.6)
        cd_budget = self.budget - es_budget
        
        # Divide ES budget among restarts
        n_restarts = min(5, es_budget // 50 + 1)
        budget_per_restart = es_budget // n_restarts
        
        for r in range(n_restarts):
            if time.time() > deadline:
                break
                
            # Initialize: first restart at center, others random
            if r == 0:
                x = np.zeros(self.dim)
            else:
                x = rng.uniform(-4, 4, self.dim)
            
            fx = func(x)
            if fx < best_f:
                best_f, best_x = fx, x.copy()
            
            sigma = 2.0
            no_improve = 0
            max_no_improve = 30  # stagnation detection
            
            for i in range(budget_per_restart):
                if time.time() > deadline:
                    break
                if no_improve >= max_no_improve:
                    break
                    
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                    no_improve = 0
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                else:
                    sigma *= 0.9
                    no_improve += 1
        
        # Phase 2: Coordinate descent polish on best solution found
        if best_x is not None and cd_budget > 0:
            x = best_x.copy()
            fx = best_f
            
            # Use remaining budget for coordinate descent with adaptive steps
            calls_used = 0
            step_sizes = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
            
            for step in step_sizes:
                if time.time() > deadline or calls_used >= cd_budget:
                    break
                
                improved = True
                while improved and calls_used < cd_budget and time.time() < deadline:
                    improved = False
                    for d in range(self.dim):
                        if calls_used >= cd_budget or time.time() > deadline:
                            break
                        
                        # Try positive step
                        y = x.copy()
                        y[d] = np.clip(y[d] + step, -5, 5)
                        fy = func(y)
                        calls_used += 1
                        
                        if fy < fx:
                            x, fx = y, fy
                            improved = True
                            if fx < best_f:
                                best_f, best_x = fx, x.copy()
                            continue
                        
                        # Try negative step
                        y = x.copy()
                        y[d] = np.clip(y[d] - step, -5, 5)
                        fy = func(y)
                        calls_used += 1
                        
                        if fy < fx:
                            x, fx = y, fy
                            improved = True
                            if fx < best_f:
                                best_f, best_x = fx, x.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
