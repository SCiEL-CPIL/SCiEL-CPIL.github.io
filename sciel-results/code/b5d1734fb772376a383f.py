"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """Population-based ES with time guard and multi-start."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        
        # Budget tracking
        calls_made = 0
        max_calls = self.budget
        
        # Helper to evaluate with budget check
        def eval_with_budget(x):
            nonlocal calls_made
            if calls_made >= max_calls:
                return None
            calls_made += 1
            return func(x)

        # Multi-start strategy: try several initial points
        best_x = None
        best_f = float('inf')
        
        # Try a few random starts first to find a good region
        n_starts = 5
        for i in range(n_starts):
            if calls_made >= max_calls:
                break
            if time.time() > deadline:
                break
                
            x = rng.uniform(-5, 5, self.dim)
            fx = eval_with_budget(x)
            if fx is not None and fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        if best_x is None:
            # Fallback: single point
            x = np.zeros(self.dim)
            fx = eval_with_budget(x)
            if fx is not None:
                best_f = fx
                best_x = x.copy()
            return best_x, best_f

        # Population-based (1+lambda)-ES
        lam = 10
        sigma = 1.0
        no_improve = 0
        max_no_improve = 200
        
        while calls_made < max_calls:
            if time.time() > deadline:
                break
            
            # Generate offspring
            offspring = []
            for _ in range(lam):
                if calls_made >= max_calls:
                    break
                if time.time() > deadline:
                    break
                    
                y = np.clip(best_x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = eval_with_budget(y)
                if fy is not None:
                    offspring.append((y, fy))
            
            if not offspring:
                break
            
            # Select best offspring
            best_off = min(offspring, key=lambda t: t[1])
            
            if best_off[1] < best_f:
                best_f = best_off[1]
                best_x = best_off[0].copy()
                no_improve = 0
                # Increase sigma slightly on success
                sigma *= 1.2
            else:
                no_improve += 1
                # Decrease sigma on failure
                sigma *= 0.9
                
            # Reset if no improvement for too long
            if no_improve >= max_no_improve:
                # Try a new random start
                if calls_made >= max_calls:
                    break
                if time.time() > deadline:
                    break
                    
                x = rng.uniform(-5, 5, self.dim)
                fx = eval_with_budget(x)
                if fx is not None:
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    no_improve = 0
                    sigma = 1.0
        
        # Final polish: local search with smaller sigma
        if calls_made < max_calls and best_x is not None:
            sigma = 0.1
            for _ in range(min(50, max_calls - calls_made)):
                if time.time() > deadline:
                    break
                y = np.clip(best_x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = eval_with_budget(y)
                if fy is None:
                    break
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    sigma *= 1.1
                else:
                    sigma *= 0.95
        
        return best_x, best_f
# EVOLVE-BLOCK-END
