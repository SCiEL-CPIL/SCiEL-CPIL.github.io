"""New optimizer using Multi-start Simulated Annealing with Nelder-Mead polishing."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-start SA + Nelder-Mead optimizer for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # Allocate budget: 70% for global search (SA), 30% for local polish
        # We will run multiple SA chains.
        # Let's say we have enough budget for ~50 SA steps per chain and ~100 NM steps.
        # Total budget 500.
        
        best_x = None
        best_f = np.inf
        calls = 0
        
        # Helper to track calls
        def count_call(x):
            nonlocal calls
            f = func(x)
            calls += 1
            return f

        # Phase 1: Global Search via Multi-start Simulated Annealing
        # We will run several short SA chains.
        # Each chain: start random, do T steps of SA.
        # SA step: propose neighbor, accept if better or with prob exp(-delta/T).
        
        n_chains = 10
        steps_per_chain = 20 # 10 * 20 = 200 calls
        
        for i in range(n_chains):
            if calls >= budget - 50: # Leave room for final polish
                break
                
            # Start point: random in [-5, 5]^d
            x = rng.uniform(-5, 5, dim)
            f = count_call(x)
            
            if f < best_f:
                best_f = f
                best_x = x.copy()
            
            T = 1.0 # Initial temperature
            cooling = 0.9
            step_size = 1.0
            
            for step in range(steps_per_chain):
                if calls >= budget - 50:
                    break
                    
                # Propose move
                delta = rng.normal(0, step_size, dim)
                x_new = np.clip(x + delta, -5, 5)
                f_new = count_call(x_new)
                
                if f_new < f:
                    x = x_new
                    f = f_new
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                else:
                    # Accept with probability
                    prob = np.exp(-(f_new - f) / T)
                    if rng.random() < prob:
                        x = x_new
                        f = f_new
                
                T *= cooling
                # Slowly reduce step size
                step_size *= 0.95

        # Phase 2: Local Polish using Nelder-Mead on the best point found
        if best_x is not None and calls < budget:
            # Use remaining budget for NM
            # NM options: maxiter, maxfev
            # We need to ensure we don't exceed budget.
            # scipy minimize doesn't have a strict "call budget" stopper that is easy to enforce without wrapping.
            # We wrap func to count.
            
            remaining = budget - calls
            if remaining > 5:
                try:
                    res = minimize(
                        count_call, 
                        best_x, 
                        method='Nelder-Mead',
                        options={'maxiter': remaining, 'maxfev': remaining}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
