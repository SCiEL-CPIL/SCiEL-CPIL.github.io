"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Adaptive Simulated Annealing (ASA) optimizer."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s limit
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        
        # Multi-start with different initial temperatures
        n_starts = 5
        calls_made = 0
        
        for start_idx in range(n_starts):
            # Check time budget
            if time.time() > deadline:
                break
            
            # Initialize with a random point
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            # ASA parameters
            T = 5.0 + start_idx * 2.0  # Initial temperature varies per start
            alpha = 0.95  # Cooling rate
            min_T = 1e-6
            
            # Run ASA until time or local budget exhausted
            local_budget = max(1, (budget - calls_made) // (n_starts - start_idx))
            
            for step in range(local_budget):
                if time.time() > deadline:
                    break
                if calls_made >= budget:
                    break
                
                # Generate candidate: x' = x + delta * (randn - 0.5) * T
                # ASA direction: deterministic gradient-free update
                delta = rng.standard_normal(dim) * T
                x_new = np.clip(x + delta, -5, 5)
                fx_new = func(x_new)
                calls_made += 1
                
                if fx_new < fx:
                    # Accept improvement
                    x, fx = x_new, fx_new
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    # Accept with probability exp(-delta_f / T)
                    delta_f = fx_new - fx
                    if delta_f > 0:
                        accept_prob = np.exp(-delta_f / max(T, min_T))
                        if rng.random() < accept_prob:
                            x, fx = x_new, fx_new
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                    
                    # Cool down
                    T *= alpha
                    if T < min_T:
                        T = min_T
        
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
