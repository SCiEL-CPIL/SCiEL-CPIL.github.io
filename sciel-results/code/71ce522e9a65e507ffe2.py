"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid optimizer: 1-D coordinate-wise search + (1+1)-ES + L-BFGS-B polish for BBOB f20 (Schwefel)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill
        
        best_x = None
        best_f = float('inf')
        calls_made = 0
        
        def count_wrapper(x):
            nonlocal calls_made, best_x, best_f
            calls_made += 1
            fval = func(x)
            if fval < best_f:
                best_f = fval
                best_x = np.array(x, dtype=float)
            return fval
        
        # Phase 1: Coordinate-wise 1-D ternary search to estimate local centers
        # Schwefel has known periodic structure; use ternary search per dimension
        # to find a good starting point by optimizing each coordinate independently
        
        # Start from a few candidate base points
        n_base = 3
        base_points = [rng.uniform(-4, 4, dim) for _ in range(n_base)]
        
        for base_idx, base in enumerate(base_points):
            if calls_made >= budget - 50 or time.time() > deadline:
                break
            
            # For each dimension, do a quick ternary search
            x_current = base.copy()
            for d in range(dim):
                if calls_made >= budget - 50 or time.time() > deadline:
                    break
                
                # Ternary search on dimension d
                a, b = -5.0, 5.0
                for _ in range(15):  # ~15 iterations for decent precision
                    if calls_made >= budget - 50 or time.time() > deadline:
                        break
                    
                    m1 = a + (b - a) / 3.0
                    m2 = b - (b - a) / 3.0
                    
                    x1 = x_current.copy()
                    x1[d] = m1
                    f1 = count_wrapper(x1)
                    
                    x2 = x_current.copy()
                    x2[d] = m2
                    f2 = count_wrapper(x2)
                    
                    if f1 < f2:
                        b = m2
                    else:
                        a = m1
                
                # Update coordinate to midpoint of final interval
                x_current[d] = (a + b) / 2.0
            
            # Evaluate the refined point
            f_refined = count_wrapper(x_current)
        
        # Phase 2: (1+1)-ES from the best point found so far
        if best_x is not None and calls_made < budget - 20 and time.time() < deadline:
            x_es = best_x.copy()
            sigma = 1.0  # Initial step size
            
            # Run (1+1)-ES for a portion of remaining budget
            es_budget = min(100, budget - calls_made - 20)
            no_improve_count = 0
            max_no_improve = 20
            
            for _ in range(es_budget):
                if calls_made >= budget - 20 or time.time() > deadline:
                    break
                if no_improve_count >= max_no_improve:
                    break
                
                # Generate mutant
                z = rng.standard_normal(dim)
                x_mutant = np.clip(x_es + sigma * z, -5, 5)
                f_mutant = count_wrapper(x_mutant)
                
                if f_mutant <= best_f:
                    x_es = x_mutant.copy()
                    sigma = max(0.05, sigma * 1.05)  # Increase step size slightly
                    no_improve_count = 0
                else:
                    sigma = max(0.05, sigma * 0.95)  # Decrease step size
                    no_improve_count += 1
        
        # Phase 3: L-BFGS-B local refinement from best point
        if best_x is not None and calls_made < budget - 10 and time.time() < deadline:
            bounds = [(-5, 5)] * dim
            remaining_calls = budget - calls_made - 5
            
            try:
                result = minimize(
                    count_wrapper,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 50, 'maxfun': remaining_calls}
                )
            except Exception:
                pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
