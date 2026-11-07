import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        calls_made = 0
        best_f = float('inf')
        best_x = None
        start_time = time.time()
        deadline = start_time + 285.0

        # Wrapper to count calls and enforce deadline
        def wrapper(x):
            nonlocal calls_made
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            if calls_made >= total_budget:
                raise TimeoutError("Budget exceeded")
            calls_made += 1
            x = np.clip(x, -5.0, 5.0)
            return float(func(x))

        rng = np.random.default_rng(42)

        # Phase 1: Deterministic Sobol sequence for global exploration
        # Use 64 points from a Sobol sequence for uniform coverage
        n_global = 64
        sobol = qmc.Sobol(d=dim, scramble=False, seed=42)
        # Generate 2^6 = 64 points
        sobol_points = sobol.random(n_global)
        
        # Map from [0,1]^dim to [-5,5]^dim
        for i in range(n_global):
            if calls_made >= total_budget or time.time() > deadline:
                break
            x = sobol_points[i] * 10.0 - 5.0
            try:
                fx = wrapper(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            except TimeoutError:
                break

        if best_x is None:
            return np.zeros(dim), best_f

        # Phase 2: Multi-scale stochastic hill-climbing with decaying sigma
        # Weierstrass function has self-similar structure, so we use multiple scales
        
        # Start with a medium step size
        sigma = 1.5
        min_sigma = 1e-7
        max_sigma = 2.0
        
        stagnation_count = 0
        consecutive_rejections = 0
        
        # Adaptive parameters
        shrink_factor = 0.85
        expand_factor = 1.05
        tunnel_threshold = 50
        tunnel_sigma = 0.8
        
        # Also keep a secondary best for potential tunneling
        second_best_f = float('inf')
        second_best_x = None
        
        while calls_made < total_budget and time.time() < deadline:
            # Generate trial point with multi-scale perturbation
            # Occasionally use a larger jump to escape local minima
            if rng.random() < 0.05 and sigma > 0.3:
                # Large jump
                x_trial = best_x + 2.0 * sigma * rng.standard_normal(dim)
            else:
                x_trial = best_x + sigma * rng.standard_normal(dim)
            
            try:
                fx_trial = wrapper(x_trial)
            except TimeoutError:
                break
            
            if fx_trial < best_f:
                # Improvement
                # Update second best if current best was second best
                if second_best_x is not None and fx_trial < second_best_f:
                    if best_f < second_best_f:
                        second_best_f = best_f
                        second_best_x = best_x.copy()
                
                best_f = fx_trial
                best_x = x_trial.copy()
                stagnation_count = 0
                consecutive_rejections = 0
                # Expand sigma slightly to explore around new best
                sigma = min(sigma * expand_factor, max_sigma)
            else:
                # No improvement
                consecutive_rejections += 1
                
                # Track second best
                if fx_trial < second_best_f:
                    second_best_f = fx_trial
                    second_best_x = x_trial.copy()
                
                # Adaptive shrinking
                if consecutive_rejections > 8:
                    sigma = max(sigma * shrink_factor, min_sigma)
                    consecutive_rejections = 0
                    
                # Stagnation detection
                stagnation_count += 1
                
                # Tunneling mechanism:
                # If we are stuck for a long time, try a jump from second best
                if stagnation_count > tunnel_threshold and second_best_x is not None:
                    # Jump from second best with a different scale
                    x_jump = second_best_x + tunnel_sigma * rng.standard_normal(dim)
                    try:
                        fx_jump = wrapper(x_jump)
                    except TimeoutError:
                        break
                    
                    if fx_jump < best_f:
                        best_f = fx_jump
                        best_x = x_jump.copy()
                        stagnation_count = 0
                        sigma = tunnel_sigma
                    else:
                        # Reset stagnation counter
                        stagnation_count = 0
                        # Shrink sigma more aggressively
                        sigma = max(sigma * 0.7, min_sigma)

        return best_x, best_f
