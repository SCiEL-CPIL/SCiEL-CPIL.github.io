"""Derivative-Free Coordinate Descent with multi-resolution grid for Lunacek bi-Rastrigin."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-resolution coordinate descent optimizer for bi-modal functions."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initialize at center (Lunacek has a global minimum near center and a local minimum offset)
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        # Phase 1: Coarse grid scan per dimension to find promising regions
        # Use a multi-resolution approach: start coarse, refine locally
        
        # First, do a quick random restart to see if we can find a better region
        # Lunacek bi-Rastrigin has two minima: one at x0 (global) and one at x0 + shift (local)
        # The global minimum is typically closer to the center in BBOB instances
        
        # Phase 1: Coarse coordinate scan
        # For each dimension, try a few points to find the best axis-aligned line
        resolutions = [5, 10, 20, 40, 80]
        
        for res in resolutions:
            if calls >= budget or time.time() > deadline:
                break
            
            for d in range(dim):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Try res points along dimension d, spaced across [-5, 5]
                # But focus around current best to be efficient
                # Use a window around current x[d] with some exploration
                center = x[d]
                spread = 10.0 / res  # Full range divided by resolution
                
                for i in range(res):
                    if calls >= budget or time.time() > deadline:
                        break
                    
                    # Sample point along dimension d
                    new_x = x.copy()
                    # Distribute points across [-5, 5] but weight towards center
                    t = (i + 0.5) / res
                    new_x[d] = -5 + 10 * t
                    
                    # Clip to bounds
                    new_x[d] = np.clip(new_x[d], -5, 5)
                    
                    fy = func(new_x)
                    calls += 1
                    
                    if fy < fx:
                        x = new_x
                        fx = fy
                        if fy < best_fx:
                            best_x = x.copy()
                            best_fx = fy
        
        # Phase 2: Local refinement with shrinking step sizes
        if calls < budget and time.time() < deadline:
            step = 1.0
            no_improve_count = 0
            max_no_improve = 50
            
            while calls < budget and time.time() < deadline and step > 1e-6:
                improved = False
                for d in range(dim):
                    if calls >= budget or time.time() > deadline:
                        break
                    
                    # Try positive and negative steps
                    for sign in [1, -1]:
                        if calls >= budget or time.time() > deadline:
                            break
                        
                        new_x = x.copy()
                        new_x[d] += sign * step
                        new_x[d] = np.clip(new_x[d], -5, 5)
                        
                        fy = func(new_x)
                        calls += 1
                        
                        if fy < fx:
                            x = new_x
                            fx = fy
                            improved = True
                            if fy < best_fx:
                                best_x = x.copy()
                                best_fx = fy
                            # Don't break; try other directions too
                        else:
                            # If no improvement, try smaller step in this direction
                            pass
                
                if improved:
                    no_improve_count = 0
                    step *= 0.9  # Slow shrink when improving
                else:
                    no_improve_count += 1
                    step *= 0.7  # Faster shrink when stuck
                    if no_improve_count > max_no_improve:
                        break
        
        # Phase 3: Final random restarts if budget remains
        if calls < budget and time.time() < deadline:
            # Try a few random restarts to escape local minima
            n_restarts = min(10, (budget - calls) // 10)
            for _ in range(n_restarts):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Random point, biased towards center
                x_r = rng.uniform(-2, 2, dim)
                fx_r = func(x_r)
                calls += 1
                
                if fx_r < best_fx:
                    best_x = x_r.copy()
                    best_fx = fx_r
                    # Do a quick local refinement from this point
                    x = x_r.copy()
                    fx = fx_r
                    step = 0.5
                    for _ in range(min(50, budget - calls)):
                        if calls >= budget or time.time() > deadline:
                            break
                        improved = False
                        for d in range(dim):
                            if calls >= budget or time.time() > deadline:
                                break
                            for sign in [1, -1]:
                                if calls >= budget or time.time() > deadline:
                                    break
                                new_x = x.copy()
                                new_x[d] += sign * step
                                new_x[d] = np.clip(new_x[d], -5, 5)
                                fy = func(new_x)
                                calls += 1
                                if fy < fx:
                                    x = new_x
                                    fx = fy
                                    improved = True
                                    if fy < best_fx:
                                        best_x = x.copy()
                                        best_fx = fy
                        if not improved:
                            step *= 0.5
                            if step < 1e-4:
                                break
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
