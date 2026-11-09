"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid Optimizer for BBOB f20 (Schwefel).
    Combines stratified global sampling with (1+1)-ES local optimization and restarts.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit

        # State tracking
        best_x_global = np.zeros(dim)
        best_f_global = np.inf
        call_count = 0

        def safe_call(x):
            nonlocal call_count, best_x_global, best_f_global
            if call_count >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            
            x = np.asarray(x, dtype=np.float64).flatten()
            x = np.clip(x, -5.0, 5.0)
            
            try:
                fx = float(func(x))
            except Exception:
                return np.inf
            
            if not np.isfinite(fx):
                fx = np.inf
                
            call_count += 1
            if fx < best_f_global:
                best_f_global = fx
                best_x_global = x.copy()
            return fx

        rng = np.random.default_rng(42)

        # Phase 1: Stratified Global Exploration
        # We want to cover the space efficiently. 
        # For dim=5, 3^5=243 is too many for just exploration if we want ES time.
        # Let's use a smaller set, e.g., 30-50 points, stratified.
        
        n_global = 40
        # Create stratified points: divide each dimension into k bins
        # k such that k^dim is close to n_global? 
        # 2^5=32, 3^5=243. 
        # Let's use 2 bins per dim -> 32 points, plus some random.
        # Or just random but ensure spread? 
        # Stratified: for each of the 32 cells, pick a random point.
        
        global_points = []
        n_bins = 2
        from itertools import product
        grid_indices = list(product(range(n_bins), repeat=dim))
        
        for idx in grid_indices:
            point = np.zeros(dim)
            for d in range(dim):
                lo = -5.0 + 10.0 * idx[d] / n_bins
                hi = -5.0 + 10.0 * (idx[d] + 1) / n_bins
                point[d] = rng.uniform(lo, hi)
            global_points.append(point)
        
        # Add a few fully random points for extra diversity
        for _ in range(n_global - len(grid_indices)):
            global_points.append(rng.uniform(-5.0, 5.0, dim))

        # Shuffle to avoid bias in order
        rng.shuffle(global_points)

        for x in global_points:
            if time.time() > deadline or call_count >= budget:
                break
            safe_call(x)

        # Phase 2: (1+1)-ES with Restarts
        # Strategy:
        # - Start from current best.
        # - Run ES for a fixed number of steps (e.g., 50).
        # - If stagnation (no improvement in last 20 steps) or sigma too small, restart.
        # - Restart from: 50% random global point, 50% perturbation of best.
        
        remaining_budget = budget - call_count
        if remaining_budget <= 0:
            return best_x_global, best_f_global

        # ES Parameters
        # Initial sigma: large enough to move, small enough to converge.
        # Domain is [-5, 5]. Diagonal is ~11. 
        # Start with sigma = 1.0
        
        stagnation_limit = 30
        es_chunk_size = 50
        
        while call_count < budget and time.time() < deadline:
            # Initialize ES state
            x = best_x_global.copy()
            fx = best_f_global
            sigma = 1.0
            no_improve_count = 0
            steps_in_chunk = 0
            
            # Run ES for this chunk
            while steps_in_chunk < es_chunk_size and call_count < budget and time.time() < deadline:
                # Generate candidate
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                
                fy = safe_call(y)
                steps_in_chunk += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.1  # Increase step size on success
                    no_improve_count = 0
                else:
                    sigma *= 0.9  # Decrease step size on failure
                    no_improve_count += 1
                
                # Check for stagnation within the chunk
                if no_improve_count >= stagnation_limit:
                    break

            # If we broke out due to stagnation or budget, decide whether to restart
            if call_count >= budget or time.time() > deadline:
                break
            
            # Check if we should restart
            # If sigma is very small, we've converged locally.
            # If no_improve_count is high, we're stuck.
            if sigma < 1e-3 or no_improve_count >= stagnation_limit:
                # Restart
                # 50% chance to jump to a new random point, 50% to perturb best
                if rng.random() < 0.5:
                    x = rng.uniform(-5.0, 5.0, dim)
                    # We don't evaluate x here immediately to save calls? 
                    # No, we need fx to start the next ES run.
                    # But safe_call increments count. 
                    # Let's just let the next ES loop handle the first step.
                    # However, the next ES loop starts with x and fx=best_f_global?
                    # No, if we jump to a random point, we don't know its fx yet.
                    # We must evaluate it.
                    
                    # To be efficient, we can just set x to the new point and 
                    # let the first step of the next ES loop evaluate it?
                    # No, (1+1)-ES needs a current best to compare against.
                    # So we must evaluate the restart point.
                    
                    # Actually, if we restart from a random point, we lose the 
                    # "current best" context for the ES step. 
                    # The ES logic above assumes `fx` is the value of `x`.
                    # So we must evaluate `x` first.
                    
                    # But wait, if we evaluate a random point and it's worse, 
                    # we've wasted a call and our `x` is now a bad point.
                    # The ES will then try to improve from that bad point.
                    # This is valid, but might be slow.
                    
                    # Alternative: Restart from a perturbation of the GLOBAL best.
                    # This maintains a high-quality starting point.
                    # Let's do this: 80% perturb global best, 20% random.
                    
                    if rng.random() < 0.8:
                        x = best_x_global + 1.0 * rng.standard_normal(dim)
                        x = np.clip(x, -5.0, 5.0)
                    else:
                        x = rng.uniform(-5.0, 5.0, dim)
                    
                    # Evaluate the restart point
                    if call_count < budget and time.time() < deadline:
                        fx = safe_call(x)
                    else:
                        break
                        
                    # Reset sigma for the new run
                    sigma = 1.0
                else:
                    # Continue with same x if not stuck? 
                    # But we only break the inner loop if stuck.
                    # If not stuck, we just finished a chunk. 
                    # We can just continue the outer loop, which re-initializes?
                    # No, if we don't restart, we should continue the ES trajectory.
                    # The current structure breaks the inner loop and goes to outer.
                    # The outer loop resets x to best_x_global. This is a "soft restart" 
                    # to the best found so far. This is safe.
                    pass

        return best_x_global, best_f_global

# EVOLVE-BLOCK-END
