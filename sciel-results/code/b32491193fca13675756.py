"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-Start Adaptive (1+1)-ES with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin before 300s kill
        
        n = self.dim
        total_budget = self.budget
        calls_made = 0
        best_x = None
        best_f = np.inf
        
        def safe_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= total_budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x = np.clip(x, -5, 5)
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)
        
        # Phase 1: Coarse Initialization
        # Evaluate a few strategic points to find a good basin
        # 1. Origin
        # 2. Random points
        init_points = []
        init_points.append(np.zeros(n))
        for _ in range(9):
            init_points.append(rng.uniform(-4, 4, n))
            
        for p in init_points:
            fp = safe_func(p)
        
        # If no valid point was found (should not happen), return
        if best_x is None:
            return np.zeros(n), np.inf

        x = best_x.copy()
        fx = best_f
        sigma = 2.0
        min_sigma = 1e-6
        max_sigma = 10.0
        
        # Phase 2: Adaptive (1+1)-ES with Restarts
        # We will run this until budget or time is exhausted
        # Restart strategy: If no improvement for 20 steps, or sigma < min_sigma, restart
        # from best_x with a larger sigma, or from a new random point.
        
        no_improve_count = 0
        restart_count = 0
        max_restarts = 5 # Limit restarts to avoid wasting budget
        
        while calls_made < total_budget and time.time() < deadline:
            y = x + sigma * rng.standard_normal(n)
            fy = safe_func(y)
            
            if fy == np.inf:
                break
            
            if fy < fx:
                x = y
                fx = fy
                # Success: increase sigma
                sigma *= 1.1
                no_improve_count = 0
            else:
                # Failure: decrease sigma
                sigma *= 0.8
                no_improve_count += 1
            
            # Check for stall
            if no_improve_count > 30 or sigma < min_sigma:
                restart_count += 1
                if restart_count > max_restarts:
                    break
                
                # Restart strategy:
                # 1. Try a new random point if we haven't explored much
                # 2. Or restart from best_x with larger sigma
                
                # Let's try a new random point to escape local minima
                # But keep the global best
                x = rng.uniform(-4, 4, n)
                fx = safe_func(x)
                
                if fx == np.inf:
                    break
                
                if fx < best_f:
                    best_x = x.copy()
                    best_f = fx
                
                # Reset sigma for new start
                sigma = 2.0
                no_improve_count = 0
                
        # Phase 3: Polish
        # If we have budget left, do a final polish around the global best
        if best_x is not None and calls_made < total_budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            # Run a short (1+1)-ES with small sigma
            while calls_made < total_budget and time.time() < deadline:
                y = x + sigma * rng.standard_normal(n)
                fy = safe_func(y)
                if fy == np.inf:
                    break
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.05
                else:
                    sigma *= 0.9
                    if sigma < 1e-8:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
