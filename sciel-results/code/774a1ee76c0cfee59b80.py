"""Derivative-free Nelder-Mead optimizer for BBOB f18 (Schaffers F7 ill-conditioned)."""
import numpy as np
import time


class Optimizer:
    """Nelder-Mead with initial random sampling."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # safety margin before 300s limit
        
        rng = np.random.default_rng(42)
        
        x_best = None
        f_best = np.inf
        calls = 0
        
        # Phase 1: Coarse Random Search to find a good starting point
        # Use 10% of budget for random search
        n_random = min(10, budget // 10)
        
        for _ in range(n_random):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-4.0, 4.0, n)
            fx = func(x)
            calls += 1
            if fx < f_best:
                f_best = fx
                x_best = x.copy()
        
        if x_best is None:
            x_best = np.zeros(n)
            f_best = func(x_best)
            calls += 1
            
        # Phase 2: Nelder-Mead
        # Initialize simplex
        # Step size for simplex
        step = 0.5
        
        # Create initial simplex: x_best + step * e_i
        simplex = [x_best.copy()]
        for i in range(n):
            xi = x_best.copy()
            xi[i] += step
            simplex.append(xi)
        
        # Evaluate initial simplex
        f_values = []
        for i in range(n + 1):
            if calls >= budget or time.time() > deadline:
                break
            if i == 0:
                f_values.append(f_best)
            else:
                fx = func(simplex[i])
                f_values.append(fx)
                calls += 1
                if fx < f_best:
                    f_best = fx
                    x_best = simplex[i].copy()
        
        # Nelder-Mead parameters
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrink
        
        while calls < budget and time.time() < deadline:
            if time.time() > deadline - 5.0:
                break
                
            # Sort simplex by function value
            indices = np.argsort(f_values)
            simplex = [simplex[i] for i in indices]
            f_values = [f_values[i] for i in indices]
            
            x_worst = simplex[-1]
            f_worst = f_values[-1]
            
            # Check for convergence (optional, but good for efficiency)
            if np.std(f_values) < 1e-10:
                break
                
            # Centroid of all but worst
            x_centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = x_centroid + alpha * (x_centroid - x_worst)
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            calls += 1
            if f_r < f_best:
                f_best = f_r
                x_best = x_r.copy()
            
            if f_r < f_values[0]:
                # Expansion
                x_e = x_centroid + gamma * (x_r - x_centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                calls += 1
                if f_e < f_best:
                    f_best = f_e
                    x_best = x_e.copy()
                
                if f_e < f_r:
                    simplex[-1] = x_e
                    f_values[-1] = f_e
                else:
                    simplex[-1] = x_r
                    f_values[-1] = f_r
            elif f_r < f_values[-2]:
                # Simple reflection
                simplex[-1] = x_r
                f_values[-1] = f_r
            else:
                # Contraction
                if f_r < f_worst:
                    # Outside contraction
                    x_c = x_centroid + rho * (x_r - x_centroid)
                else:
                    # Inside contraction
                    x_c = x_centroid - rho * (x_r - x_centroid)
                    
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                calls += 1
                if f_c < f_best:
                    f_best = f_c
                    x_best = x_c.copy()
                
                if f_c < min(f_r, f_worst):
                    simplex[-1] = x_c
                    f_values[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        if calls >= budget or time.time() > deadline:
                            break
                        simplex[i] = x_centroid + sigma * (simplex[i] - x_centroid)
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        f_values[i] = func(simplex[i])
                        calls += 1
                        if f_values[i] < f_best:
                            f_best = f_values[i]
                            x_best = simplex[i].copy()
                    # Re-sort not strictly necessary for next iteration's logic if we just shrinked, 
                    # but the loop will re-sort at the start.
                    # However, we need to update the "worst" logic? 
                    # The loop restarts and sorts, so it's fine.
                    continue # Skip to next iteration to re-sort
            
        return x_best, f_best
