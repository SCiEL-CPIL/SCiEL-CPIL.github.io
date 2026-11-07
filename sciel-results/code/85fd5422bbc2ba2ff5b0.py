import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325  # Safety margin before 330s kill
        
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls = 0
        x_history = []
        f_history = []

        def wrapper(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                return 1e9
            if calls >= budget:
                return 1e9
            calls += 1
            x = np.clip(x, -5, 5)
            try:
                f = float(func(x))
            except:
                f = 1e9

            x_history.append(x.copy())
            f_history.append(f)
            
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Initial Space-Filling Sampling
        # Use a mix of Sobol-like and random points for good coverage
        n_init = min(30, budget)
        
        # Generate initial points: mix of random and structured
        points = []
        # Random points
        for _ in range(n_init // 2):
            points.append(np.random.uniform(-5, 5, dim))
        # Structured points (grid-like with jitter)
        for _ in range(n_init // 2):
            pt = np.random.uniform(-4, 4, dim)
            points.append(pt)
            
        # Evaluate initial points
        for pt in points:
            if calls >= budget or time.time() > deadline:
                break
            wrapper(pt)
            
        if best_x is None:
            return best_x, best_f

        # Phase 2: Iterative Local Refinement with Global Exploration
        # Strategy: 
        # 1. Refine the current best with Nelder-Mead
        # 2. Occasionally explore new regions to escape local minima
        # 3. Use the best found so far as the anchor
        
        remaining_budget = budget - calls
        n_local_trials = 0
        
        while calls < budget and time.time() < deadline:
            n_local_trials += 1
            
            # Check for stagnation: if we haven't improved in the last ~50 calls, explore
            if len(f_history) > 50:
                recent_best = min(f_history[-50:])
                if best_f - recent_best < 1e-6 and n_local_trials % 3 == 0:
                    # Explore: pick a random point or a point far from current best
                    explore_x = np.random.uniform(-5, 5, dim)
                    f_explore = wrapper(explore_x)
                    if f_explore < best_f:
                        # New best found, continue refining from here
                        pass
                    continue
            
            # Local refinement from current best
            # Use a smaller step size as we get closer to the end
            progress = calls / budget
            xatol = 1e-8 * (1 - progress * 0.5)
            fatol = 1e-10 * (1 - progress * 0.5)
            
            # Limit iterations based on remaining budget
            maxiter = max(1, min(50, (budget - calls) // 10))
            
            try:
                res = minimize(
                    wrapper, 
                    best_x, 
                    method='Nelder-Mead',
                    options={
                        'maxiter': maxiter,
                        'xatol': xatol,
                        'fatol': fatol,
                        'adaptive': True
                    }
                )
            except Exception:
                pass
            
            # After each local refinement, consider a small perturbation exploration
            if calls < budget - 20 and n_local_trials % 2 == 0:
                # Perturb the best point slightly
                scale = max(0.1, 1.0 - progress)
                perturbed = best_x + np.random.normal(0, scale, dim)
                perturbed = np.clip(perturbed, -5, 5)
                wrapper(perturbed)
        
        return best_x, best_f
