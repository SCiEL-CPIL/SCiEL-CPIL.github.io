import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        calls = 0
        start_time = time.time()
        DEADLINE = 280.0

        best_x = None
        best_f = float('inf')

        def evaluate(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return float('inf')
            if (time.time() - start_time) > DEADLINE:
                return float('inf')
            
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            calls += 1
            fx = float(func(x))
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.RandomState(42)

        # --- Phase 1: Global Sampling ---
        n_initial = 50
        points = []
        for i in range(n_initial):
            # Latin Hypercube Sampling
            x = np.zeros(dim)
            for d in range(dim):
                interval = rng.randint(n_initial)
                x[d] = -5.0 + (interval + rng.rand()) * (10.0 / n_initial)
            x = np.clip(x, -4.0, 4.0)
            
            if calls >= budget or (time.time() - start_time) > DEADLINE:
                break
            
            fx = evaluate(x)
            points.append((x, fx))

        # Sort points by fitness
        points.sort(key=lambda p: p[1])
        
        # Queue of points to polish
        polish_queue = [p[0] for p in points[:10]]
        
        # --- Phase 2: Local Search ---
        while calls < budget * 0.9 and (time.time() - start_time) < DEADLINE:
            if not polish_queue:
                # Generate new random points to explore
                for _ in range(5):
                    x = rng.uniform(-4.0, 4.0, size=dim)
                    fx = evaluate(x)
                    polish_queue.append(x)
                # Sort and keep best 5
                # We don't have f for these new points in a list, so we just use them as is
                # A better way: evaluate, store, sort, then queue top 5
                # For simplicity, we just add them.
                
            x_start = polish_queue.pop(0)
            
            # Run Nelder-Mead
            def nm_func(x):
                return evaluate(x)
            
            try:
                # Limit iterations to save budget
                res = minimize(
                    nm_func, 
                    x_start, 
                    method='Nelder-Mead', 
                    options={'maxiter': 30, 'xatol': 1e-5, 'fatol': 1e-5}
                )
            except Exception:
                pass

            # If we have budget left, add a perturbed version of the best point to the queue
            if best_x is not None and calls < budget * 0.9:
                x_perturbed = best_x + rng.normal(0, 0.5, size=dim)
                x_perturbed = np.clip(x_perturbed, -4.0, 4.0)
                polish_queue.append(x_perturbed)
            else:
                x_new = rng.uniform(-4.0, 4.0, size=dim)
                polish_queue.append(x_new)

        return best_x, best_f
