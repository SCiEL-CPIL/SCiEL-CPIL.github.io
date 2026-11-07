import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_f = float('inf')

        def evaluate(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return float('inf')
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Deterministic Multi-Start Exploration
        # Generate a diverse set of structured points to quickly locate the basin.
        # Katsuura has a global minimum at a specific point (often near -5 or shifted).
        # We test center, diagonals, and axis-aligned points.
        
        candidates = []
        
        # 1. Center
        candidates.append(np.zeros(dim))
        
        # 2. Diagonal points (all coordinates equal)
        # Katsuura's periodicity and structure might make diagonals useful.
        diag_vals = [2.0, -2.0, 4.0, -4.0, 1.0, -1.0, 3.0, -3.0, 0.5, -0.5, 1.5, -1.5]
        for v in diag_vals:
            candidates.append(np.full(dim, v))
            
        # 3. Axis points (one coordinate non-zero, others zero)
        # Helps if the function is separable or has strong single-axis features.
        for d in range(dim):
            for v in [2.0, -2.0, 4.0, -4.0, 1.0, -1.0]:
                x = np.zeros(dim)
                x[d] = v
                candidates.append(x)
                
        # 4. Random points to ensure coverage
        rng = np.random.default_rng(42)
        n_random = 30
        for _ in range(n_random):
            candidates.append(rng.uniform(-5, 5, dim))

        # Evaluate candidates
        for x in candidates:
            if time.time() > deadline or calls >= budget:
                break
            evaluate(x)
            
        if best_x is None:
            # Fallback if something went wrong (should not happen)
            best_x = np.zeros(dim)
            best_f = evaluate(best_x)

        # Phase 2: Local Refinement
        # Use the remaining budget for Nelder-Mead, which is robust and derivative-free.
        # We allocate the rest of the budget to this.
        remaining_budget = budget - calls
        if remaining_budget > 10 and time.time() < deadline:
            # Nelder-Mead is good for smooth-ish functions with local minima.
            # Katsuura is continuous but has many local minima.
            # We use the best point found so far as the starting point.
            
            # To improve robustness, we can try a few local optimizations from the top few candidates?
            # No, stick to the single best to conserve budget for fine-tuning.
            
            def obj(x):
                if time.time() > deadline:
                    raise Exception("Timeout")
                return evaluate(x)

            # Set maxfev to remaining budget to ensure we don't exceed
            options = {
                'maxiter': 10000,
                'maxfev': remaining_budget,
                'xatol': 1e-10,
                'fatol': 1e-10
            }
            
            try:
                res = minimize(obj, best_x, method='Nelder-Mead', options=options)
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                # If minimize raises an exception (e.g. due to timeout check), 
                # we rely on the best_x/best_f updated inside obj()
                pass

        return best_x, best_f
