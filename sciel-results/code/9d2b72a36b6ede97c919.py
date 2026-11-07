import numpy as np
import scipy.optimize
import scipy.stats
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: Evaluator kills at 330s. We aim to finish well before.
        # 290s gives a 40s buffer for overhead/exceptions.
        deadline = start_time + 290.0 
        call_count = 0
        best_f = float('inf')
        best_x = None

        def wrapped_func(x):
            nonlocal call_count, best_f, best_x
            # Check budget and time
            if call_count >= self.budget:
                return 1e10
            if time.time() > deadline:
                return 1e10
            
            call_count += 1
            try:
                # Ensure x is a numpy array and clip to bounds
                x_arr = np.asarray(x, dtype=float)
                if x_arr.shape != (self.dim,):
                    # Reshape if necessary, though we control inputs mostly
                    x_arr = x_arr.flatten()[:self.dim]
                x_clipped = np.clip(x_arr, -5, 5)
                
                fx = float(func(x_clipped))
                
                if np.isnan(fx) or np.isinf(fx):
                    return 1e10
                
                if fx < best_f:
                    best_f = fx
                    best_x = x_clipped.copy()
                
                return fx
            except Exception:
                return 1e10

        # Phase 1: Global Initialization
        # Knowledge suggests ~100 starts are better than 30.
        # We use a mix of Sobol (low discrepancy) and Random (stochastic) to cover the space.
        n_initial = 100
        if self.budget < n_initial:
            n_initial = self.budget - 10 # Leave some for local search if budget is tight
        
        # Generate Sobol points
        try:
            sobol = scipy.stats.qmc.Sobol(d=self.dim, scramble=True, seed=42)
            # Take half from Sobol
            n_sobol = n_initial // 2
            points_sobol = sobol.random(n_sobol) * 10.0 - 5.0
        except Exception:
            points_sobol = np.empty((0, self.dim))
            n_sobol = 0

        # Generate Random points for the rest
        n_random = n_initial - n_sobol
        rng = np.random.default_rng(42)
        points_random = rng.uniform(-5, 5, (n_random, self.dim))
        
        # Combine
        all_initial_points = np.vstack([points_sobol, points_random])
        
        # Shuffle to avoid bias if the first Sobol points happen to be good/bad
        rng.shuffle(all_initial_points)

        candidates = []
        
        # Evaluate initial points
        for x in all_initial_points:
            if call_count >= self.budget or time.time() > deadline:
                break
            fx = wrapped_func(x)
            candidates.append((fx, x))
        
        if not candidates:
            return best_x, best_f

        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])
        
        # Phase 2: Local Optimization on Top Candidates
        # Select top K. With 100 starts, the top 5-10 are likely in the basin of the global minimum
        # for the easier functions. For Rastrigin/Gallagher, more starts help find the right basin.
        # We have roughly 200 calls left.
        # L-BFGS-B in 5D typically converges in ~20-50 function evaluations.
        # We can afford about 5-10 local optimizations.
        
        remaining_budget = self.budget - call_count
        top_k = min(8, len(candidates))
        
        # Allocate budget per local search
        # Reserve 10 calls for final polish
        local_budget_total = max(0, remaining_budget - 10)
        local_budget_per = local_budget_total // top_k if top_k > 0 else 0
        
        local_results = []
        
        for i in range(top_k):
            if call_count >= self.budget or time.time() > deadline:
                break
            
            start_x = candidates[i][1]
            
            # Use L-BFGS-B for smooth functions (Sphere, Rosenbrock)
            # It is efficient and handles bounds well.
            try:
                res = scipy.optimize.minimize(
                    wrapped_func,
                    start_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={
                        'maxiter': 50, 
                        'maxfun': local_budget_per, 
                        'ftol': 1e-10, 
                        'gtol': 1e-8, 
                        'disp': False
                    }
                )
                if res.fun < 1e9: # Valid result
                    local_results.append((res.fun, res.x))
            except Exception:
                # If L-BFGS-B fails, keep the initial point
                local_results.append((candidates[i][0], start_x))
        
        # Update best
        if local_results:
            local_results.sort(key=lambda t: t[0])
            best_local_f, best_local_x = local_results[0]
            if best_local_f < best_f:
                best_f = best_local_f
                best_x = best_local_x

        # Phase 3: Final Polish
        # If we have budget left, do a high-precision polish on the best point.
        # Nelder-Mead is derivative-free and robust, good for final touches.
        # Or a tighter L-BFGS-B.
        if call_count < self.budget - 5 and time.time() < deadline - 2:
            try:
                # Try Nelder-Mead with strict tolerances
                # maxfev should be small as we are just polishing
                polish_budget = min(20, self.budget - call_count)
                
                res = scipy.optimize.minimize(
                    wrapped_func,
                    best_x,
                    method='Nelder-Mead',
                    bounds=[(-5, 5)] * self.dim,
                    options={
                        'maxiter': 100,
                        'maxfev': polish_budget,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'disp': False
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_f
