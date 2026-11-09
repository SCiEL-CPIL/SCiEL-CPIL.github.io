import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 50s safety margin before 330s kill

        dim = self.dim
        budget = self.budget

        # State tracking
        calls = 0
        best_x = None
        best_f = np.inf
        all_points = []
        all_fvals = []

        def evaluate(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline - 10:
                raise Exception("Budget or time limit exceeded")
            x = np.asarray(x, dtype=float)
            if x.shape != (dim,):
                x = x.reshape(dim)
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls += 1
            all_points.append(x.copy())
            all_fvals.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # 1. Initialization: Halton sequence + structured points
        def halton_sequence(index, base):
            q = 1.0
            r = 0.0
            while index > 0:
                q = q / base
                r = r + q * (index % base)
                index = index // base
            return r

        points = []
        n_init = 20
        for i in range(1, n_init + 1):
            p = np.array([halton_sequence(i, base) for base in range(2, 2 + dim)])
            p = p * 10.0 - 5.0
            points.append(p)

        # Add some structured points
        points.append(np.zeros(dim))
        points.append(np.ones(dim) * 3.0)
        points.append(np.ones(dim) * -3.0)

        # Evaluate initial points
        for p in points:
            if calls >= budget or time.time() > deadline - 10:
                break
            try:
                evaluate(p)
            except Exception:
                break

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls += 1

        # 2. Local Optimization using Nelder-Mead
        # Sort by function value and run NM from top candidates
        valid_indices = sorted(range(len(all_fvals)), key=lambda i: all_fvals[i])
        
        # Run NM from top 3 distinct points
        top_k = min(3, len(valid_indices))
        used_starts = set()
        
        for i in range(top_k):
            if calls >= budget or time.time() > deadline - 15:
                break
            
            idx = valid_indices[i]
            x0 = all_points[idx].copy()
            
            # Check if we've already tried a very similar start
            key = tuple(np.round(x0, 2))
            if key in used_starts:
                continue
            used_starts.add(key)
            
            # Run Nelder-Mead
            try:
                minimize(evaluate, x0, method='Nelder-Mead', 
                         options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            except Exception:
                pass

        # 3. Coordinate Descent Polish
        # Sharp ridge functions often benefit from coordinate-wise refinement
        if calls < budget - 5 and time.time() < deadline - 15:
            x_current = best_x.copy()
            f_current = best_f
            
            # Adaptive step size for coordinate descent
            step_sizes = [1.0, 0.5, 0.25, 0.1, 0.05, 0.01, 0.005, 0.001]
            
            for step in step_sizes:
                if calls >= budget or time.time() > deadline - 10:
                    break
                
                improved = True
                while improved and calls < budget and time.time() < deadline - 10:
                    improved = False
                    for d in range(dim):
                        if calls >= budget or time.time() > deadline - 10:
                            break
                        
                        # Try positive and negative steps
                        for direction in [1, -1]:
                            x_trial = x_current.copy()
                            x_trial[d] += direction * step
                            x_trial = np.clip(x_trial, -5.0, 5.0)
                            
                            # Skip if we've already evaluated this exact point
                            # (simple check: distance to all previous points)
                            skip = False
                            for prev_p in all_points:
                                if np.allclose(prev_p, x_trial, atol=1e-10):
                                    skip = True
                                    break
                            if skip:
                                continue
                            
                            try:
                                f_trial = evaluate(x_trial)
                            except Exception:
                                break
                            
                            if f_trial < f_current:
                                f_current = f_trial
                                x_current = x_trial.copy()
                                improved = True
                                break  # Restart coordinate loop after improvement
        
        # 4. Final high-precision polish with Powell if budget remains
        if calls < budget - 10 and time.time() < deadline - 15:
            try:
                minimize(evaluate, best_x, method='Powell',
                         options={'maxiter': 100, 'xtol': 1e-10, 'ftol': 1e-10})
            except Exception:
                pass

        return best_x, best_f
