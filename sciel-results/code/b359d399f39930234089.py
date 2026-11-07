"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Nelder-Mead based optimizer for BBOB f14 with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 40s safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        # Phase 1: Diversified initialization
        n_init = min(8, budget)
        rng = np.random.default_rng(42)
        candidates = [np.zeros(dim)]
        for i in range(min(3, dim)):
            c = np.zeros(dim)
            c[i] = rng.choice([-4.0, 4.0])
            candidates.append(c)
        while len(candidates) < n_init:
            candidates.append(rng.uniform(-4.5, 4.5, dim))

        best_x = None
        best_f = np.inf
        calls_made = 0
        for c in candidates:
            if time.time() > deadline or calls_made >= budget:
                break
            c = np.clip(c, -5, 5)
            fc = func(c)
            calls_made += 1
            if fc < best_f:
                best_f = fc
                best_x = c.copy()

        if calls_made >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: Nelder-Mead with adaptive restarts
        # Use scipy's Nelder-Mead which has superlinear convergence
        # Wrap func to count calls and enforce deadline
        
        nm_calls = [0]
        
        def nm_func(x):
            if time.time() > deadline or nm_calls[0] >= budget - calls_made:
                return np.inf
            x_clipped = np.clip(x, -5, 5)
            val = func(x_clipped)
            nm_calls[0] += 1
            return val
        
        # Run Nelder-Mead with initial simplex around best_x
        # Initial simplex: center at best_x, with small perturbations
        x0 = best_x.copy()
        simplex_size = 0.5  # Initial simplex scale
        
        # Create initial simplex for Nelder-Mead
        # scipy's minimize with Nelder-Mead uses x0 as the first vertex
        # We need to set initial_simplex for better control
        initial_simplex = [x0]
        for i in range(dim):
            v = x0.copy()
            v[i] += simplex_size
            v = np.clip(v, -5, 5)
            initial_simplex.append(v)
        
        # Run Nelder-Mead
        options = {
            'maxiter': 200,
            'xatol': 1e-8,
            'fatol': 1e-10,
            'adaptive': True
        }
        
        try:
            result = minimize(nm_func, x0, method='Nelder-Mead', 
                            options=options,
                            initial_simplex=np.array(initial_simplex))
            if result.fun < best_f and np.isfinite(result.fun):
                best_x = result.x.copy()
                best_f = result.fun
        except Exception:
            pass

        calls_made += nm_calls[0]

        # Phase 3: If we have remaining budget, do local refinement
        if calls_made < budget and time.time() < deadline:
            # Try a few more Nelder-Mead runs from different starting points
            remaining_budget = budget - calls_made
            if remaining_budget > 50:
                # Restart from best_x with smaller simplex
                nm_calls[0] = 0
                x0 = best_x.copy()
                simplex_size = 0.1
                initial_simplex = [x0]
                for i in range(dim):
                    v = x0.copy()
                    v[i] += simplex_size
                    v = np.clip(v, -5, 5)
                    initial_simplex.append(v)
                
                options2 = {
                    'maxiter': 100,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
                
                try:
                    result2 = minimize(nm_func, x0, method='Nelder-Mead',
                                     options=options2,
                                     initial_simplex=np.array(initial_simplex))
                    if result2.fun < best_f and np.isfinite(result2.fun):
                        best_x = result2.x.copy()
                        best_f = result2.fun
                except Exception:
                    pass
                calls_made += nm_calls[0]

        # Phase 4: Coordinate descent polish if budget remains
        if calls_made < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.05
            
            while calls_made < budget and time.time() < deadline:
                improved = False
                for d in range(dim):
                    if calls_made >= budget or time.time() > deadline:
                        break
                    for direction in [-1, 1]:
                        trial = x.copy()
                        trial[d] = x[d] + direction * sigma
                        trial[d] = np.clip(trial[d], -5, 5)
                        if time.time() > deadline or calls_made >= budget:
                            break
                        ft = func(trial)
                        calls_made += 1
                        if ft < fx:
                            x = trial
                            fx = ft
                            improved = True
                            best_x = x.copy()
                            best_f = fx
                    if time.time() > deadline or calls_made >= budget:
                        break
                if not improved:
                    sigma *= 0.5
                    if sigma < 1e-6:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
