"""Hybrid optimizer: Nelder-Mead for robust basin finding, L-BFGS-B for fast descent, SLSQP for final polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Hybrid Nelder-Mead + L-BFGS-B + SLSQP optimizer for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        calls_made = 0
        max_calls = self.budget
        
        def time_ok():
            return time.time() < deadline

        def count_func(x):
            nonlocal calls_made
            x = np.asarray(x).flatten()
            fx = func(x)
            calls_made += 1
            return fx

        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf

        # Phase 1: Multi-start Nelder-Mead for robust basin finding
        # Nelder-Mead is more robust to hidden transformations than gradient methods
        n_starts = 8
        nm_results = []
        
        for i in range(n_starts):
            if not time_ok() or calls_made >= max_calls - 50:
                break
            
            x0 = rng.uniform(-5, 5, self.dim)
            f0 = count_func(x0)
            
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
            
            # Run a quick Nelder-Mead from this start
            nm_result = self._nelder_mead(x0, f0, func, count_func, time_ok, deadline, max_calls, calls_made)
            if nm_result is not None:
                nm_results.append(nm_result)
        
        # Sort Nelder-Mead results by function value
        nm_results.sort(key=lambda r: r[1])
        
        # Phase 2: L-BFGS-B from the best Nelder-Mead results
        # L-BFGS-B exploits the smooth structure for fast convergence
        bounds = [(-5.0, 5.0)] * self.dim
        top_k = min(3, len(nm_results))
        
        for idx in range(top_k):
            if not time_ok() or calls_made >= max_calls - 20:
                break
            
            x0 = nm_results[idx][0]
            f0 = nm_results[idx][1]
            
            def callback(xk):
                nonlocal calls_made
                if not time_ok() or calls_made >= max_calls - 10:
                    return True
                return False

            try:
                res = minimize(
                    count_func, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    callback=callback,
                    options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-10}
                )
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except Exception:
                pass

        # Phase 3: SLSQP for final high-precision polish
        # SLSQP has quadratic convergence near the optimum, ideal for the last push
        if time_ok() and calls_made < max_calls - 5 and best_x is not None:
            try:
                res_final = minimize(
                    count_func, 
                    best_x, 
                    method='SLSQP', 
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-15}
                )
                
                if res_final.fun < best_f:
                    best_f = res_final.fun
                    best_x = res_final.x.copy()
                    
            except Exception:
                pass

        # Fallback if no valid result
        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
            calls_made += 1

        return best_x, best_f

    def _nelder_mead(self, x0, f0, func, count_func, time_ok, deadline, max_calls, calls_made_ref):
        """Run Nelder-Mead from x0 with f0. Returns (x, f) or None."""
        dim = self.dim
        n = dim
        
        # Initial simplex
        simplex = [x0.copy()]
        fvals = [f0]
        
        for i in range(n):
            x_i = x0.copy()
            if x0[i] == 0:
                x_i[i] = 0.5
            else:
                x_i[i] = x0[i] + 0.5 * np.sign(x0[i])
            x_i = np.clip(x_i, -5, 5)
            simplex.append(x_i)
            fvals.append(count_func(x_i))
            
            if not time_ok() or calls_made_ref >= max_calls - 30:
                return None
        
        # Sort simplex
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrink
        
        max_iter = 50
        for _ in range(max_iter):
            if not time_ok() or calls_made_ref >= max_calls - 30:
                break
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = count_func(x_r)
            
            if f_r < fvals[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = count_func(x_e)
                if f_e < f_r:
                    simplex[-1] = x_e
                    fvals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            elif f_r < fvals[-2]:
                simplex[-1] = x_r
                fvals[-1] = f_r
            else:
                # Contraction
                if f_r < fvals[-1]:
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    x_c = centroid - rho * (simplex[-1] - centroid)
                x_c = np.clip(x_c, -5, 5)
                f_c = count_func(x_c)
                
                if f_c < min(f_r, fvals[-1]):
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = count_func(simplex[i])
                        
                        if not time_ok() or calls_made_ref >= max_calls - 30:
                            break
            
            # Check convergence
            if np.max(fvals) - np.min(fvals) < 1e-12:
                break
        
        min_idx = np.argmin(fvals)
        return (simplex[min_idx].copy(), fvals[min_idx])
# EVOLVE-BLOCK-END
