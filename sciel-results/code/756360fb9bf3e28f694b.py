"""Combined optimizer: L-BFGS-B primary with structured sampling and Nelder-Mead polish."""
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """L-BFGS-B based optimizer for Rosenbrock with structured initial sampling."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0  # Safety margin before 300s limit

        rng = np.random.default_rng(42)

        # Phase 1: Structured + random initial sampling
        # Include origin, some structured points, and random samples
        n_structured = 5
        structured_points = [
            np.zeros(self.dim),
            np.ones(self.dim) * 0.5,
            np.ones(self.dim) * -0.5,
            np.ones(self.dim) * 1.0,
            np.ones(self.dim) * -1.0,
        ]
        
        n_random = min(25, max(10, self.budget // 10))
        random_points = rng.uniform(-5, 5, (n_random, self.dim))
        
        all_points = np.vstack([structured_points, random_points])
        
        # Evaluate all initial points
        fvals = np.array([func(x) for x in all_points])
        calls_used = len(all_points)
        
        best_idx = np.argmin(fvals)
        best_x = all_points[best_idx].copy()
        best_f = fvals[best_idx]
        
        if time.time() > deadline - 10:
            return best_x, best_f

        remaining = self.budget - calls_used
        
        # Phase 2: L-BFGS-B local refinement from the best point
        call_count = [0]
        
        def lbfgs_wrapper(x):
            if call_count[0] >= remaining:
                raise ValueError("Budget exceeded")
            if time.time() > deadline - 5:
                raise ValueError("Time exceeded")
            call_count[0] += 1
            return func(x)
        
        try:
            res = minimize(
                lbfgs_wrapper, 
                best_x, 
                method='L-BFGS-B',
                bounds=[(-5, 5)] * self.dim,
                options={'maxiter': 500, 'maxfun': remaining, 'maxls': 20, 'gtol': 1e-6}
            )
            
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()
                
        except Exception:
            pass
        
        remaining = self.budget - (calls_used + call_count[0])
        
        if time.time() > deadline - 15:
            return best_x, best_f

        # Phase 3: Additional local optimizations from perturbed starts if budget allows
        if remaining > 30:
            n_restarts = min(5, remaining // 15)
            for i in range(n_restarts):
                if time.time() > deadline - 10:
                    break
                if remaining < 15:
                    break
                
                # Perturb the best point with decreasing magnitude
                scale = 0.5 * (1.0 - i / n_restarts) + 0.1
                perturbation = rng.normal(0, scale, self.dim)
                start_x = np.clip(best_x + perturbation, -5, 5)
                
                restart_count = [0]
                
                def restart_wrapper(x, rc=restart_count):
                    if rc[0] >= remaining:
                        raise ValueError("Budget exceeded")
                    if time.time() > deadline - 5:
                        raise ValueError("Time exceeded")
                    rc[0] += 1
                    return func(x)
                
                try:
                    res_r = minimize(
                        restart_wrapper, 
                        start_x, 
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * self.dim,
                        options={'maxiter': 200, 'maxfun': remaining, 'maxls': 15, 'gtol': 1e-6}
                    )
                    
                    if res_r.fun < best_f:
                        best_f = res_r.fun
                        best_x = res_r.x.copy()
                        
                except Exception:
                    pass
                
                remaining = self.budget - (calls_used + call_count[0] + restart_count[0])

        # Phase 4: Final Nelder-Mead polish if budget remains
        remaining = self.budget - calls_used
        # Recalculate remaining more carefully
        total_used = calls_used + call_count[0]
        # Need to track restart calls too, but they're captured in remaining above
        # Let's just check if we have enough for a quick NM polish
        
        if remaining > 50 and time.time() < deadline - 20:
            nm_count = [0]
            nm_remaining = min(remaining - 10, 100)  # Cap NM at 100 calls
            
            def nm_wrapper(x):
                if nm_count[0] >= nm_remaining:
                    raise ValueError("NM budget exceeded")
                if time.time() > deadline - 5:
                    raise ValueError("Time exceeded")
                nm_count[0] += 1
                return func(x)
            
            try:
                res_nm = minimize(
                    nm_wrapper, 
                    best_x, 
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                
                if res_nm.fun < best_f:
                    best_f = res_nm.fun
                    best_x = res_nm.x.copy()
                    
            except Exception:
                pass

        return best_x, best_f
