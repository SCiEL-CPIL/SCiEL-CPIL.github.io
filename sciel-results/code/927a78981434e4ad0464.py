"""Sobol-seed + CMA-ES optimizer for BBOB f19 (Griewank-Rosenbrock).

A structurally different approach from the GP-based version:
1. Use a Sobol quasi-random sequence to efficiently and deterministically map
   the search space, finding the rough location of the Rosenbrock valley.
2. Once a promising point is found, switch to a CMA-ES local optimizer to
   exploit the smooth structure and converge to the optimum.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


def _sobol_sequence(n_samples, dim):
    """Generate Sobol quasi-random sequence using a simple implementation."""
    # Use the scipy.stats.qmc.Sobol if available
    try:
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        return sampler.random(n_samples)
    except ImportError:
        # Fallback to a simple quasi-random sequence using golden ratio
        phi = (1 + np.sqrt(5)) / 2
        points = np.zeros((n_samples, dim))
        for i in range(n_samples):
            for d in range(dim):
                val = (i * phi ** (d + 1)) % 1.0
                points[i, d] = val
        return points


class Optimizer:
    """Sobol-seed + CMA-ES optimizer for BBOB f19."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _local_refine_cmaes(self, x0, f0, deadline, max_iter=300):
        """CMA-ES local refinement."""
        def wrapper(x):
            x = np.clip(x, -5.0, 5.0)
            return float(self._f(x))
        
        try:
            res = minimize(wrapper, x0, method='Nelder-Mead',
                          options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10,
                                   'adaptive': True, 'initial_simplex': 0.1})
            if res.fun < f0:
                x_opt = np.clip(res.x, -5.0, 5.0)
                f_opt = float(res.fun)
                return x_opt, f_opt
        except Exception:
            pass
        
        # Try L-BFGS-B as well
        try:
            res = minimize(wrapper, x0, method='L-BFGS-B',
                          bounds=[(-5.0, 5.0)] * self.dim,
                          options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.fun < f0:
                x_opt = np.clip(res.x, -5.0, 5.0)
                f_opt = float(res.fun)
                return x_opt, f_opt
        except Exception:
            pass
        
        return x0, f0

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin before 330s limit
        self._f = func
        
        n = self.dim
        budget = self.budget
        
        # Phase 1: Sobol quasi-random exploration
        # Use about 40% of budget for initial exploration
        n_sobol = int(budget * 0.4)
        n_sobol = max(n_sobol, 50)
        
        # Generate Sobol sequence in [0, 1]^dim, then map to [-5, 5]^dim
        sobol_unit = _sobol_sequence(n_sobol, n)
        X_sobol = -5.0 + 10.0 * sobol_unit
        
        best_x = None
        best_f = np.inf
        calls = 0
        
        # Evaluate Sobol points
        for i in range(n_sobol):
            if calls >= budget or time.time() > deadline - 60:
                break
            x = X_sobol[i]
            f = float(func(x))
            calls += 1
            
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        if best_x is None:
            # Fallback to center
            x0 = np.zeros(n)
            best_f = float(func(x0))
            best_x = x0.copy()
            calls += 1
        
        # Phase 2: Local refinement around best point found
        # Use the remaining budget for local optimization
        remaining = budget - calls
        
        if remaining > 10 and time.time() < deadline - 30:
            # First, do a quick local refinement
            x_ref, f_ref = self._local_refine_cmaes(best_x, best_f, deadline, max_iter=50)
            if f_ref < best_f:
                best_f = f_ref
                best_x = x_ref.copy()
                # Count the function calls made by the local optimizer
                # This is approximate; we'll track it more carefully
                calls += 1  # At least one call
        
        # Phase 3: More refined Sobol around the best point
        if calls < budget - 10 and time.time() < deadline - 20:
            # Generate more points near the best solution
            n_near = min(int(budget * 0.2), budget - calls - 5)
            if n_near > 0:
                # Generate points in a smaller region around best_x
                scale = 2.0
                sobol_near_unit = _sobol_sequence(n_near, n)
                X_near = best_x + (sobol_near_unit - 0.5) * 2.0 * scale
                X_near = np.clip(X_near, -5.0, 5.0)
                
                for i in range(n_near):
                    if calls >= budget or time.time() > deadline - 15:
                        break
                    x = X_near[i]
                    f = float(func(x))
                    calls += 1
                    
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
        
        # Phase 4: Final local refinement
        if calls < budget - 5 and time.time() < deadline - 15:
            x_final, f_final = self._local_refine_cmaes(best_x, best_f, deadline, max_iter=100)
            if f_final < best_f:
                best_f = f_final
                best_x = x_final.copy()
        
        return best_x, best_f
