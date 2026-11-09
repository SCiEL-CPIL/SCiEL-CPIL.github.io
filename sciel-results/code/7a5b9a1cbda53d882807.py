import numpy as np
import time
from scipy.stats.qmc import Sobol

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin within 300s limit
        
        calls_made = 0
        best_x = None
        best_f = np.inf
        rng = np.random.default_rng(42)

        # Wrapper to track calls, time, and best result
        def wrapped_func(x):
            nonlocal calls_made, best_x, best_f
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            if calls_made >= self.budget:
                raise Exception("Budget exceeded")
            
            calls_made += 1
            # Ensure x is a 1-D numpy array of correct shape
            x = np.asarray(x, dtype=float).flatten()
            if x.shape[0] != self.dim:
                # Pad or truncate if needed (shouldn't happen)
                if x.shape[0] < self.dim:
                    x = np.pad(x, (0, self.dim - x.shape[0]))
                else:
                    x = x[:self.dim]
            
            val = func(x)
            
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Phase 1: Global Exploration with Sobol Sequence
        # Use 100 points for global search in 5D
        n_global = min(100, self.budget)
        
        try:
            # Generate Sobol points
            sampler = Sobol(d=self.dim, scramble=True, seed=12345)
            # Generate n_global points in [0, 1]^d
            sobol_points = sampler.random(n_global)
            # Map to [-5, 5]^d
            sobol_points = sobol_points * 10.0 - 5.0
            
            for x in sobol_points:
                wrapped_func(x)
                
        except (TimeoutError, Exception):
            pass

        # If we haven't made any calls (should not happen), return
        if best_x is None:
             # Fallback: try center
             x0 = np.zeros(self.dim)
             try:
                 wrapped_func(x0)
             except:
                 pass
             if best_x is None:
                 return np.zeros(self.dim), np.inf

        # Phase 2: Local Refinement with Nelder-Mead
        # Use remaining budget
        remaining_budget = self.budget - calls_made
        
        if remaining_budget > 0:
            try:
                # Nelder-Mead options
                # xatol and fatol control convergence
                # maxiter is high, but budget/time guards will stop it
                # We implement a custom Nelder-Mead to avoid scipy overhead and allow tighter control
                x0 = best_x.copy()
                # Initial simplex
                simplex = [x0.copy()]
                for i in range(self.dim):
                    xi = x0.copy()
                    # Step size: 0.05 if 0, else 0.05 * abs(xi[i])
                    step = 0.05 if abs(xi[i]) < 1e-8 else 0.05 * abs(xi[i])
                    xi[i] += step
                    simplex.append(xi)
                
                # Evaluate initial simplex
                f_values = []
                for i in range(self.dim + 1):
                    f_val = wrapped_func(simplex[i])
                    f_values.append(f_val)
                
                # Nelder-Mead loop
                while calls_made < self.budget and time.time() < deadline:
                    # Sort simplex by f-value
                    indices = np.argsort(f_values)
                    simplex = [simplex[i].copy() for i in indices]
                    f_values = [f_values[i] for i in indices]
                    
                    # Check convergence
                    if abs(f_values[-1] - f_values[0]) < 1e-10:
                        break
                    if np.max(np.abs(np.array(simplex) - np.array(simplex[0]))) < 1e-10:
                        break
                    
                    centroid = np.mean(simplex[:-1], axis=0)
                    
                    # Reflection
                    xr = centroid + 1.0 * (centroid - simplex[-1])
                    fr = wrapped_func(xr)
                    
                    if fr < f_values[0]:
                        # Expansion
                        xe = centroid + 2.0 * (centroid - simplex[-1])
                        fe = wrapped_func(xe)
                        if fe < fr:
                            simplex[-1] = xe
                            f_values[-1] = fe
                        else:
                            simplex[-1] = xr
                            f_values[-1] = fr
                    elif fr < f_values[-2]:
                        # Reflection
                        simplex[-1] = xr
                        f_values[-1] = fr
                    else:
                        # Contraction
                        # Try inside contraction first
                        xc = centroid + 0.5 * (simplex[-1] - centroid)
                        fc = wrapped_func(xc)
                        if fc < fr:
                            simplex[-1] = xc
                            f_values[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, self.dim + 1):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                f_values[i] = wrapped_func(simplex[i])
                
            except (TimeoutError, Exception):
                pass

        # Phase 3: Final Polish with smaller steps
        if calls_made < self.budget and time.time() < deadline:
            try:
                current_x = best_x.copy()
                current_f = best_f
                # Use a very small step size for final polish
                step_sizes = [1e-3, 1e-4, 1e-5, 1e-6]
                for step in step_sizes:
                    if calls_made >= self.budget or time.time() > deadline:
                        break
                    for _ in range(10):
                        if calls_made >= self.budget or time.time() > deadline:
                            break
                        # Random direction
                        direction = rng.normal(size=self.dim)
                        direction /= np.linalg.norm(direction)
                        
                        # Try step forward and backward
                        x_plus = current_x + step * direction
                        x_minus = current_x - step * direction
                        
                        f_plus = wrapped_func(x_plus)
                        f_minus = wrapped_func(x_minus)
                        
                        if f_plus < current_f:
                            current_x = x_plus
                            current_f = f_plus
                        elif f_minus < current_f:
                            current_x = x_minus
                            current_f = f_minus
                        else:
                            # No improvement, try a different direction
                            continue
            except (TimeoutError, Exception):
                pass

        return best_x, best_f
