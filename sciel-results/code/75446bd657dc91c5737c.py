import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 280  # 20s buffer before 300s kill
        
        calls_made = 0
        best_x = None
        best_f = float('inf')
        N = self.dim
        budget = self.budget

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if time.time() - start_time > time_limit:
                raise TimeoutError()
            if calls_made >= self.budget:
                raise RuntimeError()
            
            calls_made += 1
            try:
                f_val = func(x)
            except Exception:
                return best_f if best_f != float('inf') else 0.0
            
            if f_val < best_f:
                best_f = f_val
                best_x = np.array(x)
            return f_val

        # Phase 1: Multi-Start Global Search
        # Use 30 diverse random points for better basin coverage (~6% of budget)
        num_starts = 30
        if budget < 100:
            num_starts = 10
        
        rng = np.random.default_rng(42)
        
        start_points = []
        # Add the origin
        start_points.append(np.zeros(N))
        
        # Generate random points in [-4, 4]^N
        for _ in range(num_starts - 1):
            x = rng.uniform(-4.0, 4.0, size=N)
            start_points.append(x)
        
        # Evaluate all start points
        start_results = []
        try:
            for x in start_points:
                f_val = evaluate(x)
                start_results.append((f_val, x))
        except (TimeoutError, RuntimeError):
            pass

        if not start_results:
            return best_x, best_f

        # Sort by fitness, best first
        start_results.sort(key=lambda item: item[0])
        
        # Phase 2: Nelder-Mead local optimization on top basins
        # NM is more robust for Rosenbrock's narrow valley than gradient methods
        def nm_wrapper(x):
            nonlocal calls_made, best_x, best_f
            if time.time() - start_time > time_limit:
                raise TimeoutError()
            if calls_made >= self.budget:
                raise RuntimeError()
            
            calls_made += 1
            try:
                f_val = func(x)
            except Exception:
                return best_f if best_f != float('inf') else 0.0
            
            if f_val < best_f:
                best_f = f_val
                best_x = np.array(x)
            return f_val

        # Refine top 6 basins with Nelder-Mead
        top_basins = start_results[:6]
        
        for i, (f_init, x_init) in enumerate(top_basins):
            if calls_made >= budget or time.time() - start_time > time_limit:
                break
            
            # NM maxfev controls function evaluations; allocate more to best basin
            maxfev = 120 if i == 0 else 60
            
            try:
                res = minimize(
                    nm_wrapper, 
                    x_init, 
                    method='Nelder-Mead',
                    options={
                        'maxfev': maxfev,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.array(res.x)
                    
            except (TimeoutError, RuntimeError):
                break
            except Exception:
                pass

        # Phase 3: Final high-precision polish with L-BFGS-B
        def lbf_wrapper(x):
            nonlocal calls_made, best_x, best_f
            if time.time() - start_time > time_limit:
                raise TimeoutError()
            if calls_made >= self.budget:
                raise RuntimeError()
            
            calls_made += 1
            try:
                f_val = func(x)
            except Exception:
                return best_f if best_f != float('inf') else 0.0
            
            if f_val < best_f:
                best_f = f_val
                best_x = np.array(x)
            return f_val

        if calls_made < budget and time.time() - start_time < time_limit:
            try:
                res = minimize(
                    lbf_wrapper, 
                    best_x, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * N,
                    options={
                        'maxiter': 500,
                        'ftol': 1e-14,
                        'gtol': 1e-12,
                        'maxls': 50
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.array(res.x)
            except (TimeoutError, RuntimeError):
                pass
            except Exception:
                pass

        return best_x, best_f
