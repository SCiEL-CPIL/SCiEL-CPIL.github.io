import numpy as np
import time
import itertools
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
        rng = np.random.default_rng(42)

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

        # Phase 1: Random Multi-Start
        # Use a small number of random starts to cover the space efficiently
        num_starts = 10
        start_points = []
        # Include origin and a few deterministic points
        start_points.append(np.zeros(N))
        start_points.append(np.ones(N) * 2.0)
        start_points.append(np.ones(N) * -2.0)
        start_points.append(np.ones(N) * 4.0)
        start_points.append(np.ones(N) * -4.0)
        
        for _ in range(num_starts - len(start_points)):
            start_points.append(rng.uniform(-5, 5, size=N))

        # Evaluate starts and sort
        start_results = []
        try:
            for x in start_points:
                f_val = evaluate(x)
                start_results.append((f_val, x))
        except (TimeoutError, RuntimeError):
            pass

        start_results.sort(key=lambda x: x[0])
        
        # Phase 2: Local Polish with L-BFGS-B
        # Run L-BFGS-B from the best starting points
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

        # Allocate budget for local searches
        # We have 500 calls. ~15 used for starts.
        # Let's try 5 local searches.
        local_starts = [x for _, x in start_results[:5]]
        
        for i, start_x in enumerate(local_starts):
            if calls_made >= budget or time.time() - start_time > time_limit:
                break
            
            # Limit iterations to ensure we finish multiple starts
            max_iter = 100 if i == 0 else 50
            
            try:
                res = minimize(
                    lbf_wrapper, 
                    start_x, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * N,
                    options={
                        'maxiter': max_iter,
                        'ftol': 1e-12,
                        'gtol': 1e-10,
                        'maxls': 20
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.array(res.x)
                    
            except (TimeoutError, RuntimeError):
                break
            except Exception:
                pass

        # Phase 3: Nelder-Mead Polish
        # Nelder-Mead can refine the solution further, especially near the optimum
        if calls_made < budget and time.time() - start_time < time_limit and best_x is not None:
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

            try:
                # Nelder-Mead doesn't support bounds directly, so we rely on the fact
                # that the function clips inputs. However, it might wander out of bounds.
                # The task says func(x) clips a finite length-5 vector to the box.
                # So we are safe.
                res = minimize(
                    nm_wrapper, 
                    best_x, 
                    method='Nelder-Mead',
                    options={
                        'maxiter': 500,
                        'xatol': 1e-8,
                        'fatol': 1e-12,
                        'adaptive': True
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
