import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 285.0
        calls_made = 0
        best_x = None
        best_f = np.inf
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                raise StopIteration
            if time.time() - start_time > time_limit:
                raise StopIteration
            calls_made += 1
            fx = float(func(x))
            if fx < best_f:
                best_f = fx
                best_x = np.array(x)
            return fx

        # Phase 1: Diversified initial sampling
        n_initial = min(20, budget)
        for i in range(n_initial):
            if time.time() - start_time > time_limit or calls_made >= budget:
                break
            if i < 5:
                x = rng.uniform(-5, 5, dim)
            elif i < 10:
                x = rng.normal(0, 2, dim)
                x = np.clip(x, -5, 5)
            elif i < 15:
                # Sparse structured points
                x = rng.choice([-4, -2, 0, 2, 4], dim)
            else:
                x = rng.uniform(-5, 5, dim)
            try:
                f = safe_call(x)
            except StopIteration:
                break
            except Exception:
                continue

        if best_x is None:
            try:
                best_x = np.zeros(dim)
                best_f = float(func(best_x))
                calls_made += 1
            except Exception:
                best_x = np.zeros(dim)
                best_f = np.inf

        # Phase 2: Iterative local optimization from diverse starts
        # Use L-BFGS-B from multiple starting points
        n_starts = min(30, budget - calls_made - 50)
        
        # Generate diverse starting points
        starts = []
        # Best point and its neighborhood
        for _ in range(5):
            perturbation = rng.normal(0, 0.5, dim)
            x = np.clip(best_x + perturbation, -5, 5)
            starts.append(x)
        
        # Random points
        for _ in range(min(10, max(0, n_starts - 5))):
            x = rng.uniform(-5, 5, dim)
            starts.append(x)
        
        # Gaussian points
        for _ in range(min(10, max(0, n_starts - 15))):
            x = np.clip(rng.normal(0, 2, dim), -5, 5)
            starts.append(x)
        
        # Structured points based on Rosenbrock valley
        for _ in range(min(5, max(0, n_starts - 20))):
            # Try to be near the valley: x_i ≈ x_{i-1}^2
            x = rng.uniform(-2, 2, dim)
            for i in range(1, dim):
                x[i] = x[i-1]**2
            x = np.clip(x, -5, 5)
            starts.append(x)

        # Run L-BFGS-B from each start
        bounds = [(-5.0, 5.0) for _ in range(dim)]
        for i, x0 in enumerate(starts):
            if time.time() - start_time > time_limit or calls_made >= budget - 20:
                break
            
            try:
                def lbfun(x_in):
                    return safe_call(x_in)
                
                res = minimize(
                    lbfun,
                    x0=x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'maxfun': budget - calls_made, 'gtol': 1e-8}
                )
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
                    # If significant improvement, do additional polish
                    if calls_made < budget - 50 and time.time() - start_time < time_limit:
                        try:
                            def lbfun2(x_in):
                                return safe_call(x_in)
                            
                            res2 = minimize(
                                lbfun2,
                                x0=best_x,
                                method='L-BFGS-B',
                                bounds=bounds,
                                options={'maxiter': 500, 'maxfun': budget - calls_made, 'gtol': 1e-10}
                            )
                            if res2.fun < best_f:
                                best_f = res2.fun
                                best_x = res2.x.copy()
                        except StopIteration:
                            pass
                        except Exception:
                            pass
            except StopIteration:
                break
            except Exception:
                pass

        # Phase 3: Final high-precision polish with remaining budget
        if best_x is not None and calls_made < budget and time.time() - start_time < time_limit:
            try:
                def lbfun_final(x_in):
                    return safe_call(x_in)
                
                res_final = minimize(
                    lbfun_final,
                    x0=best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 1000, 'maxfun': budget - calls_made, 'gtol': 1e-12}
                )
                if res_final.fun < best_f:
                    best_f = res_final.fun
                    best_x = res_final.x.copy()
            except StopIteration:
                pass
            except Exception:
                pass

        return best_x, best_f
