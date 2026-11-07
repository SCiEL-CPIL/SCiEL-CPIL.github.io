import numpy as np
import scipy.optimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.start_time = time.time()
        # 330s limit, use 320s for safety margin
        self.deadline = self.start_time + 320

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        # Use a fixed seed for reproducibility in evaluation, but Sobol is deterministic anyway
        # We just need to ensure we don't exceed budget.
        
        eval_count = 0
        best_x = None
        best_f = np.inf
        candidates = []

        def counted_func(x):
            nonlocal eval_count, best_f, best_x
            if time.time() > self.deadline:
                raise TimeoutError
            if eval_count >= budget:
                raise RuntimeError("Budget exceeded")
            
            x = np.asarray(x, dtype=float)
            x_clipped = np.clip(x, -5.0, 5.0)
            
            eval_count += 1
            fx = func(x_clipped)
            
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            
            candidates.append((fx, x_clipped.copy()))
            
            return fx

        # Phase 1: Global Search using Sobol sequences
        # Sobol provides better coverage than uniform random for low dimensions
        try:
            # scipy.stats.qmc.Sobol is available in recent SciPy versions
            # If not, we fall back to Halton or simple random
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
            n_global = 30
            # Generate points in [0, 1]^dim and scale to [-5, 5]^dim
            points = sampler.random(n_global)
            points = points * 10.0 - 5.0
        except ImportError:
            # Fallback to Halton or Random
            rng = np.random.default_rng(42)
            n_global = 30
            points = rng.uniform(-5.0, 5.0, (n_global, dim))

        for x in points:
            if time.time() > self.deadline or eval_count >= budget:
                break
            try:
                counted_func(x)
            except (TimeoutError, RuntimeError):
                break
            except Exception:
                continue

        if best_x is None:
            return np.zeros(dim), np.inf

        # Sort candidates to find the best starting points for local search
        candidates.sort(key=lambda item: item[0])
        
        # Phase 2: Local Refinement
        # We use Powell's method as it is derivative-free and robust to the valley shape.
        # Then we polish with L-BFGS-B for high precision.
        
        bounds = [(-5.0, 5.0)] * dim
        top_candidates = candidates[:5]
        
        best_local_x = best_x
        best_local_f = best_f
        
        for idx, (f_val, x_start) in enumerate(top_candidates):
            if time.time() > self.deadline or eval_count >= budget - 10:
                break
            
            remaining_budget = budget - eval_count
            if remaining_budget <= 20:
                break
                
            # Try Powell's method
            # Powell is good for the Rosenbrock valley
            try:
                res_powell = scipy.optimize.minimize(
                    counted_func,
                    x_start,
                    method='Powell',
                    bounds=bounds,
                    options={'maxiter': 50, 'xtol': 1e-8, 'ftol': 1e-8}
                )
                if res_powell.fun < best_local_f:
                    best_local_f = res_powell.fun
                    best_local_x = res_powell.x
            except (TimeoutError, RuntimeError):
                break
            except Exception:
                continue

            # If we have budget left, polish with L-BFGS-B
            if eval_count < budget - 10 and time.time() < self.deadline:
                try:
                    res_lbgfs = scipy.optimize.minimize(
                        counted_func,
                        best_local_x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxfun': budget - eval_count - 5, 'maxiter': 100, 'gtol': 1e-10}
                    )
                    if res_lbgfs.fun < best_local_f:
                        best_local_f = res_lbgfs.fun
                        best_local_x = res_lbgfs.x
                except (TimeoutError, RuntimeError):
                    break
                except Exception:
                    continue
            
            # If we found a very good solution, we might stop early to save time
            # Rosenbrock minimum is 0. If we are close, stop.
            if best_local_f < 1e-3:
                break

        return best_local_x, best_local_f
