import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 270  # 30s buffer before 300s kill
        
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

        # Phase 1: Global Search using Latin Hypercube Sampling (LHS)
        # LHS provides better coverage of the domain than random sampling for small N
        num_global = 50
        if budget < 100:
            num_global = 20
            
        rng = np.random.default_rng(42)
        
        # Generate LHS points
        lhs_points = np.zeros((num_global, N))
        for dim in range(N):
            # Create strata centers
            strata = (np.arange(num_global) + 0.5) / num_global
            # Random offset within stratum
            offsets = rng.uniform(-0.5, 0.5, size=num_global) / num_global
            points_dim = strata + offsets
            # Shuffle
            rng.shuffle(points_dim)
            lhs_points[:, dim] = points_dim
        
        # Map [0,1] to [-5,5]
        lhs_points = lhs_points * 10.0 - 5.0
        
        # Add explicit center and corners to be safe
        explicit_points = [
            np.zeros(N),
            np.ones(N) * 5.0,
            np.ones(N) * -5.0,
            np.ones(N) * 2.5,
            np.ones(N) * -2.5
        ]
        
        all_initial_points = np.vstack([lhs_points] + [np.array(p) for p in explicit_points])
        
        # Evaluate all initial points
        initial_results = []
        try:
            for x in all_initial_points:
                f_val = evaluate(x)
                initial_results.append((f_val, x))
        except (TimeoutError, RuntimeError):
            pass

        if not initial_results:
            return best_x, best_f

        # Sort by fitness
        initial_results.sort(key=lambda item: item[0])
        
        # Phase 2: Local Optimization with Powell's Method
        # Powell is robust and derivative-free, often outperforms Nelder-Mead for Rosenbrock
        # We run it on the top 3 candidates
        
        top_candidates = initial_results[:3]
        
        for i, (f_init, x_init) in enumerate(top_candidates):
            if calls_made >= budget or time.time() - start_time > time_limit:
                break
            
            # Powell Method
            try:
                res_powell = minimize(
                    evaluate, 
                    x_init, 
                    method='Powell', 
                    bounds=[(-5, 5)] * N,
                    options={
                        'maxiter': 200,
                        'ftol': 1e-12,
                        'xtol': 1e-12
                    }
                )
                if res_powell.fun < best_f:
                    best_f = res_powell.fun
                    best_x = np.array(res_powell.x)
            except (TimeoutError, RuntimeError):
                break
            except Exception:
                pass

            # Phase 3: Polish with L-BFGS-B
            # L-BFGS-B is very fast for final convergence
            if calls_made < budget and time.time() - start_time < time_limit:
                try:
                    res_lbf = minimize(
                        evaluate, 
                        best_x, 
                        method='L-BFGS-B', 
                        bounds=[(-5, 5)] * N,
                        options={
                            'maxiter': 100,
                            'ftol': 1e-14,
                            'gtol': 1e-12
                        }
                    )
                    if res_lbf.fun < best_f:
                        best_f = res_lbf.fun
                        best_x = np.array(res_lbf.x)
                except (TimeoutError, RuntimeError):
                    break
                except Exception:
                    pass

        # Phase 4: If we have budget left, try a second local optimization from the next best global point
        if len(initial_results) > 3 and calls_made < budget and time.time() - start_time < time_limit:
            next_best = initial_results[3]
            try:
                res_powell2 = minimize(
                    evaluate, 
                    next_best[1], 
                    method='Powell', 
                    bounds=[(-5, 5)] * N,
                    options={
                        'maxiter': 100,
                        'ftol': 1e-12,
                        'xtol': 1e-12
                    }
                )
                if res_powell2.fun < best_f:
                    best_f = res_powell2.fun
                    best_x = np.array(res_powell2.x)
                    
                    # Polish again
                    if calls_made < budget:
                        res_lbf2 = minimize(
                            evaluate, 
                            best_x, 
                            method='L-BFGS-B', 
                            bounds=[(-5, 5)] * N,
                            options={
                                'maxiter': 50,
                                'ftol': 1e-14,
                                'gtol': 1e-12
                            }
                        )
                        if res_lbf2.fun < best_f:
                            best_f = res_lbf2.fun
                            best_x = np.array(res_lbf2.x)
            except (TimeoutError, RuntimeError):
                pass
            except Exception:
                pass

        return best_x, best_f
