"""Combined program: Multi-point (1+1)-ES with Restarts + L-BFGS-B + Nelder-Mead Refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize


class Optimizer:
    """
    Hybrid optimizer:
    1. Global phase: Multi-point (1+1)-ES with adaptive step size and restarts from best.
    2. Local phase: L-BFGS-B for efficient smooth convergence, followed by Nelder-Mead polish.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 300s limit, use 280s to be safe
        deadline = start_time + 280
        
        n = self.dim
        calls = 0
        best_x = None
        best_f = np.inf
        
        def safe_func(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            if calls >= self.budget:
                raise StopIteration("Budget exceeded")
            
            x_arr = np.asarray(x, dtype=np.float64)
            if x_arr.shape != (n,):
                x_arr = x_arr.reshape(-1)
                if x_arr.size != n:
                    raise ValueError("Wrong dimension")
            f_val = float(func(x_arr))
            calls += 1
            
            if f_val < best_f:
                best_f = f_val
                best_x = x_arr.copy()
            return f_val

        rng = np.random.default_rng(0)
        
        # Phase 1: Global Search with Multi-point (1+1)-ES and Restarts
        # Allocate 55% of budget to global exploration (leave more for local refinement)
        global_budget = int(self.budget * 0.55)
        
        # Multi-point initialization (5 points) for better early coverage
        num_starts = 5
        starts = []
        for i in range(num_starts):
            if i == 0:
                x = np.zeros(n)
            else:
                x = rng.uniform(-4, 4, n)
            try:
                fx = safe_func(x)
            except (TimeoutError, StopIteration):
                break
            starts.append((x.copy(), fx))
        
        if not starts:
            return best_x, best_f

        # Start from the best initial point
        best_start = min(starts, key=lambda t: t[1])
        x = best_start[0].copy()
        fx = best_start[1]
        
        sigma = 1.0
        stagnation = 0
        max_stagnation = 40
        
        # Track best in current run for restarts
        run_best_x = x.copy()
        run_best_f = fx
        
        # Track diverse good points for multi-start local refinement
        diverse_points = []  # list of (x, f) pairs, kept diverse
        diversity_threshold = 1.0  # minimum distance between stored points
        
        def is_diverse(x_new, existing_points, thresh):
            """Check if x_new is sufficiently different from all existing points."""
            for (x_old, _) in existing_points:
                if np.linalg.norm(x_new - x_old) < thresh:
                    return False
            return True

        # Add initial best to diverse list
        diverse_points.append((x.copy(), fx))

        for _ in range(global_budget):
            if calls >= self.budget or time.time() > deadline:
                break
                
            # Generate candidate
            y = x + sigma * rng.standard_normal(n)
            y = np.clip(y, -5, 5)
            
            try:
                fy = safe_func(y)
            except (TimeoutError, StopIteration):
                break
                
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.2
                if sigma > 4.0:
                    sigma = 4.0
                stagnation = 0
                if fx < run_best_f:
                    run_best_f = fx
                    run_best_x = x.copy()
                # Add to diverse list if sufficiently different
                if len(diverse_points) < 8 and is_diverse(x, diverse_points, diversity_threshold):
                    diverse_points.append((x.copy(), fx))
            else:
                sigma *= 0.9
                if sigma < 0.05:
                    sigma = 0.05
                stagnation += 1
                
                # If stagnating, restart from global best with larger step
                if stagnation > max_stagnation:
                    if best_x is not None:
                        x = best_x.copy()
                        fx = best_f
                    else:
                        x = rng.uniform(-4, 4, n)
                        try:
                            fx = safe_func(x)
                        except (TimeoutError, StopIteration):
                            break
                    sigma = 2.0
                    stagnation = 0
                    # Reduce threshold for more aggressive restarts later
                    max_stagnation = max(12, max_stagnation // 2)
                    # Add restart point to diverse list
                    if len(diverse_points) < 8 and is_diverse(x, diverse_points, diversity_threshold):
                        diverse_points.append((x.copy(), fx))

        # Phase 2: Multi-start local refinement
        # Sort diverse points by function value and refine the best ones
        diverse_points.sort(key=lambda t: t[1])
        
        # Use L-BFGS-B on the best point first, then Nelder-Mead on top diverse points
        if calls < self.budget and time.time() < deadline and best_x is not None:
            remaining_calls = self.budget - calls
            
            # Step 2a: L-BFGS-B on the single best point
            if remaining_calls > 15:
                max_iter = max(5, remaining_calls // 4)
                try:
                    result = scipy.optimize.minimize(
                        safe_func, 
                        best_x, 
                        method='L-BFGS-B', 
                        bounds=[(-5, 5)] * n,
                        options={
                            'maxiter': max_iter,
                            'ftol': 1e-10,
                            'gtol': 1e-8,
                            'maxls': 15
                        }
                    )
                except (TimeoutError, StopIteration, Exception):
                    pass
        
        # Step 2b: Nelder-Mead on top diverse points (excluding the one already refined)
        if calls < self.budget - 5 and time.time() < deadline and best_x is not None:
            remaining_calls = self.budget - calls
            if remaining_calls > 10:
                # Allocate remaining budget among up to 3 diverse points
                num_local_starts = min(3, len(diverse_points))
                calls_per_start = max(5, remaining_calls // num_local_starts)
                
                for i in range(num_local_starts):
                    if calls >= self.budget or time.time() > deadline:
                        break
                    
                    start_x, start_f = diverse_points[i]
                    
                    # Skip if this is essentially the same as current best
                    if np.linalg.norm(start_x - best_x) < 0.1:
                        continue
                    
                    max_iter = max(3, calls_per_start // (n + 1))
                    try:
                        result2 = scipy.optimize.minimize(
                            safe_func, 
                            start_x, 
                            method='Nelder-Mead', 
                            options={
                                'maxiter': max_iter,
                                'xatol': 1e-5,
                                'fatol': 1e-5,
                                'adaptive': True
                            }
                        )
                    except (TimeoutError, StopIteration, Exception):
                        pass
        
        # Phase 3: Final tight Nelder-Mead polish on the overall best
        if calls < self.budget - 3 and time.time() < deadline and best_x is not None:
            remaining_calls = self.budget - calls
            if remaining_calls > 5:
                max_iter = max(3, remaining_calls // (n + 2))
                try:
                    result3 = scipy.optimize.minimize(
                        safe_func, 
                        best_x, 
                        method='Nelder-Mead', 
                        options={
                            'maxiter': max_iter,
                            'xatol': 1e-7,
                            'fatol': 1e-7,
                            'adaptive': True
                        }
                    )
                except (TimeoutError, StopIteration, Exception):
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
