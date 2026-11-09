import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 330s limit, keep 10s buffer
        deadline = start_time + 320.0 
        n_calls = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = float('inf')

        def call_f(x):
            nonlocal n_calls, best_x, best_f
            if n_calls >= budget:
                return None
            # Ensure valid shape and range
            x = np.asarray(x, dtype=float)
            if x.shape != (dim,):
                x = np.clip(x, -5.0, 5.0).reshape(dim)
            else:
                x = np.clip(x, -5.0, 5.0)
            
            n_calls += 1
            f_val = func(x)
            
            # Handle potential NaN/Inf
            if not np.isfinite(f_val):
                return float('inf')
                
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        def nelder_mead(x0, max_iters=500):
            """
            Standard Nelder-Mead Simplex method.
            Returns the best point found in this specific run.
            """
            nonlocal n_calls, best_x, best_f
            
            if n_calls >= budget:
                return best_x, best_f

            # Initialize simplex
            # Start with x0 and small perturbations
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = x0
            for i in range(1, dim + 1):
                simplex[i] = x0.copy()
                simplex[i, i-1] += 0.5 # Initial step size 0.5
            
            # Evaluate initial simplex
            f_values = np.zeros(dim + 1)
            for i in range(dim + 1):
                f_values[i] = call_f(simplex[i])
                if f_values[i] is None:
                    return best_x, best_f
            
            # Sort simplex by f values
            order = np.argsort(f_values)
            simplex = simplex[order]
            f_values = f_values[order]

            alpha = 1.0   # Reflection
            gamma = 2.0   # Expansion
            rho = 0.5     # Contraction
            sigma = 0.5   # Shrinkage

            for _ in range(max_iters):
                if n_calls >= budget:
                    break
                if time.time() > deadline - 5:
                    break

                # 1. Reflection
                x_worst = simplex[-1]
                x_best = simplex[0]
                
                # Centroid of all points except worst
                centroid = np.sum(simplex[:-1], axis=0) / dim
                
                x_ref = centroid + alpha * (centroid - x_worst)
                x_ref = np.clip(x_ref, -5.0, 5.0)
                f_ref = call_f(x_ref)
                
                if f_ref is None:
                    break

                if f_ref < f_values[0]:
                    # 2. Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5.0, 5.0)
                    f_exp = call_f(x_exp)
                    if f_exp is None:
                        break
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        f_values[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        f_values[-1] = f_ref
                elif f_ref < f_values[-2]:
                    # Accept reflection
                    simplex[-1] = x_ref
                    f_values[-1] = f_ref
                else:
                    # 3. Contraction
                    if f_ref < f_values[-1]:
                        # Outside contraction
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - rho * (x_ref - centroid)
                    
                    x_con = np.clip(x_con, -5.0, 5.0)
                    f_con = call_f(x_con)
                    if f_con is None:
                        break
                    
                    if f_con <= min(f_ref, f_values[-1]):
                        simplex[-1] = x_con
                        f_values[-1] = f_con
                    else:
                        # 4. Shrinkage
                        for i in range(1, dim + 1):
                            simplex[i] = x_best + sigma * (simplex[i] - x_best)
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            f_values[i] = call_f(simplex[i])
                            if f_values[i] is None:
                                return best_x, best_f
                        
                        # Re-sort after shrinkage
                        order = np.argsort(f_values)
                        simplex = simplex[order]
                        f_values = f_values[order]
                        continue # Skip re-sorting logic at end of loop as we just did it

                # Re-sort simplex
                order = np.argsort(f_values)
                simplex = simplex[order]
                f_values = f_values[order]
                
                # Check convergence
                if abs(f_values[0] - f_values[-1]) < 1e-8:
                    break

            return best_x, best_f

        # Strategy:
        # 1. Initial Nelder-Mead from center (0,0,0,0,0)
        # 2. Multiple Nelder-Mead restarts from random points
        # 3. Final fine-grained random walk polish

        # Phase 1: Center start
        nelder_mead(np.zeros(dim), max_iters=100)

        # Phase 2: Random restarts
        # We have 500 calls. Center NM uses ~ (dim+1) + 100*2 ~ 110 calls.
        # Remaining ~ 390 calls.
        # Each NM restart uses ~ 50-100 calls depending on convergence.
        
        max_restarts = 8
        for i in range(max_restarts):
            if n_calls >= budget - 20:
                break
            if time.time() > deadline - 10:
                break
            
            # Generate random start point
            # Use a spread distribution to cover the box
            x_rand = rng.uniform(-4.0, 4.0, dim)
            
            # Limit iterations per restart to conserve budget for polish
            # If we have many calls left, allow more iterations
            remaining = budget - n_calls
            # Heuristic: allow ~50% of remaining budget for this restart, capped
            max_iters_this = min(150, remaining // 3)
            
            if max_iters_this < 10:
                break
                
            nelder_mead(x_rand, max_iters=max_iters_this)

        # Phase 3: Final Polish
        # Use remaining budget for a very fine local search around best_x
        remaining = budget - n_calls
        if remaining > 5 and time.time() < deadline - 2:
            x = best_x.copy()
            fx = best_f
            sigma = 0.05 # Start with small step
            for _ in range(remaining):
                if time.time() > deadline - 1:
                    break
                if n_calls >= budget:
                    break
                
                # Gaussian perturbation
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = call_f(y)
                if fy is None:
                    break
                
                if fy < fx:
                    x = y
                    fx = fy
                    # Optionally increase sigma slightly if we found improvement? 
                    # No, keep shrinking or constant. Let's shrink slowly.
                    sigma *= 0.95
                else:
                    sigma *= 0.9
                    sigma = max(sigma, 1e-6)
                
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        return best_x, best_f
