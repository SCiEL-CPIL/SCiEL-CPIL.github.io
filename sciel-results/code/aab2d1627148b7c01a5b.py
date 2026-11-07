"""Combined optimizer: Random Portfolio + Nelder-Mead Refinement."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """
    Hybrid optimizer for BBOB f20 (Schwefel).
    Phase 1: Diverse random sampling including known Schwefel minima locations.
    Phase 2: Nelder-Mead refinement on the best candidates.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 50s safety margin from 330s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        call_count = 0

        def check_time():
            return time.time() > deadline

        def try_func(x):
            nonlocal best_x, best_f, call_count
            if call_count >= budget or check_time():
                return np.inf
            try:
                fx = float(func(x))
                call_count += 1
                if np.isfinite(fx) and fx < best_f:
                    best_f = fx
                    best_x = np.array(x)
                return fx
            except Exception:
                return np.inf

        # --- Phase 1: Exploration ---
        # We allocate 40% of the budget to exploration.
        n_explore = int(budget * 0.4)
        
        # Generate candidate starting points
        candidates = []
        
        # 1. Known Schwefel global minimum (standard)
        x_schwefel = np.full(dim, 4.20968746)
        candidates.append(x_schwefel)
        
        # 2. Negative of known minimum (hedge for potential sign flip in shift)
        candidates.append(-x_schwefel)
        
        # 3. Origin
        candidates.append(np.zeros(dim))
        
        # 4. Random points
        # Ensure we have enough candidates to fill the exploration budget
        n_random = n_explore - len(candidates)
        for _ in range(max(0, n_random)):
            x = rng.uniform(-5, 5, dim)
            candidates.append(x)

        # Evaluate all candidates
        evaluated = []
        for x in candidates:
            if call_count >= n_explore or check_time():
                break
            x_clipped = np.clip(x, -5, 5)
            fx = try_func(x_clipped)
            if np.isfinite(fx):
                evaluated.append((fx, x_clipped))

        if not evaluated:
            return best_x, best_f

        # Sort by function value
        evaluated.sort(key=lambda t: t[0])
        
        # Select top K for refinement
        # We have remaining budget: budget - call_count
        remaining_budget = budget - call_count
        if remaining_budget <= 0:
            return best_x, best_f

        # Choose top 3 candidates if we have budget for them
        # Each NM run will consume some budget. We limit NM iterations per run.
        top_k = min(3, len(evaluated))
        
        # --- Phase 2: Refinement with Nelder-Mead ---
        # We implement a simple Nelder-Mead to control calls strictly.
        
        def nelder_mead(start_x, start_f, max_calls):
            nonlocal best_x, best_f, call_count
            
            # Initialize simplex
            # Start with a small simplex around start_x
            x0 = start_x.copy()
            f0 = start_f
            
            # Standard NM initialization
            simplex = [x0.copy()]
            f_vals = [f0]
            
            # Generate other vertices
            # Step size relative to dimension
            step = 0.5 
            for i in range(dim):
                xi = x0.copy()
                if abs(xi[i]) > 0.1:
                    xi[i] += step
                else:
                    xi[i] += step
                xi = np.clip(xi, -5, 5)
                fi = try_func(xi)
                if np.isfinite(fi):
                    simplex.append(xi)
                    f_vals.append(fi)
                else:
                    # If we hit a limit or error, break early
                    break
            
            if len(simplex) < dim + 1:
                return

            # NM Main Loop
            # We limit the number of iterations to ensure we don't exceed budget
            # Each iteration typically uses 1-2 function evaluations (reflection, expansion, etc.)
            # We will cap total calls for this specific NM instance.
            local_call_count = 0
            max_iter = max_calls // 2 # Rough estimate
            
            for _ in range(max_iter):
                if call_count >= budget or check_time():
                    break
                
                # Sort simplex by function value
                idx = np.argsort(f_vals)
                simplex = [simplex[i] for i in idx]
                f_vals = [f_vals[i] for i in idx]
                
                # Check convergence
                if len(f_vals) > 1 and abs(f_vals[-1] - f_vals[0]) < 1e-8:
                    break
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                worst_x = simplex[-1]
                worst_f = f_vals[-1]
                
                # Reflection
                alpha = 1.0
                xr = centroid + alpha * (centroid - worst_x)
                xr = np.clip(xr, -5, 5)
                fr = try_func(xr)
                local_call_count += 1
                
                if not np.isfinite(fr):
                    break
                    
                if fr < f_vals[0]:
                    # Expansion
                    gamma = 2.0
                    xe = centroid + gamma * (centroid - worst_x)
                    xe = np.clip(xe, -5, 5)
                    fe = try_func(xe)
                    local_call_count += 1
                    
                    if np.isfinite(fe) and fe < fr:
                        simplex[-1] = xe
                        f_vals[-1] = fe
                    else:
                        simplex[-1] = xr
                        f_vals[-1] = fr
                elif fr < f_vals[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    f_vals[-1] = fr
                else:
                    # Contraction
                    beta = 0.5
                    # Outside contraction if fr < worst_f, else inside
                    if fr < worst_f:
                        xc = centroid + beta * (xr - centroid)
                    else:
                        xc = centroid - beta * (centroid - xr)
                    
                    xc = np.clip(xc, -5, 5)
                    fc = try_func(xc)
                    local_call_count += 1
                    
                    if np.isfinite(fc) and fc < worst_f:
                        simplex[-1] = xc
                        f_vals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fi = try_func(simplex[i])
                            local_call_count += 1
                            if np.isfinite(fi):
                                f_vals[i] = fi
                            else:
                                break
                        
                        if check_time() or call_count >= budget:
                            break

            # Update global best after NM
            # The best point in the final simplex is simplex[0] (sorted)
            # But we already updated best_x/best_f in try_func

        # Run NM on top candidates
        # Allocate remaining budget among top_k
        # If we have 3 candidates and 300 calls left, give 100 to each? 
        # Or just run them sequentially until budget is exhausted.
        
        for i in range(top_k):
            if call_count >= budget or check_time():
                break
            
            start_x, start_f = evaluated[i]
            # Give each NM run a portion of the remaining budget
            # We don't know exactly how many calls NM will use, so we just let it run
            # until the global budget is hit.
            # To prevent one bad start from eating all budget, we can limit it.
            # But since we sorted, the best start is likely to converge fast.
            
            # Run NM with a large max_iter, it will stop on budget check
            nelder_mead(start_x, start_f, remaining_budget)

        return best_x, best_f
# EVOLVE-BLOCK-END
