"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-Start Nelder-Mead optimizer for BBOB f17."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin for 300s limit
        
        dim = self.dim
        total_budget = self.budget
        
        best_x = None
        best_f = np.inf
        
        # We will use a multi-start Nelder-Mead approach.
        # Nelder-Mead is effective for smooth, multimodal functions if started in promising regions.
        # Schaffers F7 has a known structure but is rotated/shifted.
        
        # Phase 1: Initial Exploration (first ~50-100 calls)
        # We want to find a good starting point quickly.
        # Instead of random uniform, we can use a few "probing" points.
        # However, standard random is robust. Let's use a small number of random starts 
        # and evaluate them to pick the best one for the first NM run.
        
        # To maximize AOCC (Area Over Curve of Convergence), early progress is critical.
        # We should make the first few calls very diverse to quickly identify a low region.
        
        # Let's allocate the budget:
        # 1. Initial random sampling: 10 points.
        # 2. Run Nelder-Mead from the best of those 10.
        # 3. If NM converges (stagnates) or budget remains, restart NM from other top candidates 
        #    or new random points.
        
        rng = np.random.default_rng(42)
        
        def nelder_mead(func, x0, budget_limit, deadline, current_best_f, current_best_x, rng):
            """
            Perform Nelder-Mead optimization starting at x0.
            Returns (best_f, best_x, calls_made)
            """
            dim = len(x0)
            
            # Initialize simplex
            # Vertices: x0 and x0 +/- step * e_i
            step = 0.5  # Initial step size
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = x0
            for i in range(dim):
                simplex[i+1] = x0.copy()
                simplex[i+1][i] += step
                
            # Ensure all vertices are within bounds [-5, 5]
            # If x0 is near boundary, we might need to adjust, but x0 is from [-5,5]
            # If a vertex goes out of bounds, clip it.
            # Clipping might collapse the simplex if x0 is at boundary.
            # Better: if x0 is at boundary, reflect the step inward.
            # For simplicity, clip. If simplex degenerates, NM will fail/stagnate.
            simplex = np.clip(simplex, -5, 5)
            
            # Evaluate initial simplex
            fs = np.empty(dim + 1)
            calls = 0
            
            for i in range(dim + 1):
                if calls >= budget_limit or time.time() > deadline:
                    return current_best_f, current_best_x, calls
                fx = func(simplex[i])
                fs[i] = fx
                calls += 1
                if fx < current_best_f:
                    current_best_f = fx
                    current_best_x = simplex[i].copy()
            
            # Nelder-Mead parameters
            alpha = 1.0   # Reflection
            gamma = 2.0   # Expansion
            rho = 0.5     # Contraction
            sigma = 0.5   # Shrink
            
            max_iter = 200  # Safety cap on iterations per NM run
            
            for _ in range(max_iter):
                if calls >= budget_limit or time.time() > deadline:
                    break
                
                # Sort simplex by function value
                idx = np.argsort(fs)
                simplex = simplex[idx]
                fs = fs[idx]
                
                # Check convergence: if best and worst are close
                if fs[-1] - fs[0] < 1e-6:
                    break
                
                # Centroid of all but the worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                calls += 1
                if fr < current_best_f:
                    current_best_f = fr
                    current_best_x = xr.copy()
                
                if fr < fs[0]:
                    # Try expansion
                    xe = centroid + gamma * (xr - centroid)
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    calls += 1
                    if fe < current_best_f:
                        current_best_f = fe
                        current_best_x = xe.copy()
                    
                    if fe < fr:
                        simplex[-1] = xe
                        fs[-1] = fe
                    else:
                        simplex[-1] = xr
                        fs[-1] = fr
                elif fr < fs[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    fs[-1] = fr
                else:
                    # Contraction
                    # Outside contraction if fr < fs[-1] (original worst), else inside
                    if fr < fs[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                    
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    calls += 1
                    if fc < current_best_f:
                        current_best_f = fc
                        current_best_x = xc.copy()
                    
                    if fc < fr:
                        simplex[-1] = xc
                        fs[-1] = fc
                    else:
                        # Shrink
                        # Move all points (except best) towards best
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fi = func(simplex[i])
                            fs[i] = fi
                            calls += 1
                            if fi < current_best_f:
                                current_best_f = fi
                                current_best_x = simplex[i].copy()
                        
                        if calls >= budget_limit or time.time() > deadline:
                            break
                
                # Optional: Check for stagnation in function values
                # If no improvement in best_f for many iterations, break to restart
                # We rely on the outer loop to restart if this returns without much progress
                
            return current_best_f, current_best_x, calls

        # Phase 1: Initial Sampling
        # Let's sample 10 random points to find a good starting location.
        # This uses 10 calls.
        # We want to ensure diversity.
        
        init_n = 10
        init_points = rng.uniform(-5, 5, size=(init_n, dim))
        init_fs = np.empty(init_n)
        calls = 0
        
        for i in range(init_n):
            if calls >= total_budget or time.time() > deadline:
                break
            fx = func(init_points[i])
            init_fs[i] = fx
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = init_points[i].copy()
        
        # Sort initial points by fitness
        init_indices = np.argsort(init_fs)
        
        # Phase 2: Multi-Start Nelder-Mead
        # We will attempt NM from the best initial point.
        # If it stagnates or budget remains, we try the next best, etc.
        # We can also generate new random points if we get stuck.
        
        # To be efficient, we'll define a "budget per start" or just let it run until
        # no progress is made for a while, or budget is low.
        
        # Strategy: 
        # 1. Run NM from best initial point.
        # 2. If NM returns and we still have budget, check if we improved significantly.
        # 3. If not, try next best initial point.
        # 4. If all initial points exhausted, generate new random points and try NM.
        
        # To avoid getting stuck in a local minimum of the first start, 
        # we limit the "effort" per start if no progress is made.
        
        # Let's define a helper to run NM with a specific budget limit.
        
        for start_idx in init_indices:
            if calls >= total_budget or time.time() > deadline:
                break
            
            # Determine how much budget to give this NM run.
            # If this is the first start, give it a large chunk (e.g., 400).
            # If subsequent, give it less, or just the remainder.
            # Actually, just let it run until budget is low or it converges.
            
            # To prevent one bad start from eating all budget, 
            # we can monitor progress. But NM is fast per iteration.
            # Let's just run it. If it fails to improve best_f, we move to next start.
            
            # Save current best
            prev_best_f = best_f
            
            # Run NM
            # We pass the current best_f and best_x so it can update them
            # But NM uses its own simplex. We need to pass the starting point.
            x0 = init_points[start_idx]
            
            # We need to be careful: NM will evaluate x0 again? 
            # No, we already evaluated it in init. 
            # But the NM implementation above evaluates the whole simplex including x0.
            # This wastes 1 call. 
            # Optimization: We could modify NM to accept pre-evaluated first point.
            # For simplicity and robustness against bugs, let's just let it re-evaluate 
            # or adjust the budget. 1 call is negligible compared to 500.
            
            # However, to save calls, we can modify the NM loop to skip the first eval 
            # if we know it. But the simplex includes neighbors too.
            # Let's stick to the simple implementation: it re-evaluates x0.
            
            new_best_f, new_best_x, calls_made = nelder_mead(
                func, x0, total_budget, deadline, best_f, best_x, rng
            )
            
            calls += calls_made
            best_f = new_best_f
            best_x = new_best_x
            
            # Check if we made progress
            if best_f < prev_best_f - 1e-6:
                # Good progress, continue with this start or finish?
                # NM usually converges. If it stopped due to budget, we are done.
                # If it stopped due to convergence, we should try another start 
                # to see if we can find a global minimum.
                # How to distinguish? 
                # If calls < total_budget, we have budget left.
                # If NM converged (stagnated), best_f won't improve much with more iterations.
                # So, if we have budget left, try the next best initial point.
                pass
            else:
                # No progress from this start. Try next.
                pass

            # If we have used up most of the budget, stop.
            if calls > total_budget * 0.9:
                break

        # If we still have budget, do a final random search or restart?
        # Given the 500 budget, and NM being efficient, we might use it all.
        # If not, a few more random points + NM might help.
        
        # Simple safeguard: if budget remains, just do a few more random evaluations
        # to see if we find something better, though unlikely after NM.
        
        while calls < total_budget and time.time() < deadline:
            x_new = rng.uniform(-5, 5, size=dim)
            f_new = func(x_new)
            calls += 1
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()
            else:
                # If random search fails to improve for a bit, we are likely done.
                # But breaking immediately might be premature.
                # Let's just run out of budget.
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
