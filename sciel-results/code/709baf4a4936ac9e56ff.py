"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Adaptive Nelder-Mead optimizer with restarts on stagnation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 325.0  # Safety margin before 330s kill
        
        dim = self.dim
        budget = self.budget
        
        # Global best tracking
        best_x_global = None
        best_f_global = np.inf
        
        # Nelder-Mead parameters
        # Initial simplex size
        initial_sigma = 2.0
        # Convergence threshold
        tol = 1e-10
        # Max iterations per local run to avoid infinite loops on flat regions
        max_local_iters = 100
        
        # We will use a "multi-start" strategy:
        # 1. Start with a few diverse initial points (random sampling)
        # 2. Run Nelder-Mead from the best of these
        # 3. If Nelder-Mead stagnates (no improvement for a while), restart from a new random point
        #    or from a perturbation of the current best, until budget is exhausted.
        
        # To maximize AOCC (Area Over Convergence Curve), we want to make progress
        # early. Nelder-Mead is fast, so we can afford multiple restarts.
        
        # Phase 1: Initial diverse sampling to find a good basin
        # Use ~10% of budget for initial exploration if budget allows
        n_init_samples = max(5, min(20, budget // 10))
        
        # We'll keep a list of "candidates" for local optimization
        candidates = []
        
        for i in range(n_init_samples):
            if time.time() - start_time > time_limit:
                break
            if budget <= 0:
                break
            x = np.random.uniform(-4, 4, dim)  # Stay within [-5,5] with margin
            f = func(x)
            budget -= 1
            candidates.append((f, x))
            if f < best_f_global:
                best_f_global = f
                best_x_global = x.copy()
                
        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])
        
        # Phase 2: Local optimization with Nelder-Mead
        # We will iterate through the best candidates and run NM on them.
        # If NM fails to improve significantly, we move to the next candidate or generate new ones.
        
        # Adaptive Nelder-Mead implementation
        def run_nelder_mead(func, x0, f0, budget_remaining, max_iters, time_limit, start_time):
            """
            Runs Nelder-Mead starting from x0 with value f0.
            Returns (best_x, best_f, calls_made).
            """
            dim = len(x0)
            # Initialize simplex
            simplex = [x0.copy()]
            f_values = [f0]
            
            # Add dim more points
            for i in range(dim):
                x_i = x0.copy()
                # Perturb one dimension
                step = 0.05 if i == 0 else 0.05 # Standard NM uses 0.05 or 0.1
                # Better: use a small step relative to box size
                step = 0.1
                x_i[i] += step
                # Ensure within bounds
                x_i = np.clip(x_i, -5, 5)
                f_i = func(x_i)
                budget_remaining -= 1
                simplex.append(x_i)
                f_values.append(f_i)
                if f_i < min(f_values[:-1]):
                    pass # Just tracking
            
            # Update global best if any initial simplex point was better
            current_best_f = min(f_values)
            current_best_x = simplex[f_values.index(current_best_f)].copy()
            
            calls_made = len(f_values)
            
            # NM Loop
            no_improve_count = 0
            best_f_so_far = current_best_f
            best_x_so_far = current_best_x.copy()
            
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma = 0.5  # shrink
            
            for _ in range(max_iters):
                if time.time() - start_time > time_limit:
                    break
                if budget_remaining <= 0:
                    break
                
                # Sort simplex
                indices = np.argsort(f_values)
                simplex = [simplex[i] for i in indices]
                f_values = [f_values[i] for i in indices]
                
                # Check convergence
                if np.std(f_values) < tol:
                    break
                
                # Update best
                if f_values[0] < best_f_so_far:
                    best_f_so_far = f_values[0]
                    best_x_so_far = simplex[0].copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count > 20:
                        break # Stagnation detected
                
                # Centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)
                worst_point = simplex[-1]
                worst_f = f_values[-1]
                
                # Reflection
                x_r = centroid + alpha * (centroid - worst_point)
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                calls_made += 1
                budget_remaining -= 1
                
                if f_r < f_values[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - worst_point)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    calls_made += 1
                    budget_remaining -= 1
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_values[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_values[-1] = f_r
                elif f_r < f_values[-2]:
                    # Reflection is good
                    simplex[-1] = x_r
                    f_values[-1] = f_r
                else:
                    # Contraction
                    if f_r < worst_f:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (centroid - worst_point)
                    
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    calls_made += 1
                    budget_remaining -= 1
                    
                    if f_c < min(f_r, worst_f):
                        simplex[-1] = x_c
                        f_values[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            f_values[i] = func(simplex[i])
                            calls_made += 1
                            budget_remaining -= 1
                            if budget_remaining <= 0:
                                break
                        if budget_remaining <= 0:
                            break
                
                # Update global best in this local run
                if f_values[0] < best_f_so_far:
                    best_f_so_far = f_values[0]
                    best_x_so_far = simplex[0].copy()
                
            return best_x_so_far, best_f_so_far, calls_made

        # Main loop: iterate through candidates and potentially generate new ones
        # We prioritize the best known points but also allow new random starts if budget remains
        # and the current best is not improving fast enough (though NM is fast, so this is less likely).
        
        # To maximize AOCC, we should spend budget on the most promising local searches first.
        # We will process the initial candidates in order of their initial fitness.
        
        # Also, we can generate new candidates on the fly if we have budget left after finishing a local search.
        
        # Strategy: 
        # 1. For each initial candidate (sorted by fitness):
        #    a. Run NM.
        #    b. If budget remains, generate a new random candidate and run NM on it?
        #       Or just continue to next candidate?
        #       Given NM is very efficient (usually < 50 evals for dim 5), we can afford many restarts.
        
        # Let's allocate budget dynamically.
        # We'll keep a list of "unprocessed" starts.
        
        processed_starts = 0
        max_restarts = budget // 20  # Rough estimate: NM takes ~20-50 evals. Allow for many restarts.
        
        while budget > 0 and time.time() - start_time < time_limit:
            # Get next start point
            if processed_starts < len(candidates):
                f_start, x_start = candidates[processed_starts]
                processed_starts += 1
            else:
                # Generate new random start
                x_start = np.random.uniform(-4, 4, dim)
                f_start = func(x_start)
                budget -= 1
                if f_start < best_f_global:
                    best_f_global = f_start
                    best_x_global = x_start.copy()
                
                if budget <= 0:
                    break
            
            # Run Nelder-Mead
            # Reserve some budget for the next potential restart?
            # No, just run until budget is low or time is up.
            x_best_local, f_best_local, calls_used = run_nelder_mead(
                func, x_start, f_start, budget, max_local_iters, time_limit, start_time
            )
            
            budget -= calls_used
            
            if f_best_local < best_f_global:
                best_f_global = f_best_local
                best_x_global = x_best_local.copy()
            
            # If we have significantly more budget left, we might want to do another restart
            # even if we just finished one. The loop above handles this by generating a new start.
            
            # Optimization: If the local search converged very quickly (few calls),
            # it might have hit a local minimum. Restarting is good.
            # If it used most of the budget, we are close to the limit.

        return best_x_global, best_f_global
# EVOLVE-BLOCK-END
