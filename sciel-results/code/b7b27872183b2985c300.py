"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

class Optimizer:
    """Nelder-Mead optimizer with restarts on the box [-5, 5]^dim."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls = 0
        
        # Helper to project to box
        def project(x):
            return np.clip(x, -5.0, 5.0)
        
        # Helper to evaluate and track calls
        def evaluate(x):
            nonlocal calls
            if calls >= budget:
                return np.inf
            x_proj = project(x)
            f_val = func(x_proj)
            calls += 1
            return f_val

        # Initial global exploration: Random points to find a good starting basin
        # Discus f11 is multimodal-ish, so a few random starts help.
        # Budget allocation: 50% for global search (random/restarts), 50% for local refinement?
        # Actually, NM is fast. Let's do a few random restarts of NM.
        
        # Strategy: 
        # 1. Evaluate 5 random points to find a good initial center.
        # 2. Run Nelder-Mead from the best point.
        # 3. If stuck, restart from best point with larger step or new random point.
        
        # Phase 1: Global Scattering
        num_global = 10
        global_points = np.random.uniform(-5, 5, (num_global, n))
        global_fvals = np.array([evaluate(x) for x in global_points])
        
        # If we ran out of budget during global phase (unlikely with 10 calls)
        if calls >= budget:
            min_idx = np.argmin(global_fvals)
            return global_points[min_idx], global_fvals[min_idx]

        best_idx = np.argmin(global_fvals)
        best_x = global_points[best_idx].copy()
        best_f = global_fvals[best_idx]
        
        # Phase 2: Nelder-Mead Optimization with Restarts
        # We have budget - calls remaining.
        # NM iterations consume n+1 points per iteration roughly (reflection, etc).
        # Let's run NM in loops, restarting if no improvement for 2*n iterations.
        
        max_nm_iters = 50 # per run
        max_restarts = 10
        
        for restart in range(max_restarts):
            if calls >= budget:
                break
            
            # Initialize simplex
            # Center is current best_x
            center = best_x.copy()
            
            # Initial step size
            # Start with a moderate step, maybe scale down if previous runs failed
            step = 1.0 if restart == 0 else 0.5 * (1.0 / (restart + 1))
            
            # Build simplex
            simplex = np.zeros((n + 1, n))
            simplex[0] = center
            for i in range(1, n + 1):
                simplex[i] = center.copy()
                simplex[i][i - 1] += step
            
            # Evaluate simplex vertices
            fvals = np.zeros(n + 1)
            for i in range(n + 1):
                fvals[i] = evaluate(simplex[i])
                if calls >= budget:
                    break
            
            if calls >= budget:
                break
            
            # Track best in this NM run
            run_best_f = np.min(fvals)
            run_best_idx = np.argmin(fvals)
            
            # Nelder-Mead Loop
            stall_count = 0
            for it in range(max_nm_iters):
                if calls >= budget:
                    break
                
                # Sort vertices by f-value
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]
                
                # Check for improvement
                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = simplex[0].copy()
                    stall_count = 0
                else:
                    stall_count += 1
                    # If stalled for too long, break to restart
                    if stall_count > 2 * n:
                        break
                
                # Reflection
                x_w = simplex[-1]
                f_w = fvals[-1]
                centroid = np.mean(simplex[:-1], axis=0)
                
                alpha = 1.0
                x_r = centroid + alpha * (centroid - x_w)
                f_r = evaluate(x_r)
                
                if calls >= budget:
                    break
                
                if f_r < fvals[0]:
                    # Expansion
                    gamma = 2.0
                    x_e = centroid + gamma * (centroid - x_w)
                    f_e = evaluate(x_e)
                    if calls >= budget:
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    # Accept Reflection
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    rho = 0.5
                    if f_r < f_w:
                        # Outside Contraction
                        x_c = centroid + rho * (x_r - centroid)
                        f_c = evaluate(x_c)
                        if calls >= budget:
                            break
                        if f_c <= f_r:
                            simplex[-1] = x_c
                            fvals[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                fvals[i] = evaluate(simplex[i])
                            if calls >= budget:
                                break
                    else:
                        # Inside Contraction
                        x_c = centroid - rho * (centroid - x_w)
                        f_c = evaluate(x_c)
                        if calls >= budget:
                            break
                        if f_c < f_w:
                            simplex[-1] = x_c
                            fvals[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                fvals[i] = evaluate(simplex[i])
                            if calls >= budget:
                                break
            
            # End of NM run
            # If we didn't find a better solution than the global best, 
            # we might want to try a new random center next restart.
            # But for now, we just restart from the current best_x with smaller step.
            # Alternatively, if we haven't improved at all in this run, 
            # pick a new random point for the next restart center.
            
            if restart < max_restarts - 1 and calls < budget:
                # If this run didn't improve the global best, 
                # let's try a new random center for the next restart
                # to escape local minima.
                if run_best_f >= best_f:
                    # Pick a new random center
                    new_center = np.random.uniform(-5, 5, n)
                    f_new = evaluate(new_center)
                    if calls < budget:
                        if f_new < best_f:
                            best_f = f_new
                            best_x = new_center
                        # Next restart will use this best_x as center
                # Else, continue with current best_x but smaller step (handled by step calc above)

        return best_x, best_f
# EVOLVE-BLOCK-END
