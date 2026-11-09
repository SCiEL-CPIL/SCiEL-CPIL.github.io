"""Combined program: Multi-start (1+1)-ES with 1/5 success rule and dedicated refinement phase."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with robust adaptation and refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # 300s limit, 20s safety margin
        dim = self.dim
        budget = self.budget

        # Wrap func to track calls and time
        call_count = 0
        best_x = None
        best_f = np.inf

        def wrapped_func(x):
            nonlocal call_count, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        def run_11es(x0, sigma0, max_evals, window_size=15):
            """Run (1+1)-ES with 1/5 success rule adaptation."""
            nonlocal call_count, best_f, best_x, deadline, budget
            
            x = x0.copy()
            fx = wrapped_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            sigma = sigma0
            successes = []
            
            for step in range(max_evals):
                if time.time() >= deadline or call_count >= budget:
                    break
                
                # Generate candidate
                step_dir = rng.standard_normal(dim)
                candidate = np.clip(x + sigma * step_dir, -5, 5)
                f_candidate = wrapped_func(candidate)
                
                if f_candidate < fx:
                    # Success
                    x = candidate
                    fx = f_candidate
                    successes.append(1)
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    successes.append(0)
                
                # Maintain window
                if len(successes) > window_size:
                    successes.pop(0)
                
                # Adapt sigma based on 1/5 success rule
                if len(successes) == window_size:
                    success_rate = sum(successes) / window_size
                    if success_rate > 0.2:
                        sigma *= 1.1
                    elif success_rate < 0.2:
                        sigma *= 0.9
                    # else keep sigma
                    
                # Stop if sigma too small
                if sigma < 1e-8:
                    break

        def run_nelder_mead(x0, max_evals, tol=1e-10):
            """Run Nelder-Mead on the best point for final polish."""
            nonlocal call_count, best_f, best_x, deadline, budget
            
            if call_count >= budget or time.time() >= deadline:
                return
            
            # Initialize simplex
            simplex = [x0.copy()]
            fvals = [wrapped_func(x0)]
            
            # Create initial simplex with small perturbation
            for i in range(dim):
                xi = x0.copy()
                step = 0.01 if abs(x0[i]) < 0.5 else 0.1
                xi[i] += step
                xi = np.clip(xi, -5, 5)
                simplex.append(xi)
                fvals.append(wrapped_func(xi))
            
            if call_count >= budget:
                return
            
            # Track best in simplex
            best_idx = int(np.argmin(fvals))
            if fvals[best_idx] < best_f:
                best_f = fvals[best_idx]
                best_x = simplex[best_idx].copy()
            
            for iteration in range(max_evals):
                if time.time() >= deadline or call_count >= budget:
                    break
                
                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                # Check convergence
                f_range = fvals[-1] - fvals[0]
                if f_range < tol:
                    break
                
                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                alpha = 1.0
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = wrapped_func(x_ref)
                
                if fvals[0] <= f_ref < fvals[-2]:
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
                    if f_ref < best_f:
                        best_f = f_ref
                        best_x = x_ref.copy()
                    continue
                
                # Expansion
                if f_ref < fvals[0]:
                    gamma = 2.0
                    x_exp = centroid + gamma * (centroid - simplex[-1])
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = wrapped_func(x_exp)
                    
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        fvals[-1] = f_exp
                        if f_exp < best_f:
                            best_f = f_exp
                            best_x = x_exp.copy()
                    else:
                        simplex[-1] = x_ref
                        fvals[-1] = f_ref
                        if f_ref < best_f:
                            best_f = f_ref
                            best_x = x_ref.copy()
                    continue
                
                # Contraction
                if f_ref < fvals[-1]:
                    rho = 0.5
                    x_con = centroid + rho * (simplex[-1] - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = wrapped_func(x_con)
                    
                    if f_con < f_ref:
                        simplex[-1] = x_con
                        fvals[-1] = f_con
                        if f_con < best_f:
                            best_f = f_con
                            best_x = x_con.copy()
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = wrapped_func(simplex[i])
                            if fvals[i] < best_f:
                                best_f = fvals[i]
                                best_x = simplex[i].copy()
                        if call_count >= budget:
                            break
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = wrapped_func(simplex[i])
                        if fvals[i] < best_f:
                            best_f = fvals[i]
                            best_x = simplex[i].copy()
                    if call_count >= budget:
                        break

        # Phase 1: Exploration with multiple starts
        # Allocate 50% of budget to exploration
        alloc_explore = int(budget * 0.5)
        n_starts = 5
        starts = [np.zeros(dim)]
        for i in range(n_starts - 1):
            starts.append(rng.uniform(-4, 4, dim))
        
        # Distribute budget among starts
        steps_per_start = alloc_explore // n_starts
        
        for i, x0 in enumerate(starts):
            if time.time() >= deadline or call_count >= budget:
                break
            
            # Use remaining budget for this start, capped by steps_per_start
            remaining_for_start = min(steps_per_start, budget - call_count)
            if remaining_for_start < 10:
                break
                
            run_11es(x0, 1.5, remaining_for_start)

        # Phase 2: Exploitation on the best point found
        # Allocate remaining budget to refinement
        if best_x is not None and call_count < budget:
            remaining = budget - call_count
            if remaining > 20:
                # Start refinement with smaller sigma
                run_11es(best_x, 0.2, remaining)

        # Phase 3: Final polish with Nelder-Mead
        if best_x is not None and call_count < budget:
            remaining = budget - call_count
            if remaining > 10:
                run_nelder_mead(best_x, remaining)

        return best_x, best_f
# EVOLVE-BLOCK-END
