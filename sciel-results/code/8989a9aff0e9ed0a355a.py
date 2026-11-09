"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Hybrid Nelder-Mead + Coordinate Line Search for BBOB f12 (Bent Cigar)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        N = self.budget
        
        # Deadline guard
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        rng = np.random.default_rng(42)

        best_x = np.zeros(d)
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made, best_f, best_x
            # Ensure x is a valid numpy array
            x = np.asarray(x, dtype=float)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # 1. Evaluate center to establish a baseline
        x0 = np.zeros(d)
        evaluate(x0)
        
        # 2. Multi-start Nelder-Mead
        # We use 5 restarts. 
        # Budget allocation: 80% for NM, 20% for Coordinate Search
        nm_budget_per_restart = int((N * 0.8) / 5)
        
        # Initial simplex for Nelder-Mead
        def nelder_mead(func, x_start, budget, rng):
            d = len(x_start)
            # Initial simplex
            simplex = []
            f_values = []
            
            # Start point
            x0 = x_start.copy()
            f0 = evaluate(x0)
            simplex.append(x0)
            f_values.append(f0)
            
            # Generate other vertices
            for i in range(d):
                xi = x0.copy()
                step = 0.25 if x0[i] == 0 else 0.05 * x0[i]
                xi[i] += step
                # Clip to bounds
                xi = np.clip(xi, -5.0, 5.0)
                fi = evaluate(xi)
                simplex.append(xi)
                f_values.append(fi)
                
            # Main Loop
            alpha = 1.0
            gamma = 2.0
            rho = -0.5
            sigma = 0.5
            
            while budget > 0:
                if time.time() > deadline:
                    break
                    
                # Sort simplex by fitness
                order = np.argsort(f_values)
                simplex = [simplex[i] for i in order]
                f_values = [f_values[i] for i in order]
                
                # Check convergence
                if len(set([round(f, 12) for f in f_values])) < 2:
                    break
                
                x_worst = simplex[-1]
                f_worst = f_values[-1]
                x_best = simplex[0]
                
                # Centroid of all but worst
                x_c = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = x_c + alpha * (x_c - x_worst)
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = evaluate(x_r)
                budget -= 1
                
                if f_r < f_values[0]:
                    # Expansion
                    x_e = x_c + gamma * (x_r - x_c)
                    x_e = np.clip(x_e, -5.0, 5.0)
                    f_e = evaluate(x_e)
                    budget -= 1
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_values[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_values[-1] = f_r
                elif f_r < f_values[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    f_values[-1] = f_r
                else:
                    # Contraction
                    # Outside contraction
                    x_co = x_c + sigma * (x_r - x_c)
                    x_co = np.clip(x_co, -5.0, 5.0)
                    f_co = evaluate(x_co)
                    budget -= 1
                    if f_co < f_r:
                        simplex[-1] = x_co
                        f_values[-1] = f_co
                    else:
                        # Inside contraction
                        x_ci = x_c - sigma * (x_r - x_c)
                        x_ci = np.clip(x_ci, -5.0, 5.0)
                        f_ci = evaluate(x_ci)
                        budget -= 1
                        if f_ci < f_worst:
                            simplex[-1] = x_ci
                            f_values[-1] = f_ci
                        else:
                            # Shrink
                            for i in range(1, d + 1):
                                simplex[i] = x_best + 0.5 * (simplex[i] - x_best)
                                simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                                f_values[i] = evaluate(simplex[i])
                                budget -= 1
                                if budget <= 0:
                                    return simplex[0], f_values[0]
                            
            # Return best in simplex
            idx = np.argmin(f_values)
            return simplex[idx], f_values[idx]

        # Perform 5 restarts
        # First restart at 0
        # Subsequent restarts at random locations within [-2, 2] to cover the space
        restart_points = [np.zeros(d)]
        for _ in range(4):
            restart_points.append(rng.uniform(-2.0, 2.0, d))
            
        for rp in restart_points:
            if calls_made >= N:
                break
            if time.time() > deadline:
                break
            
            # Run Nelder-Mead with allocated budget
            # We pass the remaining budget capped by nm_budget_per_restart
            # But we must not exceed N
            avail_budget = min(nm_budget_per_restart, N - calls_made)
            if avail_budget <= 0:
                break
            
            # NM uses evaluate() which increments calls_made
            # We need to track local budget for NM loop logic
            # The NM function above takes a 'budget' param for its internal loop, 
            # but it relies on the global 'calls_made' via 'evaluate'.
            # To be safe, we just let it run until 'budget' internal counter hits 0 
            # or global deadline/budget is hit.
            
            # Note: The internal 'budget' in NM is just a counter for iterations, 
            # but each iteration consumes 1-3 global evals.
            # We should ensure the internal budget doesn't allow too many global calls.
            # Let's just run it. The global 'calls_made' check in 'evaluate' 
            # is not there, but the 'deadline' is.
            # We need to be careful not to exceed N.
            
            # Let's modify NM to respect a hard stop on global calls.
            # Actually, the current NM implementation above doesn't check global N.
            # Let's add a check in the main loop or pass a stop condition.
            # For simplicity and safety, we will just run it. 
            # If it exceeds N, the next restart will be skipped.
            # However, exceeding N is invalid.
            
            # Let's wrap the NM call to ensure we don't exceed N.
            # We'll create a wrapper func that raises or stops if calls_made >= N.
            
            def safe_func(x):
                if calls_made >= N:
                    return np.inf # Signal to stop? NM might not handle inf well.
                return func(x)
            
            # Better: Just check before calling NM and limit the internal budget.
            # The internal budget in NM is the number of *iterations*, not evaluations.
            # 1 iteration = 1 to 3 evals.
            # So if we give internal budget = avail_budget // 2, it's safe.
            internal_budget = max(1, avail_budget // 2)
            
            # We need to pass the 'evaluate' function that updates global state.
            # The NM function defined above uses 'evaluate' from the closure.
            # This is correct.
            
            # Run NM
            # We don't strictly need the return value of NM as we track global best
            nelder_mead(func, rp, internal_budget, rng)
            
        # 3. Coordinate-wise Line Search for the remaining budget
        # This phase exploits the anisotropy.
        # We iterate through dimensions, doing a 1D minimization for each.
        
        remaining_budget = N - calls_made
        if remaining_budget > 0 and time.time() < deadline:
            x_curr = best_x.copy()
            f_curr = best_f
            
            # Simple 1D minimization using Golden Section Search
            # We allocate a few evaluations per dimension per cycle.
            # Total evals = d * evals_per_dim * cycles
            # Let's use 5 evals per dimension per cycle.
            evals_per_dim = 5
            cycles = remaining_budget // (d * evals_per_dim)
            
            # If cycles is 0, we still have some budget, use it for a few steps
            if cycles == 0:
                # Just do a few random coordinate steps or a simple line search
                # For simplicity, if budget is low, do a small step line search on the best axis
                # Or just stop.
                pass
            
            for _ in range(cycles):
                if time.time() > deadline:
                    break
                for i in range(d):
                    if calls_made >= N:
                        break
                    if time.time() > deadline:
                        break
                    
                    # Golden Section Search along dimension i
                    # Determine a reasonable interval
                    # Use current step or a fixed width
                    width = 0.5
                    a = x_curr[i] - width
                    b = x_curr[i] + width
                    
                    # Clip interval to bounds
                    a = max(a, -5.0)
                    b = min(b, 5.0)
                    
                    if b - a < 1e-6:
                        continue
                        
                    # Golden ratio
                    gr = (np.sqrt(5) - 1) / 2
                    c = b - gr * (b - a)
                    dpt = a + gr * (b - a)
                    
                    # Evaluate f at c and d
                    x_c = x_curr.copy()
                    x_c[i] = c
                    f_c = evaluate(x_c)
                    
                    x_d = x_curr.copy()
                    x_d[i] = dpt
                    f_d = evaluate(x_d)
                    
                    # We have used 2 evals. We have evals_per_dim - 2 left for this dim.
                    # Let's do 3 more iterations of golden section if budget allows.
                    for _ in range(3):
                        if calls_made >= N:
                            break
                        if time.time() > deadline:
                            break
                        if f_c < f_d:
                            b = dpt
                            dpt = c
                            f_d = f_c
                            c = b - gr * (b - a)
                            x_c = x_curr.copy()
                            x_c[i] = c
                            f_c = evaluate(x_c)
                        else:
                            a = c
                            c = dpt
                            f_c = f_d
                            dpt = a + gr * (b - a)
                            x_d = x_curr.copy()
                            x_d[i] = dpt
                            f_d = evaluate(x_d)
                    
                    # Update x_curr with the best found in this dimension
                    if f_c < f_d:
                        x_curr[i] = c
                        if f_c < f_curr:
                            f_curr = f_c
                    else:
                        x_curr[i] = dpt
                        if f_d < f_curr:
                            f_curr = f_d

        return best_x, best_f

# EVOLVE-BLOCK-END
