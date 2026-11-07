"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead ensemble for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Number of parallel NM runs
        n_runs = 5
        
        # Each NM run needs dim+1 initial evaluations
        init_cost = n_runs * (dim + 1)
        
        if budget < init_cost:
            # Fallback to simple random search if budget is too small
            best_x = None
            best_f = float('inf')
            for _ in range(budget):
                x = rng.uniform(-5, 5, dim)
                fx = func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x
            return best_x, best_f

        # Initialize simplexes using Latin Hypercube Sampling
        # LHS ensures stratified coverage
        simplex_centers = np.zeros((n_runs, dim))
        for i in range(dim):
            # Divide range [-5, 5] into n_runs intervals
            intervals = np.linspace(-5, 5, n_runs + 1)
            # Pick one point from each interval
            for j in range(n_runs):
                simplex_centers[j, i] = rng.uniform(intervals[j], intervals[j+1])
        
        # Shuffle the centers to ensure they are not correlated in a bad way
        rng.shuffle(simplex_centers)

        # Data structure for each run
        # Each run has a simplex (list of dim+1 points) and their values
        runs = []
        calls_used = 0
        global_best_x = None
        global_best_f = float('inf')

        # Initialize each run
        for k in range(n_runs):
            center = simplex_centers[k]
            simplex = np.zeros((dim + 1, dim))
            fvals = np.zeros(dim + 1)
            
            # First point is the center
            simplex[0] = center
            fvals[0] = func(simplex[0])
            calls_used += 1
            if fvals[0] < global_best_f:
                global_best_f = fvals[0]
                global_best_x = simplex[0].copy()

            # Other points are center +/- delta * e_i
            delta = 0.5 # Initial step size
            for i in range(dim):
                simplex[i+1] = center.copy()
                simplex[i+1][i] += delta
                # Clip to box
                simplex[i+1] = np.clip(simplex[i+1], -5, 5)
                fvals[i+1] = func(simplex[i+1])
                calls_used += 1
                if fvals[i+1] < global_best_f:
                    global_best_f = fvals[i+1]
                    global_best_x = simplex[i+1].copy()

            runs.append({
                'simplex': simplex,
                'fvals': fvals,
                'active': True
            })

        # NM Parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # Main loop
        while calls_used < budget:
            if time.time() > deadline:
                break

            # Select the run with the best current minimum
            # If all are converged or stalled, pick the one with the smallest simplex size?
            # Or just round-robin? Best-first is good for AOCC.
            best_run_idx = -1
            best_run_val = float('inf')
            for i, run in enumerate(runs):
                if not run['active']:
                    continue
                min_val = np.min(run['fvals'])
                if min_val < best_run_val:
                    best_run_val = min_val
                    best_run_idx = i
            
            if best_run_idx == -1:
                # All runs inactive or converged
                break

            run = runs[best_run_idx]
            simplex = run['simplex']
            fvals = run['fvals']
            
            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]
            
            # Check convergence: if the difference between best and worst is small
            if fvals[-1] - fvals[0] < 1e-8:
                run['active'] = False
                continue

            centroid = np.mean(simplex[1:], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[0])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            calls_used += 1
            if fr < global_best_f:
                global_best_f = fr
                global_best_x = xr.copy()

            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[0])
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                calls_used += 1
                if fe < global_best_f:
                    global_best_f = fe
                    global_best_x = xe.copy()
                if fe < fr:
                    simplex[0] = xe
                    fvals[0] = fe
                else:
                    simplex[0] = xr
                    fvals[0] = fr
            elif fr < fvals[-2]:
                # Accept reflection
                simplex[0] = xr
                fvals[0] = fr
            else:
                # Contraction
                if fr < fvals[0]:
                    # Outer contraction
                    xc = centroid + rho * (xr - centroid)
                else:
                    # Inner contraction
                    xc = centroid - rho * (centroid - simplex[0])
                
                xc = np.clip(xc, -5, 5)
                fc = func(xc)
                calls_used += 1
                if fc < global_best_f:
                    global_best_f = fc
                    global_best_x = xc.copy()

                if fc < fr:
                    simplex[0] = xc
                    fvals[0] = fc
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        calls_used += 1
                        if fvals[i] < global_best_f:
                            global_best_f = fvals[i]
                            global_best_x = simplex[i].copy()
                    
                    # If we ran out of budget during shrink, break
                    if calls_used >= budget:
                        break

            run['simplex'] = simplex
            run['fvals'] = fvals

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
