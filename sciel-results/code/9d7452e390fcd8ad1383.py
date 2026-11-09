"""Multi-Start Nelder-Mead + (1+1)-ES hybrid optimizer for BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-Start Nelder-Mead + (1+1)-ES hybrid for BBOB f11.
    
    1. Diverse exploration to find multiple promising basins.
    2. Run Nelder-Mead from the top candidates to handle the rotated/ill-conditioned landscape.
    3. Polish with (1+1)-ES.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 310  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        def time_ok():
            return time.time() < deadline

        evals = 0
        best_x = None
        best_f = np.inf

        # --- Phase 1: Diverse Exploration ---
        # Goal: Find several distinct local minima or good starting points.
        # f11 is shifted and rotated. Random points help find the basin.
        # Axis probes help detect the shift direction and magnitude.
        
        points = []
        values = []

        # 1. Structured Probes (Axis-aligned)
        # Probe along each axis with +/- signs and different magnitudes
        # This helps if the shift is aligned or if we need to break symmetry.
        mags = [1.0, 2.0, 3.0, 4.0]
        for i in range(dim):
            for m in mags:
                for s in [1.0, -1.0]:
                    if evals >= budget - 100: # Reserve budget for optimization
                        break
                    x = np.zeros(dim)
                    x[i] = s * m
                    fx = func(x)
                    points.append(x)
                    values.append(fx)
                    evals += 1
                if evals >= budget - 100:
                    break
            if evals >= budget - 100:
                break
        
        # 2. Random Sampling
        # Fill the rest of the exploration budget with random points
        explore_budget = min(150, budget - 200) # Ensure we leave room for NM and ES
        while evals < explore_budget:
            x = rng.uniform(-4.5, 4.5, dim)
            fx = func(x)
            points.append(x)
            values.append(fx)
            evals += 1

        if not points:
            # Fallback if budget is too small for exploration
            x = np.zeros(dim)
            fx = func(x)
            points.append(x)
            values.append(fx)
            evals += 1

        # Select top K candidates
        # We want distinct candidates. Simple k-best is okay, but let's try to ensure some diversity.
        # For now, just take the K best values.
        K = 3
        indices = np.argsort(values)
        top_indices = indices[:K]
        
        candidates = []
        for idx in top_indices:
            candidates.append((points[idx].copy(), values[idx]))
            
        # Update global best
        best_idx = int(np.argmin(values))
        best_x = points[best_idx].copy()
        best_f = values[best_idx]

        if evals >= budget or not time_ok():
            return best_x, best_f

        # --- Phase 2: Multi-Start Nelder-Mead ---
        # Allocate remaining budget to NM runs.
        # We have K candidates. Let's give each a share of the remaining budget, 
        # but stop early if one converges well or stagnates.
        
        polish_budget = 50
        nm_total_budget = budget - evals - polish_budget
        nm_per_start = max(10, nm_total_budget // K)

        nm_params = {
            'alpha': 1.0,
            'gamma': 2.0,
            'rho': 0.5,
            'sigma': 0.5
        }

        for start_idx, (start_x, start_f) in enumerate(candidates):
            if evals >= budget - polish_budget or not time_ok():
                break
            
            # Skip if this start is already worse than current best by a lot? 
            # No, let's run it, it might be in a different basin.
            
            x0 = start_x.copy()
            f0 = start_f
            
            # Initialize Simplex
            # Use a step size relative to the box size
            step = 0.5
            simplex = [x0.copy()]
            fvals = [f0]
            
            for i in range(dim):
                xi = x0.copy()
                xi[i] += step
                simplex.append(xi.copy())
                fi = func(xi)
                evals += 1
                fvals.append(fi)
                
                if fi < best_f:
                    best_f = fi
                    best_x = xi.copy()

            # Run Nelder-Mead
            nm_iter = 0
            max_nm_iter = nm_per_start
            stagnation_count = 0
            prev_best_f = f0
            
            while nm_iter < max_nm_iter and evals < budget - polish_budget and time_ok():
                nm_iter += 1
                
                # Sort simplex
                idx = np.argsort(fvals)
                simplex = [simplex[i].copy() for i in idx]
                fvals = [fvals[i] for i in idx]
                
                # Check for improvement
                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = simplex[0].copy()
                    stagnation_count = 0
                else:
                    stagnation_count += 1
                    if stagnation_count > 10:
                        break # Stagnated

                # Centroid of best dim points
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                
                # Reflection
                x_r = centroid + nm_params['alpha'] * (centroid - worst)
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                evals += 1
                
                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + nm_params['gamma'] * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    evals += 1
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    # Reflection
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    if f_r < fvals[-1]:
                        x_c = centroid + nm_params['rho'] * (x_r - centroid)
                    else:
                        x_c = centroid - nm_params['rho'] * (worst - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    evals += 1
                    if f_c < min(f_r, fvals[-1]):
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + nm_params['sigma'] * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = func(simplex[i])
                            evals += 1
                            
                        if evals >= budget - polish_budget:
                            break

        # --- Phase 3: (1+1)-ES Polish ---
        if evals < budget and time_ok():
            x = best_x.copy()
            fx = best_f
            sigma = 0.1 # Start with a reasonable step size
            
            successes = []
            window = 10
            
            while evals < budget and time_ok():
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                evals += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    successes.append(1)
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    successes.append(0)
                
                if len(successes) > window:
                    successes.pop(0)
                    
                if len(successes) == window:
                    rate = np.mean(successes)
                    if rate > 0.2:
                        sigma *= 1.1
                    elif rate < 0.1:
                        sigma *= 0.9
                    
                sigma = np.clip(sigma, 1e-4, 5.0)
                
                # Early stop if sigma is very small and no improvement for a while
                if sigma < 1e-3 and not any(s == 1 for s in successes):
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
