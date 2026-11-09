"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """LHS global sampling followed by Nelder-Mead local polishing for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 315.0  # Safety margin before 330s kill
        budget = self.budget
        dim = self.dim
        
        # We reserve 300 calls for global exploration (LHS)
        # and the remaining 200 for local exploitation (Nelder-Mead restarts)
        n_global = 300
        n_local_budget = budget - n_global
        
        if n_local_budget < 10:
            # Fallback if budget is very small (though task says 500)
            n_global = budget - 5
            n_local_budget = 5

        best_x = None
        best_f = float('inf')
        total_calls = 0
        
        # 1. Global Phase: Latin Hypercube Sampling
        # LHS ensures that each dimension is divided into n_global bins, 
        # and exactly one point is sampled in each bin per dimension.
        # This provides better coverage than uniform random for multimodal functions.
        
        # Generate LHS points
        # For each dimension, create a permutation of the bins
        points = np.zeros((n_global, dim))
        for d in range(dim):
            # Random permutation of 0 to n_global-1
            perm = np.random.permutation(n_global)
            # Map to [0, 1)
            points[:, d] = (perm + np.random.random(n_global)) / n_global
        
        # Map from [0, 1) to [-5, 5]
        points = points * 10.0 - 5.0
        
        # Evaluate all LHS points
        candidates = []
        for i in range(n_global):
            if total_calls >= budget or time.time() - start_time > time_limit:
                break
            x = points[i]
            fx = func(x)
            total_calls += 1
            candidates.append((fx, x.copy()))
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Sort candidates by fitness to pick the best starting points for local search
        candidates.sort(key=lambda item: item[0])
        
        # 2. Local Phase: Nelder-Mead Polishing
        # We will take the top K candidates and run Nelder-Mead on them.
        # Nelder-Mead typically needs ~50-100 evaluations to converge well in 5D.
        # With 200 calls left, we can afford roughly 2-3 good restarts.
        
        # Implement a simple Nelder-Mead step
        def nelder_mead(func, x0, max_iter=100, tol=1e-8, initial_step=1.0):
            nonlocal total_calls, best_f, best_x
            dim = len(x0)
            
            # Initialize simplex
            simplex = [x0.copy() for _ in range(dim + 1)]
            simplex[0] = x0.copy()
            for i in range(1, dim + 1):
                simplex[i] = x0.copy()
                simplex[i][i-1] += initial_step
            
            fvals = []
            for i in range(dim + 1):
                if total_calls >= budget or time.time() - start_time > time_limit:
                    return
                f = func(simplex[i])
                total_calls += 1
                fvals.append(f)
                if f < best_f:
                    best_f = f
                    best_x = simplex[i].copy()
            
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            for _ in range(max_iter):
                if total_calls >= budget or time.time() - start_time > time_limit:
                    return
                
                # Sort simplex
                idx = np.argsort(fvals)
                simplex = [simplex[i] for i in idx]
                fvals = [fvals[i] for i in idx]
                
                # Check convergence
                if np.max(fvals) - np.min(fvals) < tol:
                    break
                
                # Centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                if total_calls >= budget:
                    break
                fr = func(xr)
                total_calls += 1
                if fr < best_f:
                    best_f = fr
                    best_x = xr.copy()
                
                # Expansion
                if fr < fvals[0]:
                    xe = centroid + gamma * (xr - centroid)
                    xe = np.clip(xe, -5, 5)
                    if total_calls >= budget:
                        break
                    fe = func(xe)
                    total_calls += 1
                    if fe < best_f:
                        best_f = fe
                        best_x = xe.copy()
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                # Contraction
                elif fr < fvals[-2]:
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    if total_calls >= budget:
                        break
                    fc = func(xc)
                    total_calls += 1
                    if fc < best_f:
                        best_f = fc
                        best_x = xc.copy()
                    if fc < fr:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            if total_calls >= budget:
                                break
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f = func(simplex[i])
                            total_calls += 1
                            fvals[i] = f
                            if f < best_f:
                                best_f = f
                                best_x = simplex[i].copy()
                        break
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        if total_calls >= budget:
                            break
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        f = func(simplex[i])
                        total_calls += 1
                        fvals[i] = f
                        if f < best_f:
                            best_f = f
                            best_x = simplex[i].copy()
                    break

        # Select top K candidates for local search
        # We want to spend our remaining budget on the most promising global points.
        # Let's estimate calls per NM run as ~60.
        max_restarts = max(1, n_local_budget // 60)
        
        # Ensure we don't pick duplicate or very close points
        used_starts = []
        
        for i in range(min(max_restarts, len(candidates))):
            fx, x = candidates[i]
            
            # Check if this point is too close to a previously used start
            # (Simple distance check to avoid redundant work)
            too_close = False
            for ux in used_starts:
                if np.linalg.norm(x - ux) < 0.5:
                    too_close = True
                    break
            if too_close:
                continue
                
            used_starts.append(x)
            
            if total_calls >= budget or time.time() - start_time > time_limit:
                break
            
            # Run Nelder-Mead
            # We pass a max_iter that is conservative to ensure we don't blow the budget
            # inside the loop, but the internal checks handle it.
            nelder_mead(func, x, max_iter=50)
            
            if time.time() - start_time > time_limit:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
