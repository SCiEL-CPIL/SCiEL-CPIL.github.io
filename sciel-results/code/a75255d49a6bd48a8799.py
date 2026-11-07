"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start Nelder-Mead ensemble for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # We use a small number of starts to ensure each has enough budget
        # to converge locally. 5 starts * ~100 evals each is a good balance.
        n_starts = 5
        
        # Generate initial points. 
        # To handle the hidden shift, we sample from the whole box [-5, 5]^dim.
        # Latin Hypercube Sampling (LHS) is better than pure random for coverage.
        # For simplicity and robustness, we use stratified sampling per dimension.
        
        # Generate LHS points
        # Divide [0, 1] into n_starts intervals for each dimension
        # Shuffle the order of intervals for each dimension
        points = np.zeros((n_starts, dim))
        for d in range(dim):
            # Intervals: [0, 1/n], [1/n, 2/n], ...
            # Sample one point from each interval
            indices = rng.permutation(n_starts)
            for i in range(n_starts):
                # Pick a random point within the interval assigned to index i
                # The interval for rank 'rank' is [rank/n, (rank+1)/n]
                rank = indices[i]
                low = rank / n_starts
                high = (rank + 1) / n_starts
                points[i, d] = low + rng.random() * (high - low)
        
        # Map [0, 1]^dim to [-5, 5]^dim
        points = points * 10.0 - 5.0

        best_x = None
        best_f = np.inf
        
        # Allocate budget roughly equally, but allow early termination if converged
        # Nelder-Mead typically needs ~10-20 * dim evals to converge well.
        # With dim=5, that's ~50-100 evals.
        # 500 budget / 5 starts = 100 evals per start.
        evals_per_start = max(10, budget // n_starts)
        
        for i in range(n_starts):
            if time.time() > deadline:
                break
                
            x0 = points[i]
            
            # Run Nelder-Mead
            # We implement a lightweight Nelder-Mead to avoid scipy overhead and allow tight budget control
            # Standard NM parameters
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink
            
            # Initialize simplex
            # Start at x0, and expand along axes
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = x0
            for j in range(dim):
                simplex[j + 1] = x0.copy()
                # Step size: 10% of box size (5.0) is reasonable
                step = 1.0
                if simplex[j + 1, j] + step > 5.0:
                    simplex[j + 1, j] = simplex[j + 1, j] - step
                else:
                    simplex[j + 1, j] = simplex[j + 1, j] + step
            
            # Evaluate initial simplex
            fvals = np.zeros(dim + 1)
            for k in range(dim + 1):
                if time.time() > deadline:
                    break
                fvals[k] = func(simplex[k])
            
            # Track best so far
            idx_min = np.argmin(fvals)
            if fvals[idx_min] < best_f:
                best_f = fvals[idx_min]
                best_x = simplex[idx_min].copy()
            
            # Main NM loop
            # We stop if simplex is very small or budget/time is up
            max_nm_iters = 2 * dim * 5 # Heuristic limit
            nm_iter = 0
            
            while nm_iter < max_nm_iters:
                if time.time() > deadline:
                    break
                
                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]
                
                # Check convergence: if range of fvals is small
                if fvals[-1] - fvals[0] < 1e-8:
                    break
                
                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                # Clip to box? NM doesn't handle constraints natively.
                # The task says func clips inputs. So we can just pass xr.
                # However, if we go far outside, we waste evals. 
                # Clipping the *search* points to box is safer for efficiency.
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                
                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (xr - centroid)
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    # Outside contraction if reflection was better than second worst, inside otherwise
                    # Standard NM: Inside contraction if fr < fvals[-2] (which is false here)
                    # So we do Inside Contraction
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for k in range(1, dim + 1):
                            simplex[k] = simplex[0] + sigma * (simplex[k] - simplex[0])
                            fvals[k] = func(simplex[k])
                        # After shrink, we must re-evaluate all
                        # Note: This is expensive. We might break if budget is tight.
                        # But shrink is rare.
                        
                # Update global best
                idx_min = np.argmin(fvals)
                if fvals[idx_min] < best_f:
                    best_f = fvals[idx_min]
                    best_x = simplex[idx_min].copy()
                    
                nm_iter += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
