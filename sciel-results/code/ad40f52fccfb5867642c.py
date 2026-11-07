"""Tchebycheff decomposition + Nelder-Mead local search for WFG7."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Decomposition-based optimizer with Tchebycheff scalarization."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.perf_counter()
        deadline = t0 + 270.0  # safety margin before 300s

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim, n_obj = self.dim, self.n_obj
        budget = self.budget

        # Generate 30 uniformly distributed weight vectors in the simplex
        # For 3 objectives, weights are (w1, w2, w3) with w1+w2+w3=1, wi>=0
        n_weights = 30
        weights = []
        # Use a grid approach for uniform distribution on the simplex
        # For 3 objectives: w1 from 0 to 1, w2 from 0 to 1-w1, w3 = 1-w1-w2
        # We'll use a stratified sampling approach
        for i in range(n_weights):
            # Use golden ratio for better distribution
            t1 = (i + 0.5) / n_weights
            t2 = (i * 0.618033988749895 + 0.5) % 1.0
            t2 = t2 * (1 - t1)
            w1 = t1
            w2 = t2
            w3 = 1 - w1 - w2
            # Ensure non-negative
            if w3 < 0:
                w3 = 0
                s = w1 + w2
                w1 /= s
                w2 /= s
            weights.append([w1, w2, w3])
        weights = np.array(weights)

        # Evaluate center point first
        center = (lo + hi) / 2.0
        f_center = problem(center)
        used = 1

        # For each weight vector, run Nelder-Mead to minimize the Tchebycheff scalarization
        # g(x) = max_i (w_i * (f_i(x) - z_i^*))
        # where z^* is the ideal point (we approximate with the best seen so far)

        # First pass: get initial estimates for ideal point
        # Evaluate a few more points to get a rough ideal point
        rng = np.random.default_rng(42)
        extra_points = 10
        if used + extra_points <= budget:
            for _ in range(extra_points):
                x = lo + (hi - lo) * rng.random(dim)
                f = problem(x)
                used += 1
                if used >= budget:
                    break

        # We need to track the best (minimum) objective values seen
        # Let's collect all evaluated points
        all_X = [center]
        all_F = [f_center]
        
        # Add the random points
        for _ in range(extra_points):
            if used > 1 + extra_points:
                break
            x = lo + (hi - lo) * rng.random(dim)
            f = problem(x)
            used += 1
            all_X.append(x)
            all_F.append(f)
        
        all_F = np.array(all_F)
        ideal = np.min(all_F, axis=0)

        # Now for each weight vector, run a local search
        # Tchebycheff scalarization: T(x) = max_i (w_i * |f_i(x) - ideal_i|)
        # We minimize this using Nelder-Mead

        def tchebycheff_scalar(x, w):
            f = problem(x)
            used[0] += 1
            # Clip to bounds (problem does this, but let's be safe)
            f = np.clip(f, 0, 100)  # Ensure finite
            return np.max(w * (f - ideal))

        # Use a wrapper to track evaluations
        eval_count = [used]
        best_solutions = []

        # Simple Nelder-Mead implementation
        def nelder_mead(f, x0, bounds, max_iter=50):
            """Simple Nelder-Mead implementation."""
            n = len(x0)
            # Initial simplex
            simplex = [x0.copy()]
            step = 0.1 * (bounds[1] - bounds[0])
            for i in range(n):
                xi = x0.copy()
                xi[i] += step[i]
                simplex.append(xi)
            
            fvals = [f(x) for x in simplex]
            simplex = np.array(simplex)
            fvals = np.array(fvals)
            
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma = 0.5  # shrink
            
            for it in range(max_iter):
                # Sort by function value
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflect
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, bounds[0], bounds[1])
                fr = f(xr)
                
                if fr < fvals[0]:
                    # Try expansion
                    xe = centroid + gamma * (xr - centroid)
                    xe = np.clip(xe, bounds[0], bounds[1])
                    fe = f(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        xc = centroid + rho * (simplex[-1] - centroid)
                    else:
                        xc = centroid - rho * (centroid - simplex[-1])
                    xc = np.clip(xc, bounds[0], bounds[1])
                    fc = f(xc)
                    if fc <= max(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], bounds[0], bounds[1])
                            fvals[i] = f(simplex[i])
            
            # Return best point
            best_idx = np.argmin(fvals)
            return simplex[best_idx], fvals[best_idx]

        # Run local search for each weight vector
        bounds_lo = lo
        bounds_hi = hi
        
        for w in weights:
            if eval_count[0] >= budget:
                break
            if time.perf_counter() > deadline:
                break
            
            # Start from center
            x0 = center.copy()
            
            # Define scalarization function with evaluation tracking
            def scalar_func(x):
                if eval_count[0] >= budget or time.perf_counter() > deadline:
                    return 1e10
                f = problem(x)
                eval_count[0] += 1
                return np.max(w * (f - ideal))
            
            # Run Nelder-Mead with limited iterations
            x_best, f_best = nelder_mead(scalar_func, x0, (bounds_lo, bounds_hi), max_iter=30)
            
            if eval_count[0] < budget:
                # Evaluate the best solution
                f_final = problem(x_best)
                eval_count[0] += 1
                best_solutions.append((x_best, f_final))
        
        # Collect all evaluated points for the final result
        # We need to return the non-dominated set
        # Let's collect all objective vectors we've seen
        
        # Actually, let's just return the best solutions from each subproblem
        # and also include the initial points
        
        all_results = []
        for x, f in best_solutions:
            all_results.append(f)
        
        # Also add the center and random points
        all_results.append(f_center)
        
        # Return the best solutions
        return best_solutions
# EVOLVE-BLOCK-END
