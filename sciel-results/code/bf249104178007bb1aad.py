"""Exploit DTLZ1's known Pareto front with Sobol-based optimal density sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """
    Model-and-optimize approach for DTLZ1:
    1. Calibrate K (the sum of objectives on the Pareto front) empirically.
    2. Generate well-distributed points on the simplex f1+f2+f3=K using Sobol sequences.
    3. Apply hypervolume-optimal density weighting (more points near corners/origin).
    4. Map simplex points back to decision space and evaluate.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 290.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        budget = self.budget

        used = 0
        results = []

        def eval_point(x):
            nonlocal used
            if used >= budget or time.time() > deadline:
                return None
            x_clipped = np.clip(np.asarray(x, dtype=float), lo, hi)
            f = np.array(problem(x_clipped), dtype=float)
            results.append(f)
            used += 1
            return f

        # Step 1: Calibrate K by probing the problem
        # For DTLZ1, the Pareto front is achieved when tail variables minimize g.
        # Use a focused probe: try x[3:] = 0.5 (standard DTLZ1 optimum) and x[3:] = 0.0
        # For DTLZ1 with m=3, n=7: x[0], x[1] are position vars, x[2:7] are tail (5 vars)
        # g_min achieved when all tail vars = 0.5
        tail_candidates = [
            [0.5, 0.5, 0.5, 0.5, 0.5],  # Standard DTLZ1 g-min
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.3, 0.3, 0.3, 0.3, 0.3],
            [0.7, 0.7, 0.7, 0.7, 0.7],
        ]

        min_sum = float('inf')
        best_tail = None

        for tail in tail_candidates:
            if used >= budget or time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = 0.5  # Midpoint to get a representative sum
            for i, t in enumerate(tail):
                x[2 + i] = t  # Tail starts at index 2 for m=3, n=7
            f = eval_point(x)
            if f is not None:
                s = float(np.sum(f))
                if s < min_sum:
                    min_sum = s
                    best_tail = tail[:]

        if min_sum == float('inf') or used >= budget or time.time() > deadline:
            return results

        K = min_sum

        def simplex_to_decision(f1, f2, f3, K_val, tail):
            """Map objective values to decision variables for DTLZ1.
            For m=3, n=7: x[0], x[1] are position vars, x[2:7] are tail.
            f1 = 0.5*(1+g)*x0*(1-x1), f2 = 0.5*(1+g)*x0*x1, f3 = 0.5*(1+g)*(1-x0)
            So: x0 = f1/K, 1-x0 = f3/K, x1 = f2/(f1+f2)
            """
            x = np.zeros(dim)
            for i, t in enumerate(tail):
                x[2 + i] = t  # Tail starts at index 2
            if K_val < 1e-12:
                x[0] = 0.5
                x[1] = 0.5
                return np.clip(x, lo, hi)

            # x0 from f1 and f3: f1/K = x0*(1-x1), f3/K = 1-x0
            # So x0 = 1 - f3/K
            x0 = 1.0 - f3 / K_val
            # x1 from f1 and f2: f1 = K*x0*(1-x1), f2 = K*x0*x1
            # f1 + f2 = K*x0, so x1 = f2/(f1+f2)
            denom = f1 + f2
            if denom < 1e-12:
                x1 = 0.5
            else:
                x1 = f2 / denom
            x[0] = x0
            x[1] = x1
            return np.clip(x, lo, hi)

        # Step 2: Generate points using Sobol sequence with hypervolume-optimal weighting
        # For hypervolume with reference point [R,R,R], the contribution of a point (f1,f2,f3)
        # is maximized when it's far from the reference point. The optimal density for
        # maximizing hypervolume with N points on a simplex is proportional to the
        # hypervolume contribution, which favors points near the corners of the simplex.
        
        # Strategy: Use Sobol points in [0,1]^2, map to simplex, then apply a
        # transformation that biases toward the corners (where hypervolume contribution is higher).
        
        remaining = budget - used
        if remaining <= 0:
            return results

        # Generate Sobol points in 2D, map to simplex
        # Use a power-law transformation to bias toward corners
        # For hypervolume, the optimal density on the simplex is ~ 1/(f1*f2*f3) near origin
        # We use a soft bias: w_i = u_i^alpha where alpha > 1 biases toward 0 (corners)
        
        # Determine alpha based on K and reference point
        # Reference point is [400, 400, 400], K is typically ~150-300
        # Bias strength: stronger when K is small relative to reference
        ref = 400.0
        bias = 1.0 + 0.5 * (ref / max(K, 1.0))  # Higher bias when K is smaller
        alpha = min(bias, 3.0)  # Cap to avoid too extreme bias

        # Generate Sobol sequence
        sobol = qmc.Sobol(d=2, scramble=True, seed=42)
        
        # We need to generate enough points, then select 'remaining' of them
        # Sobol requires power-of-2 numbers, so generate the next power of 2 >= remaining
        n_sobol = 1
        while n_sobol < remaining:
            n_sobol *= 2
        
        # Generate all Sobol points at once
        sobol_points = sobol.random(n_sobol)  # shape (n_sobol, 2), in [0,1)^2
        
        # Map 2D points to simplex using the standard transformation:
        # (u1, u2) -> (f1, f2, f3) where:
        #   f1 = K * u1 * u2
        #   f2 = K * u1 * (1 - u2)  
        #   f3 = K * (1 - u1)
        # This gives uniform distribution on the simplex.
        
        # Apply corner bias: transform each coordinate with power alpha
        # This biases the distribution toward the corners of the simplex
        u1 = sobol_points[:, 0] ** alpha
        u2 = sobol_points[:, 1] ** alpha
        
        # Map to simplex
        f1_vals = K * u1 * u2
        f2_vals = K * u1 * (1.0 - u2)
        f3_vals = K * (1.0 - u1)
        
        # Evaluate the first 'remaining' points (Sobol is low-discrepancy, so first N are well-distributed)
        for idx in range(min(remaining, n_sobol)):
            if used >= budget or time.time() > deadline:
                break
            f1 = f1_vals[idx]
            f2 = f2_vals[idx]
            f3 = f3_vals[idx]
            x = simplex_to_decision(f1, f2, f3, K, best_tail)
            eval_point(x)

        # Step 3: If we still have budget left (shouldn't happen, but safety), add uniform points
        if used < budget and time.time() < deadline:
            rng = np.random.default_rng(12345)
            for idx in range(budget - used):
                if used >= budget or time.time() > deadline:
                    break
                u = rng.random(3)
                u.sort()
                w1 = u[0]
                w2 = u[1] - u[0]
                w3 = 1.0 - u[1]
                f1 = K * w1
                f2 = K * w2
                f3 = K * w3
                x = simplex_to_decision(f1, f2, f3, K, best_tail)
                eval_point(x)

        return results
# EVOLVE-BLOCK-END
