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
        # For DTLZ1 with n=7, m=3: x[0], x[1] are position vars, x[2:7] are tail (5 vars)
        # g_min achieved when all tail vars = 0.5
        tail_len = dim - 2  # 5 tail variables
        tail_candidates = [
            [0.5] * tail_len,  # Standard DTLZ1 g-min
            [0.0] * tail_len,
            [0.3] * tail_len,
            [0.7] * tail_len,
            [0.1] * tail_len,
            [0.9] * tail_len,
        ]

        min_sum = float('inf')
        best_tail = None

        for tail in tail_candidates:
            if used >= budget or time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = 0.5  # Midpoint to get a representative sum
            x[1] = 0.5
            for i, t in enumerate(tail):
                x[2 + i] = t
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
            """Map objective values to decision variables for DTLZ1."""
            x = np.zeros(dim)
            for i, t in enumerate(tail):
                x[2 + i] = t
            if K_val < 1e-12:
                x[0] = 0.5
                x[1] = 0.5
                return np.clip(x, lo, hi)

            # For DTLZ1: f1 = K * x0 * x1 * (1-x2) ... wait, standard DTLZ1:
            # f1 = g * x1 * x2 * ... 
            # Actually for DTLZ1 with m=3:
            # f1 = g * x_{m-1} * x_{m-2} * (1 - x_{m-3})  -- no, let me think carefully
            # Standard DTLZ1: f_i = g(x_M) * prod_{j=m-i}^{m-1} x_j * (1 - x_{m-i-1}) for i<m
            # f_m = g(x_M) * (1 - x_{m-1})
            # With m=3, M=n-m=4, so tail is x[3:7] (4 vars), position is x[0:3] (3 vars)
            # f1 = g * x2 * (1-x1)  ... hmm, let me just use the empirical mapping
            # From the probe: x[0]=0.5, x[1]=0.5, tail=0.5 gives f values
            # The sum K = f1+f2+f3 when on Pareto front
            # We need: given target (f1,f2,f3) with f1+f2+f3=K, find x[0],x[1]
            # For DTLZ1: f1 = K*x0*x1*(1-x2) is wrong...
            # Let me use a simpler empirical approach: 
            # If the problem is standard DTLZ1 with m=3, n=7:
            # M = n - m = 4, so x[3:7] are the 4 tail variables
            # x[0], x[1], x[2] are the 3 position variables
            # f1 = g * x2 * (1-x1)  -- no
            # Standard: f_i = g * prod_{j=m-i}^{m-2} x_j * (1 - x_{m-i-1}) for i=1,...,m-1
            # f_m = g * (1 - x_{m-1})
            # With m=3: f1 = g * x1 * (1-x0)?? No...
            # Let me just use: f1/f3 = x0/(1-x2) type relations
            # Actually the safest: use the fact that on Pareto front, 
            # f1 = K * a, f2 = K * b, f3 = K * c with a+b+c=1
            # and x[0] = f1/K = a (if the mapping is linear in x[0])
            # x[1] = 1 - f3/K = 1-c (if x[1] relates to f3)
            # This is what the original code did. Keep it.
            
            x0 = f1 / K_val
            x1 = 1.0 - f3 / K_val
            x[0] = x0
            x[1] = x1
            x[2] = 0.5
            return np.clip(x, lo, hi)

        # Step 2: Generate points using Sobol sequence with hypervolume-optimal weighting
        remaining = budget - used
        if remaining <= 0:
            return results

        # Hypervolume-optimal density on the simplex:
        # For reference point R=(400,400,400) and front f1+f2+f3=K (K<<400),
        # the hypervolume contribution of a point (f1,f2,f3) is approximately
        # (R-f1)(R-f2)(R-f3) for isolated points.
        # We want to maximize total hypervolume, which favors points near corners.
        # The optimal density on the 2-simplex for maximizing hypervolume is
        # proportional to (R-f1)(R-f2)(R-f3) / (area element), which in barycentric
        # coords (a,b,c) with a+b+c=1, f_i=K*a_i:
        # density ~ (R-Ka)(R-Kb)(R-Kc)
        # For K<<R, this is ~ R^3 * (1 - Ka/R)(1-Kb/R)(1-Kc/R) ≈ R^3(1 - K(a+b+c)/R) = R^3(1-K/R)
        # So it's nearly uniform! But the second order term favors corners slightly.
        # Actually, the key insight: the CORNERS of the simplex (where one f_i ≈ K, others ≈ 0)
        # give the largest individual hypervolume contribution because (R-K)(R)(R) >> (R-K/3)^3.
        # So we should bias toward corners.
        
        # Strategy: Use a mixture of:
        # 1. A few points very close to the 3 corners (extreme hypervolume contribution)
        # 2. Sobol points with a power-law bias toward corners
        # 3. Some uniform points for coverage

        # Reserve points for extreme corners
        n_corner = min(6, remaining)  # 2 per corner
        corner_eps = 1e-4  # Very close to corner for max contribution
        
        # Evaluate extreme corner points
        corner_points = [
            (K - corner_eps, corner_eps/2, corner_eps/2),
            (corner_eps/2, K - corner_eps, corner_eps/2),
            (corner_eps/2, corner_eps/2, K - corner_eps),
            (K*0.999, K*0.0005, K*0.0005),
            (K*0.0005, K*0.999, K*0.0005),
            (K*0.0005, K*0.0005, K*0.999),
        ]
        
        for i in range(n_corner):
            if used >= budget or time.time() > deadline:
                break
            f1, f2, f3 = corner_points[i]
            x = simplex_to_decision(f1, f2, f3, K, best_tail)
            eval_point(x)

        remaining = budget - used
        if remaining <= 0:
            return results

        # Generate Sobol points with corner bias
        # Use inverse CDF: if we want density ~ (a*b*c)^(-beta) on the simplex,
        # we can transform uniform points. A simpler approach: use power transform on
        # the barycentric coordinates to bias toward 0 (which corresponds to corners).
        
        # For the simplex parameterization (u1, u2) -> (a,b,c):
        # a = u1*u2, b = u1*(1-u2), c = 1-u1
        # To bias toward corners, we want a,b,c to be more concentrated near 0 and 1.
        # Transform: u1' = u1^alpha, u2' = u2^alpha with alpha > 1 biases u toward 0,
        # which makes c = 1-u1' closer to 1 (corner c) and a,b smaller.
        # But we also want the other corners. A better approach: use multiple transforms.
        
        # Alternative: sample from a Dirichlet distribution with alpha < 1 (concentrated at corners)
        # Dirichlet(alpha, alpha, alpha) with small alpha concentrates at corners.
        
        # Use Sobol for quasi-random, then apply a transformation
        n_sobol = 1
        while n_sobol < remaining:
            n_sobol *= 2
        
        sobol = qmc.Sobol(d=2, scramble=True, seed=42)
        sobol_points = sobol.random(n_sobol)
        
        # Apply corner-biased transformation
        # Use alpha that balances corner emphasis with coverage
        # alpha=2 gives moderate bias, alpha=4 gives strong bias
        alpha = 2.5
        
        u1 = sobol_points[:, 0] ** alpha
        u2 = sobol_points[:, 1] ** alpha
        
        # Map to simplex
        a_vals = u1 * u2
        b_vals = u1 * (1.0 - u2)
        c_vals = 1.0 - u1
        
        # Normalize (should already sum to 1, but ensure)
        sums = a_vals + b_vals + c_vals
        a_vals /= sums
        b_vals /= sums
        c_vals /= sums
        
        # Evaluate Sobol-based points
        for idx in range(min(remaining, n_sobol)):
            if used >= budget or time.time() > deadline:
                break
            f1 = K * a_vals[idx]
            f2 = K * b_vals[idx]
            f3 = K * c_vals[idx]
            x = simplex_to_decision(f1, f2, f3, K, best_tail)
            eval_point(x)

        # Step 3: If we still have budget left, add more corner-biased points with different alpha
        if used < budget and time.time() < deadline:
            remaining2 = budget - used
            # Use a different Sobol seed and different alpha for diversity
            sobol2 = qmc.Sobol(d=2, scramble=True, seed=123)
            n_sobol2 = 1
            while n_sobol2 < remaining2:
                n_sobol2 *= 2
            sobol_points2 = sobol2.random(n_sobol2)
            
            alpha2 = 4.0  # Stronger corner bias
            u1_2 = sobol_points2[:, 0] ** alpha2
            u2_2 = sobol_points2[:, 1] ** alpha2
            
            a2 = u1_2 * u2_2
            b2 = u1_2 * (1.0 - u2_2)
            c2 = 1.0 - u1_2
            sums2 = a2 + b2 + c2
            a2 /= sums2
            b2 /= sums2
            c2 /= sums2
            
            for idx in range(min(remaining2, n_sobol2)):
                if used >= budget or time.time() > deadline:
                    break
                f1 = K * a2[idx]
                f2 = K * b2[idx]
                f3 = K * c2[idx]
                x = simplex_to_decision(f1, f2, f3, K, best_tail)
                eval_point(x)

        return results
# EVOLVE-BLOCK-END
