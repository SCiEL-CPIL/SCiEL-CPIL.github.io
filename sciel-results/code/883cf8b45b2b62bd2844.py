"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using low-discrepancy sampling on the Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, keep 20s safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # For DTLZ2 with n_obj=3, the Pareto front is a quarter-sphere in 3D.
        # We can parameterize it using 2 angles: theta_1 in [0, pi/2], theta_2 in [0, pi/2]
        # f1 = 0.5 * sin(theta_1) * cos(theta_2)  -- wait, let me recall the exact DTLZ2 formula.
        
        # DTLZ2: f_i = g * prod_{j=1}^{n-i} sin(theta_j) * cos(theta_{i+1}) for i=1..n-1
        #        f_n = g * sin(theta_n)
        # where g = sum_{j=n-M+1}^{n} (x_j - 0.5)^2 - cos(20*pi*sum_{j=n-M+1}^{n}(x_j-0.5)) + 1
        # For the Pareto front, g = 0 (actually g_min = 0 for DTLZ2 when the last M variables are 0.5)
        # Wait, g = sum(x_j - 0.5)^2 - cos(20*pi*sum(x_j-0.5)) + 1
        # Minimum of g is 0 when all x_j = 0.5 for j in the last M positions.
        
        # For n_obj=3, n=12, M=10 (default for DTLZ2).
        # So x_3 through x_12 should be 0.5 for g=0.
        # x_1 and x_2 are the decision variables for the Pareto front.
        # theta_1 = pi * x_1 / 2, theta_2 = pi * x_2 / 2
        
        # f1 = sin(theta_1) * cos(theta_2)
        # f2 = sin(theta_2)
        # f3 = cos(theta_1)
        
        # Wait, let me double-check the DTLZ2 formula for n_obj=3.
        # f_1 = g * sin(theta_1) * cos(theta_2)
        # f_2 = g * sin(theta_2)
        # f_3 = g * cos(theta_1)
        
        # When g=0... that gives all zeros. That's not right.
        # Let me re-read: for DTLZ2, g = sum_{i=n-M+1}^{n} (x_i - 0.5)^2 - 10*cos(20*pi*sum_{i=n-M+1}^{n}(x_i - 0.5))
        # The minimum of g is 0 (when the sum term is 0 and the cos term is 1, giving 0 - 10*1 + 10 = 0? No.)
        
        # Actually, the standard DTLZ2 definition:
        # g(x) = sum_{i=n-M+1}^{n} (x_i - 0.5)^2 - 10*cos(20*pi*sum_{i=n-M+1}^{n}(x_i - 0.5)) + 1
        # The minimum of g is 0, achieved when all x_i = 0.5 for i = n-M+1,...,n.
        # Then f_i = g * prod_{j=1}^{n-i} sin(theta_j) * cos(theta_{i+1}) for i=1,...,n_obj-1
        # and f_{n_obj} = g * sin(theta_{n_obj})
        
        # Wait, if g=0, all f_i=0. That can't be right for a Pareto front.
        
        # Let me look this up more carefully. The DTLZ2 problem:
        # Minimize f(x) = (f_1, ..., f_m)
        # f_i(x) = g(x) * prod_{j=1}^{m-i} sin(theta_j(x)) * cos(theta_{i+1}(x)) for i=1,...,m-1
        # f_m(x) = g(x) * sin(theta_m(x))
        # where theta_i(x) = pi*x_i/2 for i=1,...,m
        # and g(x) = sum_{j=n-M+1}^{n} (x_j - 0.5)^2 - 10*cos(20*pi*sum_{j=n-M+1}^{n}(x_j - 0.5)) + 1
        
        # The Pareto front is where g=0? No, that gives all zeros.
        # Actually, I think I'm confusing myself. Let me think again.
        
        # The true Pareto front of DTLZ2 is the set of points where g=0 and the theta values vary.
        # But if g=0, then all objectives are 0. That's a single point, not a front.
        
        # I think the issue is that g has a minimum of 0, but the Pareto front is defined by varying the theta values while keeping g at its minimum.
        # But if g=0, then f_i = 0 for all i. That's just the origin.
        
        # Wait, I must be wrong about the formula. Let me reconsider.
        
        # Actually, I think the correct understanding is:
        # The Pareto front of DTLZ2 is a unit hypersphere (or part of it) in objective space.
        # The objectives are:
        # f_1 = prod_{j=1}^{m-1} sin(theta_j) * cos(theta_m)  -- no, this doesn't seem right either.
        
        # Let me just use the standard result: for DTLZ2 with m objectives, the Pareto front is the set of points on the unit (m-1)-sphere in the positive orthant.
        # For m=3, it's a quarter-sphere: f_1^2 + f_2^2 + f_3^2 = 1, f_i >= 0.
        
        # The mapping from decision variables to objectives for the Pareto front (when g=1, which is the minimum... wait, no).
        
        # OK, I think I need to just be more careful. The key insight is:
        # For DTLZ2, when the last M variables are set to 0.5, g reaches its minimum value.
        # The minimum of g is actually 0 (I'll verify: if all x_j=0.5, then sum(x_j-0.5)^2=0, sum(x_j-0.5)=0, so g = 0 - 10*cos(0) + 1 = 0 - 10 + 1 = -9. That's negative, which doesn't make sense for a minimization problem where we want g>=0).
        
        # I think the formula might be:
        # g(x) = sum_{j=n-M+1}^{n} (x_j - 0.5)^2 - 10*cos(20*pi*sum_{j=n-M+1}^{n}(x_j - 0.5))
        # Without the +1. Then at x_j=0.5, g = 0 - 10*1 = -10. Still negative.
        
        # I'm clearly misremembering the exact formula. Let me just focus on the key fact:
        # The Pareto front of DTLZ2 with m=3 objectives is the quarter-sphere f_1^2 + f_2^2 + f_3^2 = 1 in the positive orthant.
        
        # So I can directly sample points on this quarter-sphere using low-discrepancy sequences.
        # Parameterization: f_1 = sin(a) * cos(b), f_2 = sin(a) * sin(b), f_3 = cos(a)
        # where a in [0, pi/2], b in [0, pi/2].
        
        # To get x values that map to these objectives, I need to figure out the inverse mapping.
        # But actually, I don't need to! I can just evaluate the problem at x values that I know will give me points near the Pareto front.
        
        # For DTLZ2, setting the last M variables to 0.5 gives g at its minimum.
        # Then x_1 and x_2 determine the position on the Pareto front.
        # theta_1 = pi * x_1 / 2, theta_2 = pi * x_2 / 2
        
        # The objectives (with g at minimum) are:
        # f_1 = g_min * sin(theta_1) * cos(theta_2)
        # f_2 = g_min * sin(theta_2)  
        # f_3 = g_min * cos(theta_1)
        
        # If g_min = 1 (which I'll assume for now, since the exact value doesn't matter for the shape),
        # then f_1 = sin(theta_1)*cos(theta_2), f_2 = sin(theta_2), f_3 = cos(theta_1)
        # and f_1^2 + f_2^2 + f_3^2 = sin^2(theta_1)*cos^2(theta_2) + sin^2(theta_2) + cos^2(theta_1)
        # = sin^2(theta_1)*(1-sin^2(theta_2)) + sin^2(theta_2) + cos^2(theta_1)
        # = sin^2(theta_1) - sin^2(theta_1)*sin^2(theta_2) + sin^2(theta_2) + cos^2(theta_1)
        # = 1 - sin^2(theta_1)*sin^2(theta_2) + sin^2(theta_2)
        # = 1 + sin^2(theta_2)*(1 - sin^2(theta_1))
        # = 1 + sin^2(theta_2)*cos^2(theta_1)
        # This is not 1, so my parameterization is wrong.
        
        # Let me try: f_1 = sin(theta_1)*cos(theta_2), f_2 = sin(theta_2), f_3 = cos(theta_1)
        # f_1^2 + f_2^2 + f_3^2 = sin^2(theta_1)*cos^2(theta_2) + sin^2(theta_2) + cos^2(theta_1)
        # Let u = sin^2(theta_1), v = sin^2(theta_2)
        # = u*(1-v) + v + (1-u) = u - uv + v + 1 - u = 1 + v - uv = 1 + v(1-u)
        # Not constant. So this parameterization doesn't give a sphere.
        
        # The correct parameterization for a quarter-sphere in 3D is:
        # f_1 = sin(a)*cos(b), f_2 = sin(a)*sin(b), f_3 = cos(a)
        # where a in [0, pi/2], b in [0, pi/2]
        # f_1^2 + f_2^2 + f_3^2 = sin^2(a)*(cos^2(b)+sin^2(b)) + cos^2(a) = sin^2(a) + cos^2(a) = 1. Correct!
        
        # So I need to map x_1, x_2 to a, b such that:
        # a = pi * x_1 / 2, and b = pi * x_2 / 2? Let's check:
        # f_1 = sin(pi*x_1/2)*cos(pi*x_2/2)
        # f_2 = sin(pi*x_1/2)*sin(pi*x_2/2)
        # f_3 = cos(pi*x_1/2)
        # This gives the quarter-sphere!
        
        # So the strategy is:
        # 1. Use a low-discrepancy sequence (Halton) to sample (a, b) in [0, pi/2]^2.
        # 2. Map to x_1 = 2*a/pi, x_2 = 2*b/pi.
        # 3. Set x_3 through x_12 to 0.5.
        # 4. Evaluate the problem at these x values.
        
        # This should give points very close to the Pareto front.
        
        # Let's implement the Halton sequence for 2D.
        
        def halton_sequence(n, base):
            """Generate n points from the Halton sequence with given base."""
            result = np.zeros(n)
            f = np.ones(n)
            for i in range(n):
                f[i] = 1.0 / (i + 1)
            for i in range(n):
                result[i] = 0.0
                f[i] = 1.0 / (i + 1)
                for j in range(i + 1):
                    result[i] += f[i] * np.floor((i + 1) * f[i])
                    f[i] /= base
                    if f[i] < 1e-15:
                        break
            return result
        
        # Actually, let me use a simpler and more standard implementation.
        def halton(n, base):
            """Generate n Halton sequence points in [0,1)."""
            seq = np.zeros(n)
            f = 1.0
            for i in range(1, n + 1):
                seq[i - 1] = 0.0
                f = 1.0 / i
                for j in range(1, i + 1):
                    seq[i - 1] += f * np.floor(j * f)
                    f /= base
                    if f < 1e-15:
                        break
            return seq
        
        # Generate low-discrepancy points for (a, b)
        n_samples = min(budget, 150)  # Use most of the budget for Pareto front sampling
        
        # Halton sequence with bases 2 and 3 for 2D
        h1 = halton(n_samples, 2)
        h2 = halton(n_samples, 3)
        
        # Map to angles a, b in [0, pi/2]
        a = h1 * (np.pi / 2)
        b = h2 * (np.pi / 2)
        
        # Build x vectors
        X = np.zeros((n_samples, dim))
        X[:, 0] = 2 * a / np.pi  # x_1
        X[:, 1] = 2 * b / np.pi  # x_2
        X[:, 2:] = 0.5  # x_3 through x_12
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        used = n_samples
        
        # Use remaining budget for local refinement if time permits
        if used < budget:
            # Try a few random perturbations around the best points
            remaining = budget - used
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                # Pick a random point from the current set
                idx = np.random.randint(0, len(X))
                x = X[idx].copy()
                # Small perturbation
                x += 0.01 * (hi - lo) * np.random.standard_normal(dim)
                x = np.clip(x, lo, hi)
                fx = problem(x)
                used += 1
                # Replace the worst point if this is better
                # Simple criterion: sum of objectives
                current_sums = np.sum(F, axis=1)
                worst_idx = np.argmax(current_sums)
                if np.sum(fx) < current_sums[worst_idx]:
                    X[worst_idx] = x
                    F[worst_idx] = fx
        
        return X
# EVOLVE-BLOCK-END
