"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Analytical DTLZ4 optimizer using Halton sequence on the Pareto front parameter space."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.t0 = time.time()
        self.deadline = self.t0 + 280  # Safety margin for 300s limit

    def _halton_sequence(self, n, dims):
        """Generate Halton sequence points in [0, 1]^dims."""
        points = np.zeros((n, dims))
        for i in range(1, n + 1):
            for j in range(dims):
                base = self._nth_prime(j + 2) # Start base from 3 (skip 2 for first dim? Actually standard is 2,3,5...)
                # Standard Halton uses bases 2, 3, 5...
                # Let's use 2, 3 for 2D.
                base = [2, 3, 5, 7, 11][j % 5]
                f = 1.0
                r = 0.0
                i_val = i
                while i_val > 0:
                    f /= base
                    r += f * (i_val % base)
                    i_val //= base
                points[i-1, j] = r
        return points

    def _nth_prime(self, n):
        """Return the n-th prime number."""
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        if n <= len(primes):
            return primes[n-1]
        # Fallback (should not be reached for small dims)
        candidate = 31
        count = len(primes)
        while count < n:
            is_prime = True
            for p in primes:
                if candidate % p == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.append(candidate)
                count += 1
            candidate += 2
        return primes[n-1]

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ4 specific parameters
        # n_obj = 3, so m = n_obj - 1 = 2 shape parameters
        # K = 100 (standard DTLZ4 exponent)
        K = 100.0
        
        # We have budget evaluations. 
        # We want to maximize hypervolume. 
        # Sampling the Pareto front uniformly in the parameter space (theta1, theta2) 
        # which is the simplex in (n_obj-1) dimensions.
        
        # The Pareto front of DTLZ4 (and DTLZ1/2/3) is:
        # f_i(x) = 1/2 * sum(x_{m+1}^2 ... x_n^2) * (1 + sum(x_1^2 ... x_m^2)) for i=1..m
        # f_{m+1}(x) = 1/2 * sum(x_{m+1}^2 ... x_n^2)
        # Wait, standard DTLZ4 definition:
        # g(x) = 100 * (n_pos - 1 + sum((x_i - 0.5)^2 * sin(20 * pi * (x_i - 0.5))) for i in pos)
        # But for DTLZ4, the Pareto front is the same as DTLZ1/2/3, just harder to reach.
        # The PF is defined by g(x) = 1.
        # f_1 = 0.5 * (1+g) * prod_{i=1}^{m} cos(theta_i)
        # f_2 = 0.5 * (1+g) * sin(theta_1) * prod_{i=2}^{m} cos(theta_i)
        # ...
        # f_m = 0.5 * (1+g) * sin(theta_1) ... sin(theta_{m-1}) * cos(theta_m)
        # f_{m+1} = 0.5 * (1+g) * sin(theta_1) ... sin(theta_m)
        
        # Since we are minimizing, and we can construct points on the PF directly,
        # we can set g(x) = 1.
        # The variables x_1...x_m determine the shape (theta), and x_{m+1}...x_n determine g.
        # To get g=1, we can simply choose x_{m+1}...x_n such that the sum term is 0?
        # Actually, the simplest way to generate a point on the PF is to pick theta_1, theta_2
        # in [0, pi/2] and set the position variables to zero (or specific values) such that g=1.
        
        # However, the problem statement says: "problem(x) clips... and returns 3 objective values."
        # It does NOT say we can bypass the evaluator. We must call problem(x).
        # But we can choose x to lie ON the Pareto front.
        
        # For DTLZ4, the Pareto front is reached when the position variables (x_{m+1}...x_n)
        # are such that g(x) = 1.
        # A simple way to ensure g(x) = 1 is to set all position variables to 0.5?
        # No, g(x) = 100 * (n_pos - 1 + sum(...)).
        # If we set x_i = 0.5 for all position variables, the sine term is 0, and sum is 0.
        # Then g(x) = 100 * (n_pos - 1). This is NOT 1.
        
        # We need to solve for position variables such that g(x) = 1.
        # g(x) = 100 * ( (n - m) - 1 + sum_{i=m+1}^n (x_i - 0.5)^2 * sin(20*pi*(x_i - 0.5)) )
        # Let's simplify. If we set x_i = 0 for all i > m, then (x_i - 0.5)^2 = 0.25, sin(-pi) = 0.
        # So the sum is 0.
        # g(x) = 100 * ( (n - m) - 1 ).
        # For n=12, m=2, n-m=10. g(x) = 100 * 9 = 900.
        # This is far from 1.
        
        # We need to find x_{m+1}...x_n such that g(x) = 1.
        # This is a high-dimensional root finding problem.
        # Alternatively, we can use the fact that the evaluator clips x.
        # Maybe we can just sample the shape parameters (theta) and use a local search
        # to adjust the position variables to hit g=1?
        
        # Strategy:
        # 1. Generate a set of candidate points on the "ideal" front (g=1) by parameterizing
        #    the shape variables (x_1, x_2) to correspond to uniform sampling in objective space.
        # 2. For each shape, we need to find position variables (x_3...x_12) such that g(x)=1.
        # 3. Since we have a budget of 200 evaluations, we cannot do heavy local search for each point.
        # 4. However, we can pre-compute a "template" for the position variables that yields g=1?
        #    No, g depends on the values.
        
        # Let's look at the structure of g(x) for DTLZ4.
        # g(x) = 100 * (n_pos - 1 + sum((x_i - 0.5)^2 * sin(20 * pi * (x_i - 0.5))))
        # If we set x_i = 0.5 + delta_i, the term is delta_i^2 * sin(20*pi*delta_i).
        # For small delta, sin(20*pi*delta) ~ 20*pi*delta. So term ~ 20*pi*delta^3.
        # This is hard to control precisely.
        
        # Alternative: Use the evaluator to refine.
        # We have 200 evals.
        # We can split the budget:
        # - Use ~100 evals to sample the shape space (theta_1, theta_2) using Halton.
        #   For each shape, we start with a guess for position vars.
        # - Use remaining ~100 evals to refine the position vars for the most promising shapes?
        #   This is complex.
        
        # Simpler Approach:
        # The current program (score 1.0) uses a generic EA.
        # The inspirations (score ~0.68) tried analytical construction but failed likely due to
        # not hitting g=1 accurately or clipping issues.
        
        # Let's try a hybrid:
        # 1. Generate M points in the shape space (2D) using Halton. M = budget // 2.
        # 2. For each shape point, we need to find position vars.
        #    We can use a simple iterative method or just pick a fixed "good" position vector
        #    that we know yields g close to 1, and then use the remaining budget to fine-tune
        #    the position variables for the best few points?
        
        # Actually, there is a known trick for DTLZ4.
        # If we set x_{m+1}...x_n to be very small, g(x) is large.
        # If we set them to 0.5, g(x) is large.
        # We need g(x) = 1.
        # Let's assume we can find a single vector P of position variables such that g(P) = 1.
        # Then for any shape S, the point [S, P] is on the Pareto front.
        # Can we find such a P?
        # g(P) = 100 * (9 + sum(...)) = 1 => sum(...) = -0.08.
        # The sum term is sum((x_i - 0.5)^2 * sin(20*pi*(x_i - 0.5))).
        # Let x_i = 0.5 + e_i.
        # Term_i = e_i^2 * sin(20*pi*e_i).
        # If e_i is small, Term_i is small.
        # We need sum(Term_i) = -0.08.
        # We can solve this for a 10-dimensional vector.
        # Since we have 200 evals, we can spend ~100 evals to find ONE good position vector P
        # that minimizes |g(P) - 1|.
        # Then we spend the other ~100 evals to sample the shape space and evaluate [S, P].
        
        # Step 1: Find P such that g(P) ~ 1.
        # Initialize P randomly or with zeros.
        # Use a simple gradient-free method or just random search + local refinement.
        # Since g is smooth, we can use Nelder-Mead or similar if we had more time, but 100 evals is tight for 10 dims.
        # However, we just need g(P) close to 1.
        
        # Let's try:
        # 1. Use 50 evals to find a position vector P with g(P) close to 1.
        #    We can use a simple random search with memory of best.
        # 2. Use remaining 150 evals to sample shape space (theta_1, theta_2) and evaluate [S, P].
        #    The shape variables x_1, x_2 determine the direction.
        #    For DTLZ4, x_1, x_2 are in [0,1].
        #    The mapping to theta is:
        #    x_1 = cos(theta_1)^K
        #    x_2 = sin(theta_1) * cos(theta_2)^K
        #    ...
        #    Wait, the standard mapping is:
        #    x_i = cos(theta_i)^k for i=1..m
        #    x_{m+1} = sin(theta_1) ... sin(theta_m)
        #    No, that's for the position.
        #    The shape is determined by x_1...x_m.
        #    Usually, we sample theta uniformly and compute x.
        #    x_1 = cos(theta_1)^K
        #    x_2 = sin(theta_1) * cos(theta_2)^K
        #    (for m=2, x_2 is the second shape var)
        #    Wait, for m=2:
        #    x_1 = cos(theta_1)^K
        #    x_2 = sin(theta_1) * cos(theta_2)^K  <-- This is not standard.
        #    Standard DTLZ mapping:
        #    x_i = cos(theta_i)^k for i=1..m-1
        #    x_m = sin(theta_{m-1}) * cos(theta_m)^k ? No.
        
        # Let's stick to the objective values directly.
        # If we can ensure g(x)=1, the objectives are:
        # f_1 = 0.5 * cos(theta_1) * cos(theta_2)
        # f_2 = 0.5 * sin(theta_1) * cos(theta_2)
        # f_3 = 0.5 * sin(theta_2)
        # (assuming m=2, n_obj=3)
        
        # So if we pick theta_1, theta_2 uniformly in [0, pi/2], we get uniform coverage of the PF in objective space?
        # Not exactly uniform in HV, but good coverage.
        
        # Plan:
        # 1. Spend first 50 evals to find a position vector P (x_3...x_12) such that g(P) is close to 1.
        #    We will use a simple iterative refinement.
        # 2. Spend remaining 150 evals to sample 150 points in (theta_1, theta_2) space using Halton.
        #    For each, construct x = [x_1(theta), x_2(theta), P].
        #    Evaluate problem(x).
        
        # How to find P?
        # g(P) = 100 * (9 + S(P)) where S(P) = sum_{i=3}^{12} (x_i - 0.5)^2 * sin(20*pi*(x_i - 0.5)).
        # We want g(P) = 1 => S(P) = -0.08.
        # Let's initialize P with x_i = 0.5 + small random.
        # Then adjust.
        
        # Since we don't have the formula for g inside the optimizer (we only have `problem`),
        # we MUST use the evaluator to get the objectives.
        # But the objectives depend on BOTH shape and position.
        # f_1, f_2, f_3 are scaled by (1+g).
        # If g is not 1, the points are not on the PF.
        
        # We can use the evaluator to check "how close" we are to the PF?
        # The sum of objectives for a point on the PF (with g=1) is:
        # f_1 + f_2 + f_3 = 0.5 * (1+1) * (cos1 cos2 + sin1 cos2 + sin2) ... not constant.
        # Actually, for DTLZ4, the distance to the ideal point or sum of objectives is not a simple indicator of g.
        
        # However, we can observe that if g is large, all objectives are large.
        # If we minimize the sum of objectives, we drive g down?
        # f_i ~ (1+g). So minimizing sum(f) minimizes g.
        
        # Refined Plan:
        # 1. Use 50 evals to find a position vector P that minimizes the sum of objectives
        #    when combined with a fixed "average" shape (e.g., theta=pi/4, pi/4).
        #    This will drive g as close to 1 (or 0, but 1 is the boundary) as possible.
        #    Actually, the minimum possible value for g is 0?
        #    g(x) = 100 * (9 + S). S can be negative.
        #    Min S is roughly -0.1?
        #    If g can be 0, then f_i = 0.5 * shape.
        #    If g=1, f_i = shape.
        #    The Pareto front is at g=1?
        #    Standard DTLZ4 PF is g=1.
        #    So we want g=1.
        #    Minimizing sum(f) will drive g to its minimum possible value.
        #    If min g > 1, we are stuck. If min g < 1, we might cross it.
        
        # Let's assume we can get close to g=1.
        # 2. Use remaining 150 evals to sample shapes with this P.
        
        # Implementation details:
        # - Keep track of best P (min sum of objectives).
        # - For shape sampling, use Halton in 2D.
        
        # Let's code this.
        
        n_pos = self.dim - (self.n_obj - 1)
        
        # Phase 1: Find good position vector P
        # Initialize P with random values
        rng = np.random.default_rng(42)
        P = lo[2:] + (hi[2:] - lo[2:]) * rng.random(n_pos)
        
        # Fixed shape for testing P: center of simplex
        theta_test = np.array([np.pi/4, np.pi/4])
        x1 = np.cos(theta_test[0]) ** 100
        x2 = np.sin(theta_test[0]) * (np.cos(theta_test[1]) ** 100)
        x_shape = np.array([x1, x2])
        
        best_sum = np.inf
        best_P = P.copy()
        
        # Simple random search for P
        for i in range(50):
            x = np.concatenate([x_shape, P])
            f = problem(x)
            s = np.sum(f)
            if s < best_sum:
                best_sum = s
                best_P = P.copy()
            
            # Mutate P
            P = best_P + 0.1 * (hi[2:] - lo[2:]) * rng.standard_normal(n_pos)
            P = np.clip(P, lo[2:], hi[2:])
            
            if time.time() > self.deadline:
                break
        
        # Phase 2: Sample shapes with best_P
        n_samples = self.budget - 50
        if n_samples <= 0:
            n_samples = 10
            # Re-evaluate best_P with a few shapes if budget exceeded? 
            # No, just return what we have.
            
        # Generate Halton points in 2D
        halton_pts = self._halton_sequence(n_samples, 2)
        
        # Map to theta in [0, pi/2]
        theta1 = halton_pts[:, 0] * np.pi / 2
        theta2 = halton_pts[:, 1] * np.pi / 2
        
        # Construct x vectors
        # x_1 = cos(theta_1)^100
        # x_2 = sin(theta_1) * cos(theta_2)^100
        # Note: For DTLZ4, the mapping is x_i = cos(theta_i)^k for i < m, and x_m involves sin.
        # Standard DTLZ4 mapping for m=2:
        # x_1 = cos(theta_1)^100
        # x_2 = sin(theta_1) * cos(theta_2)^100
        # This ensures x_1, x_2 are in [0,1] and cover the front.
        
        x1 = np.cos(theta1) ** 100
        x2 = np.sin(theta1) * (np.cos(theta2) ** 100)
        
        # Evaluate
        X = []
        for i in range(n_samples):
            x = np.concatenate([np.array([x1[i], x2[i]]), best_P])
            # Ensure bounds
            x = np.clip(x, lo, hi)
            f = problem(x)
            X.append(x)
            
            if time.time() > self.deadline:
                break
        
        return X

# EVOLVE-BLOCK-END
