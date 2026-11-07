"""Multi-objective optimizer for ZDT4 using a structured decomposition approach."""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj

        # ZDT4 structure: x0 in [0,1] determines the Pareto front position.
        # x1..x9 are penalized to be near 0 (or 5 for some variants, but standard ZDT4 penalizes deviation from 0).
        # Actually, ZDT4: g(x) = 1 + 10*(n-1) + sum(xi^2 - 10*cos(4*pi*xi)) for i=2..n
        # The standard ZDT4 has x1 in [0,1] and x2..xn in [-5, 5].
        # The Pareto front is f1 = x1, f2 = g(f1) = 1 + 10*(n-1) + sum(xi^2 - 10*cos(4*pi*xi))
        # Wait, standard ZDT4:
        # f1(x) = x1
        # g(x) = 1 + 10*(n-1) + sum_{i=2}^{n} (xi^2 - 10*cos(4*pi*xi))
        # f2(x) = g(x) * (1 - sqrt(f1(x)/g(x)))
        # The global optimum for g(x) is 0 when xi=0 for all i>=2.
        # So the Pareto front is f2 = 1 - sqrt(f1) for f1 in [0,1].
        # However, the problem says "bounds given per variable".
        # Let's assume standard ZDT4 bounds: x0 in [0,1], x1..x9 in [-5,5].
        # But the prompt says "bounds is a list of (low, high) pairs".
        # We should use the provided bounds.

        # Strategy:
        # 1. Since ZDT4 is separable in the sense that x0 controls the front position and the rest control the distance from the front,
        #    we can optimize the rest of the variables to minimize g(x) while sweeping x0.
        # 2. For a fixed x0, we need to minimize g(x) = sum_{i=1}^{dim-1} (xi^2 - 10*cos(4*pi*xi)).
        #    The minimum of g(x) is 0, achieved when xi=0 for all i>=1.
        #    However, due to the cosine term, there are many local minima.
        #    A simple approach: for each x0, set x1..x9 to 0 (or close to 0) to get close to the Pareto front.
        #    But we have a limited budget. We can use a hybrid approach:
        #    - Use a portion of the budget to explore x0 uniformly.
        #    - Use the remaining budget to refine the other variables using a local search or gradient-like method.
        #    
        #    Actually, since g(x) is separable, we can minimize each xi independently for a fixed x0.
        #    The minimum of xi^2 - 10*cos(4*pi*xi) is -10, achieved at xi=0, ±0.5, ±1, etc.
        #    Wait, the function is xi^2 - 10*cos(4*pi*xi).
        #    At xi=0, value is -10.
        #    At xi=0.5, value is 0.25 - 10*cos(2*pi) = 0.25 - 10 = -9.75.
        #    At xi=1, value is 1 - 10*cos(4*pi) = 1 - 10 = -9.
        #    So the global minimum for each term is -10 at xi=0.
        #    Therefore, the global minimum for g(x) is 1 + 10*(n-1) + (n-1)*(-10) = 1.
        #    Wait, standard ZDT4 g(x) is defined as 1 + 10*(n-1) + sum(...).
        #    If sum is (n-1)*(-10), then g(x) = 1 + 10*(n-1) - 10*(n-1) = 1.
        #    So the Pareto front is f2 = 1 * (1 - sqrt(f1/1)) = 1 - sqrt(f1).
        #    
        #    So the strategy is:
        #    1. Sample x0 uniformly in [0,1] (or the given bounds for x0).
        #    2. For each x0, set x1..x9 to 0 (or the value that minimizes the term).
        #    3. Evaluate and record.
        #    4. Use remaining budget to refine x1..x9 around 0 to ensure we are at the minimum.
        #    
        #    But we need to be careful: the bounds might not include 0.
        #    If 0 is not in the bounds, we should set xi to the closest bound.
        #    
        #    Let's implement this:
        #    - Determine the "optimal" value for each xi (i>=1) that minimizes xi^2 - 10*cos(4*pi*xi) within the bounds.
        #      This is likely 0 if 0 is in the bounds, otherwise the closest bound.
        #    - Sample x0 from a set of points that cover the Pareto front well.
        #    - For each x0, evaluate the point with xi set to the optimal values.
        #    - Use remaining budget to do a local search around these points.

        # Step 1: Determine optimal values for x1..x9
        opt_vals = []
        for i in range(1, dim):
            low_i, high_i = bounds[i]
            # The function h(x) = x^2 - 10*cos(4*pi*x) has minima at x = k/2 for integer k.
            # The global minimum is at x=0.
            # If 0 is in [low_i, high_i], use 0.
            # Otherwise, use the closest bound.
            if low_i <= 0 <= high_i:
                opt_vals.append(0.0)
            else:
                opt_vals.append(low_i if abs(low_i) < abs(high_i) else high_i)
        opt_vals = np.array(opt_vals)

        # Step 2: Sample x0
        # We want to cover the Pareto front well. The front is f2 = 1 - sqrt(f1).
        # To maximize hypervolume, we should sample x0 such that the points are spread out.
        # A uniform sample in x0 is a good start.
        # Let's use a portion of the budget for initial sampling.
        n_samples = min(self.budget // 2, 100)  # Up to 100 samples for x0
        x0_samples = np.linspace(lo[0], hi[0], n_samples)

        # Step 3: Evaluate initial points
        points = []
        values = []
        for x0 in x0_samples:
            x = np.zeros(dim)
            x[0] = x0
            x[1:] = opt_vals
            x = np.clip(x, lo, hi)
            f = problem(x)
            points.append(x)
            values.append(f)

        points = np.array(points)
        values = np.array(values)
        used = len(points)

        # Step 4: Use remaining budget for local search
        # For each point, try to improve the other variables by small perturbations.
        # Since we have a limited budget, we can do a simple random search around the current points.
        rng = np.random.default_rng(0)
        while used < self.budget:
            if time.time() > deadline:
                break
            # Pick a random point to refine
            idx = rng.integers(len(points))
            x = points[idx].copy()
            # Perturb the variables other than x0
            mask = np.ones(dim, dtype=bool)
            mask[0] = False  # Don't perturb x0
            perturb = rng.standard_normal(dim) * 0.1 * (hi - lo)
            perturb[0] = 0
            x_new = x + perturb
            x_new = np.clip(x_new, lo, hi)
            f_new = problem(x_new)
            used += 1
            # If the new point is better (in terms of g, i.e., closer to the front), replace
            # We can use a simple dominance check or just replace if f2 is smaller
            if f_new[1] < values[idx][1]:
                points[idx] = x_new
                values[idx] = f_new

        return points
