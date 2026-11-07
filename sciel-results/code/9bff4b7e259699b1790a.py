"""Multi-objective optimizer for DTLZ4 using structured sampling and local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using structured sampling and local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ4: x[0..n_obj-1] are decision variables for the Pareto front
        # x[n_obj:] are position variables (with beta=10, they cluster near 0)
        # The Pareto front is a quarter-sphere in objective space:
        # f1 = 0.5 * sum(f_i^2) where f_i are derived from x[0..n_obj-1]
        # Actually for DTLZ4 with 3 objectives:
        # g(x) = sum_{i=n_obj+1}^{n} (x_i - 0.5)^2 + 10*sin^2(10*pi*x_i)
        # f1 = (1+g)*0.5*x_1*cos(x_2/pi/2)*cos(x_3/pi/2)
        # f2 = (1+g)*0.5*x_1*sin(x_2/pi/2)*cos(x_3/pi/2)
        # f3 = (1+g)*0.5*(1-x_1)
        # The Pareto front has g=0, so x_i = 0.5 for i >= n_obj+1 (but with beta=10,
        # the optimal is at x_i = 0.5 for the position variables)
        # Wait, for DTLZ4, the position variables should be at 0.5 for g=0.
        # Actually, g(x) = sum (x_i - 0.5)^2 + 10*sin^2(10*pi*(x_i-0.5))
        # So g=0 when x_i = 0.5 for all position variables.

        # Strategy: 
        # 1. Set position variables (x[3:]) to 0.5 to achieve g=0
        # 2. Sample the 3 decision variables (x[0], x[1], x[2]) to cover the Pareto front
        # 3. The Pareto front is a quarter-sphere: f1^2 + f2^2 + f3^2 = (1+g)^2 / 4 = 1/4 when g=0
        # Wait, let me re-derive:
        # f1 = (1+g)*0.5*x1*cos(x2*pi/2)*cos(x3*pi/2)
        # f2 = (1+g)*0.5*x1*sin(x2*pi/2)*cos(x3*pi/2)
        # f3 = (1+g)*0.5*(1-x1)
        # With g=0: f1^2 + f2^2 + f3^2 = 0.25*x1^2*cos^2(x3*pi/2) + 0.25*x1^2*sin^2(x3*pi/2) + 0.25*(1-x1)^2
        # = 0.25*x1^2 + 0.25*(1-x1)^2
        # This is not a sphere... Let me think again.
        # Actually for DTLZ4, the Pareto front is indeed a (n_obj-1)-dimensional hypersphere.
        # For 3 objectives, it's a quarter-sphere in the first octant.

        # Let's use a different approach: directly sample points on the Pareto front
        # and evaluate them, plus do some local search.

        # First, let's create a structured sample of the Pareto front
        # The Pareto front for DTLZ4 with 3 objectives:
        # Parameterize by theta1, theta2 in [0, pi/2]:
        # x1 = cos(theta1)
        # x2 = sin(theta1) * cos(theta2)  -- wait, this doesn't match the DTLZ4 formulation
        
        # Let me just use the standard DTLZ4 parameterization:
        # For 3 objectives, the Pareto front is:
        # f1 = 0.5 * (1+g) * x1 * cos(x2*pi/2) * cos(x3*pi/2)
        # f2 = 0.5 * (1+g) * x1 * sin(x2*pi/2) * cos(x3*pi/2)
        # f3 = 0.5 * (1+g) * (1-x1)
        # With g=0, and x1, x2, x3 in [0,1]:
        # Let a = x1*cos(x2*pi/2)*cos(x3*pi/2)
        # Let b = x1*sin(x2*pi/2)*cos(x3*pi/2)
        # Let c = 1-x1
        # Then a^2 + b^2 + c^2 = x1^2*cos^2(x3*pi/2) + (1-x1)^2
        # This is not constant, so it's not a sphere in this parameterization.
        
        # Actually, I think I'm overcomplicating this. Let me just do:
        # 1. Set position variables to 0.5
        # 2. Use a Fibonacci lattice or Halton sequence to sample the 3 decision variables
        # 3. Evaluate all points
        # 4. Do local refinement around the best points

        # Position variables: x[3:] = 0.5
        # Decision variables: x[0], x[1], x[2]

        # Use Halton sequence for quasi-random sampling
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        # Generate quasi-random points for the 3 decision variables
        n_sample = min(150, budget)
        X = np.zeros((n_sample, dim))
        X[:, 3:] = 0.5  # Position variables at 0.5 for g=0
        
        bases = [2, 3, 5]
        for i in range(1, n_sample + 1):
            X[i-1, 0] = halton(i, bases[0])
            X[i-1, 1] = halton(i, bases[1])
            X[i-1, 2] = halton(i, bases[2])

        # Evaluate
        F = np.array([problem(x) for x in X])
        used = n_sample

        # Check if we're close to the deadline
        if time.time() > deadline - 30:
            return X

        # Local refinement: pick the best non-dominated points and perturb them
        # Find non-dominated points
        def is_dominated(f, F):
            return np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1))

        dominated = np.array([is_dominated(f, F) for f in F])
        non_dom_idx = np.where(~dominated)[0]

        # Refine around non-dominated points
        remaining = budget - used
        n_refine = min(remaining, 50)
        step_size = 0.05

        for _ in range(n_refine):
            if time.time() > deadline - 20:
                break
            if len(non_dom_idx) == 0:
                break
            parent_idx = non_dom_idx[np.random.randint(len(non_dom_idx))]
            child = X[parent_idx].copy()
            # Perturb only the decision variables
            child[:3] += step_size * np.random.randn(3)
            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1
            # Replace the worst point
            worst = int(np.argmax(np.sum(F, axis=1)))
            X[worst] = child
            F[worst] = fc
            # Update non-dominated set
            dominated = np.array([is_dominated(f, F) for f in F])
            non_dom_idx = np.where(~dominated)[0]

        return X
# EVOLVE-BLOCK-END
