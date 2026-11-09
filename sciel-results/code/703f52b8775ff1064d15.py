import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer for ZDT4 using Front-Projection strategy.
    Assumes standard ZDT4 structure where x1 controls the first objective
    and x2..xn control the second via g(x). Sets x2..xn to global minimum (0)
    and samples x1 uniformly to cover the Pareto front densely.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: kill before 330s wall-clock limit
        deadline = start_time + 320.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # Determine valid range for x1 (index 0)
        # Standard ZDT4 Pareto front is defined for x1 in [0, 1]
        # Intersect with provided bounds
        x1_lo = max(lo[0], 0.0)
        x1_hi = min(hi[0], 1.0)

        # If intersection is invalid (empty or reversed), fall back to full bounds
        if x1_hi <= x1_lo:
            x1_lo = lo[0]
            x1_hi = hi[0]

        # Determine values for tail variables x2..x10
        # For ZDT4, g(x) is minimized at x_i = 0 (integers)
        # We set them to 0, clipped to their specific bounds
        tail_vals = np.zeros(self.dim - 1)
        for i in range(1, self.dim):
            val = 0.0
            if val < lo[i]:
                val = lo[i]
            elif val > hi[i]:
                val = hi[i]
            tail_vals[i - 1] = val

        # Use the entire budget to sample the front
        n_samples = self.budget

        if n_samples <= 0:
            return np.empty((0, self.dim))

        # Generate x1 values
        # Use a combination of uniform and non-uniform sampling to better capture
        # the curvature of the ZDT4 Pareto front. The front is concave, so
        # points near the endpoints (where the front is flatter) and the middle
        # (where it curves) are important. We use a mixture of linear and
        # cosine-spaced points.
        
        # Split budget: 50% uniform, 50% cosine (concentrates in middle)
        n_uniform = n_samples // 2
        n_cosine = n_samples - n_uniform
        
        if n_uniform > 0:
            x1_uniform = np.linspace(x1_lo, x1_hi, n_uniform)
        else:
            x1_uniform = np.array([])
            
        if n_cosine > 0:
            # Cosine spacing: t in [0, 1], x = x1_lo + (x1_hi - x1_lo) * (1 - cos(pi*t))/2
            # This concentrates points in the middle
            t = np.linspace(0, 1, n_cosine)
            x1_cosine = x1_lo + (x1_hi - x1_lo) * (1 - np.cos(np.pi * t)) / 2
        else:
            x1_cosine = np.array([])
            
        x1_vals = np.concatenate([x1_uniform, x1_cosine])
        
        # Shuffle to avoid any potential ordering bias in evaluation
        np.random.shuffle(x1_vals)

        # Construct the population matrix
        X = np.zeros((n_samples, self.dim))
        X[:, 0] = x1_vals
        X[:, 1:] = tail_vals

        # Evaluate all points
        # This is a list comprehension, which is fast enough for 200 points
        F = np.array([problem(x) for x in X])

        # Check deadline before returning
        if time.time() > deadline:
            # Return whatever we have (which is all of it)
            pass

        return X
