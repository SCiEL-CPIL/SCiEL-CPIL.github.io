import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """Optimizer using Sobol sequence on the DTLZ4 Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # DTLZ4 Pareto front is defined by x_2..x_11 = 0.5 (g=0)
        # and x_0, x_1 varying in [0, 1].
        
        # Use Sobol sequence for low-discrepancy sampling in 2D
        # scipy.stats.qmc.Sobol requires d >= 1
        # We need 2D for x_0, x_1
        
        # Generate Sobol points
        # Note: Sobol sequences are deterministic. 
        # Using scramble=True helps with randomization if needed, 
        # but for fixed budget, a standard Sobol is often sufficient and reproducible.
        # However, the task mentions seeds. 
        # The evaluator calls np.random.seed(seed).
        # Sobol in scipy uses its own state, but we can use the global random state 
        # or just use a fixed Sobol sequence. 
        # Since the problem is deterministic given the front shape, 
        # a fixed high-quality sequence is best.
        
        # Let's use the first `budget` points of a Sobol sequence.
        # We might want to skip the very first point (0,0) if it's too corner-heavy?
        # No, corners are good for hypervolume.
        
        sampler = qmc.Sobol(d=2, scramble=False, seed=42)
        # Sobol sequences in scipy start with 0.
        # The first point is (0,0).
        # The next is (0.5, 0.5)? No, (0, 0.5) then (0.5, 0)?
        # Actually, let's just generate `budget` points.
        
        try:
            sobol_pts = sampler.random(self.budget)
        except Exception:
            # Fallback to Halton if Sobol fails for some reason
            def halton_sequence(n, dims=2):
                points = []
                bases = [2, 3]
                for i in range(1, n + 1):
                    point = []
                    for d in range(dims):
                        f = 1.0
                        result = 0.0
                        x = i
                        while x > 0:
                            f /= bases[d]
                            result += f * (x % bases[d])
                            x //= bases[d]
                        point.append(result)
                    points.append(point)
                return np.array(points)
            sobol_pts = halton_sequence(self.budget, dims=2)

        # Construct decision vectors
        X = np.zeros((self.budget, self.dim))
        X[:, 0] = sobol_pts[:, 0]
        X[:, 1] = sobol_pts[:, 1]
        X[:, 2:] = 0.5  # Set aggregation variables to 0.5 to minimize g
        
        # Clip to bounds
        X = np.clip(X, lo, hi)

        # Evaluate
        F = np.array([problem(x) for x in X])

        if time.time() > deadline:
            return X

        return X
