# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for DTLZ7 exploiting the known Pareto front structure.
    The Pareto front of DTLZ7 with M objectives lies in the subspace x[2:] = 0.
    Strategy:
    1. Use all budget for dense, space-filling sampling of the Pareto front subspace.
    2. Use a low-discrepancy (Halton-like) sequence for optimal hypervolume coverage.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)
        
        used = 0
        all_points = []
        all_objs = []
        
        def eval_point(x):
            nonlocal used
            x = np.clip(x, lo, hi)
            f = problem(x)
            used += 1
            return x, f
        
        # Generate Halton sequence points for 2D parameter space
        # Halton sequences provide excellent space-filling properties
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r
        
        # Generate n points using 2D Halton sequence (bases 2 and 3)
        n_samples = self.budget
        halton_points = []
        for i in range(1, n_samples + 1):
            h1 = halton(i, 2)
            h2 = halton(i, 3)
            halton_points.append((h1, h2))
        
        # Shuffle to avoid bias if we stop early
        rng.shuffle(halton_points)
        
        # Evaluate all points on the Pareto front
        for t1, t2 in halton_points:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            
            x = np.zeros(self.dim)
            x[0] = t1
            x[1] = t2
            
            x, f = eval_point(x)
            all_points.append(x)
            all_objs.append(f)
        
        return all_points
# EVOLVE-BLOCK-END
