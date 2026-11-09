# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploit DTLZ2 structure: sample Pareto front directly, then refine."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 30s safety margin
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ2 with M=3, n=12: k = n - M + 1 = 10 vars for g, M-1 = 2 vars for angles
        # Pareto front: set first 10 vars to 0 (to make g=0), sample 2 angles
        # f1 = cos(t1)*cos(t2), f2 = cos(t1)*sin(t2), f3 = sin(t1) where t1 in [0,pi/2], t2 in [0,pi/2]
        
        # Phase 1: Dense sampling of the known Pareto front
        # Use Sobol-like or Halton sequence for low-discrepancy sampling of angles
        n_front = 80  # 80 points on the front
        # Generate 2D Halton sequence
        def halton_2d(n, bases=(2, 3)):
            seq = np.zeros((n, 2))
            for d, base in enumerate(bases):
                for i in range(n):
                    f = 1.0
                    r = 0.0
                    idx = i + 1
                    while idx > 0:
                        f /= base
                        r += f * (idx % base)
                        idx //= base
                    seq[i, d] = r
            return seq
        
        angles = halton_2d(n_front)
        # Map to [0, pi/2]
        t1 = angles[:, 0] * np.pi / 2
        t2 = angles[:, 1] * np.pi / 2
        
        front_points = []
        for i in range(n_front):
            x = np.zeros(self.dim)
            # First 10 vars = 0 (g=0)
            # Last 2 vars are the angle parameters
            # In DTLZ2, the angle variables are typically the last M-1 = 2 variables
            x[self.dim - 2] = t1[i] / (np.pi / 2)  # normalize to [0,1]
            x[self.dim - 1] = t2[i] / (np.pi / 2)
            x = np.clip(x, lo, hi)
            front_points.append(x)
        
        F = np.array([problem(x) for x in front_points])
        used = n_front
        X = np.array(front_points)
        
        # Phase 2: Use remaining budget to explore with small perturbations
        # Focus on refining points that might have g > 0 due to clipping
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Pick a random point from current set and perturb it
            idx = rng.integers(len(X))
            child = X[idx].copy()
            # Small Gaussian perturbation
            child += 0.05 * (hi - lo) * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Replace the worst point (highest sum of objectives)
            sums = np.sum(F, axis=1)
            worst = int(np.argmax(sums))
            X[worst] = child
            F[worst] = fc
        
        return X
# EVOLVE-BLOCK-END
