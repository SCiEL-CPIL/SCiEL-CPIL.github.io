"""Hypervolume-driven adaptive refinement for DTLZ1."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Adaptive refinement strategy for multi-objective optimization on DTLZ1."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref_point = np.array([400.0, 400.0, 400.0])
        
        # Initialize with a small diverse set
        init_size = min(10, self.budget // 20)
        X = []
        F = []
        
        for _ in range(init_size):
            x = lo + (hi - lo) * rng.random(self.dim)
            X.append(x)
            F.append(problem(x))
        
        X = np.array(X)
        F = np.array(F)
        used = init_size
        
        # Adaptive refinement
        while used < self.budget:
            # Generate a few candidates from existing points
            candidates = []
            for i in range(min(len(X), 5)):
                for step in [0.05, 0.1, 0.2, 0.3]:
                    c = X[i] + step * (hi - lo) * rng.standard_normal(self.dim)
                    c = np.clip(c, lo, hi)
                    candidates.append(c)
            
            # Add a random candidate
            c = lo + (hi - lo) * rng.random(self.dim)
            candidates.append(c)
            
            # Evaluate candidates one by one, pick the best
            best_c = None
            best_f = None
            best_score = -np.inf
            
            for c in candidates:
                if used >= self.budget:
                    break
                fc = problem(c)
                used += 1
                
                # Score: volume of the box if dominated by ref, times diversity
                if np.all(fc < ref_point):
                    vol = np.prod(ref_point - fc)
                else:
                    vol = 0.0
                
                # Diversity: minimum distance to existing points in objective space
                if len(F) > 0:
                    min_dist = np.min(np.linalg.norm(F - fc, axis=1))
                else:
                    min_dist = 1.0
                
                score = vol * (1.0 + min_dist / 100.0)
                
                if score > best_score:
                    best_score = score
                    best_c = c
                    best_f = fc
            
            if best_c is not None:
                X = np.vstack([X, best_c])
                F = np.vstack([F, best_f])
                
                # Prune dominated points
                n = len(F)
                if n > 30:
                    dominated = np.zeros(n, dtype=bool)
                    for i in range(n):
                        for j in range(n):
                            if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                                dominated[i] = True
                                break
                    X = X[~dominated]
                    F = F[~dominated]
        
        return X
# EVOLVE-BLOCK-END
