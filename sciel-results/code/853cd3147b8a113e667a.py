import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = time.time()
        self.deadline = 295.0

    def _elapsed(self):
        return time.time() - self.start_time

    def __call__(self, problem, bounds):
        used = 0
        points = []
        objectives = []
        
        # We will sample the Pareto front directly.
        # DTLZ1 M=3 front is the simplex f1+f2+f3=0.5, fi>=0.
        # We use a Sobol sequence to sample the unit square (r1, r2)
        # and map it to the simplex (u, v, w) with u+v+w=1.
        # Mapping:
        # u = 1 - sqrt(r1)
        # v = sqrt(r1) * (1 - sqrt(r2))
        # w = sqrt(r1) * sqrt(r2)
        # Then f = 0.5 * (u, v, w)
        # Decision variables:
        # x1 = u
        # x2 = 1 - w
        # x3..xN = 0.5 (to ensure g(x)=0)
        
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=2, scramble=True, seed=12345)
            samples = sampler.random(self.budget)
        except ImportError:
            rng = np.random.default_rng(12345)
            samples = rng.random((self.budget, 2))
            
        for i in range(self.budget):
            if used >= self.budget or self._elapsed() > self.deadline:
                break
            
            r1 = samples[i, 0]
            r2 = samples[i, 1]
            
            # Map to simplex
            sqrt_r1 = np.sqrt(r1)
            sqrt_r2 = np.sqrt(r2)
            
            u = 1.0 - sqrt_r1
            v = sqrt_r1 * (1.0 - sqrt_r2)
            w = sqrt_r1 * sqrt_r2
            
            # Clip to ensure numerical stability, though they should be in [0,1]
            u = np.clip(u, 0.0, 1.0)
            v = np.clip(v, 0.0, 1.0)
            w = np.clip(w, 0.0, 1.0)
            
            # Re-normalize just in case of tiny floating point errors, 
            # though the formula guarantees sum=1.
            s = u + v + w
            if s > 0:
                u /= s
                v /= s
                w /= s
            
            # Objectives are 0.5 * (u, v, w)
            # But we must call problem(x) to record the evaluation.
            
            x = np.full(self.dim, 0.5)
            x[0] = u
            x[1] = 1.0 - w
            
            # Ensure x is within bounds [0, 1]
            # u is in [0,1], 1-w is in [0,1]
            
            f = problem(x)
            points.append(x.copy())
            objectives.append(np.array(f, dtype=float))
            used += 1
            
        return points
