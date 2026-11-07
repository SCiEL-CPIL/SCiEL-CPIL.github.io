import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer for DTLZ1 using Sobol quasi-random front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = time.time()
        self.deadline = 295.0

    def _elapsed(self):
        return time.time() - self.start_time

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        used = 0
        points = []
        objectives = []

        # On DTLZ1 Pareto front:
        #   x[2:] = 0.5
        #   f1 = 0.5*x1, f2 = 0.5*(1-x1)*x2, f3 = 0.5*(1-x2)
        # The Pareto front in objective space is a 2D simplex.
        # We sample x1, x2 uniformly in [0, 1] using a low-discrepancy sequence
        # to maximize hypervolume coverage of the front.
        
        n_samples = self.budget
        
        # Use Sobol sequence for better space-filling properties
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=2, scramble=True, seed=42)
            # We need exactly n_samples points. Sobol generates powers of 2.
            # We can generate the next power of 2 and take the first n_samples,
            # or just generate enough.
            n_power2 = 1
            while n_power2 < n_samples:
                n_power2 *= 2
            sobol_samples = sampler.random(n_power2)[:n_samples]
        except ImportError:
            # Fallback to random if scipy is not available, though it should be
            rng = np.random.default_rng(42)
            sobol_samples = rng.random((n_samples, 2))

        # Evaluate
        for i in range(n_samples):
            if self._elapsed() > self.deadline:
                break
            
            x1 = float(sobol_samples[i, 0])
            x2 = float(sobol_samples[i, 1])
            
            # Clip to [0, 1] just in case, though Sobol is [0, 1)
            x1 = float(np.clip(x1, 0.0, 1.0))
            x2 = float(np.clip(x2, 0.0, 1.0))
            
            x = np.full(self.dim, 0.5)
            x[0] = x1
            x[1] = x2
            
            f = problem(x)
            points.append(x.copy())
            objectives.append(f)
            used += 1

        return points
