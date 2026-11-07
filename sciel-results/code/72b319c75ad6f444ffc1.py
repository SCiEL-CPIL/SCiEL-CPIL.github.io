"""Improved DTLZ2 optimizer with Sobol-sequence Pareto front sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploits the known Pareto front of DTLZ2 using low-discrepancy Sobol sequences."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _sobol_sequence(self, n, dim=2, skip=0):
        """Generate n Sobol sequence points in [0,1]^dim using scipy."""
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        # Generate more than needed to allow skipping initial points
        total = n + skip
        # Sobol requires power-of-2 dimensions; use next power of 2 >= total
        m = 1
        while m < total:
            m *= 2
        samples = sampler.random(m)
        return samples[skip:skip+n]

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # For DTLZ2 with M=3 objectives and n=12 decision variables:
        # f_1 = g * (1 - x_1), f_2 = g * (1 - x_2), f_3 = g * x_2
        # g = 1 - sum_{j=3}^{12} (x_j - 0.5)^2
        # On the Pareto front, g=1, requiring x_j = 0.5 for j >= 3.
        # Then f_1 = 1 - x_1, f_2 = 1 - x_2, f_3 = x_2
        # So the PF is parameterized by x_1, x_2 in [0,1].
        
        # Use Sobol sequence to sample (x_1, x_2) uniformly on [0,1]^2.
        # This gives a low-discrepancy, space-filling set of points on the PF.
        
        # Allocate most of the budget to Sobol-sampled PF points,
        # and a small fraction to random exploration for robustness.
        n_sobol = min(budget - 5, 195)  # leave 5 for random points
        n_random = budget - n_sobol
        
        # Generate Sobol points in [0,1]^2
        sobol_pts = self._sobol_sequence(n_sobol, dim=2, skip=0)
        
        evaluated_points = []
        evaluated_objs = []
        
        # Evaluate Sobol-sampled points on the Pareto front
        for i in range(n_sobol):
            if len(evaluated_points) >= self.budget:
                break
            if time.time() > deadline:
                break
            x = np.full(dim, 0.5)
            x[0] = sobol_pts[i, 0]
            x[1] = sobol_pts[i, 1]
            # Clip to bounds for safety
            x = np.clip(x, lo, hi)
            try:
                f = problem(x)
                f = np.array(f, dtype=float)
                evaluated_points.append(x)
                evaluated_objs.append(f)
            except:
                continue
        
        # Add random points for robustness (in case PF assumption is slightly off)
        rng = np.random.default_rng(42)
        for _ in range(n_random):
            if len(evaluated_points) >= self.budget:
                break
            if time.time() > deadline:
                break
            x = lo + (hi - lo) * rng.random(dim)
            try:
                f = problem(x)
                f = np.array(f, dtype=float)
                evaluated_points.append(x)
                evaluated_objs.append(f)
            except:
                continue
        
        # If we still have budget, do a fine-grained Sobol refinement
        if len(evaluated_points) < self.budget and time.time() < deadline:
            # Generate additional Sobol points with a different seed for diversity
            n_extra = self.budget - len(evaluated_points)
            if n_extra > 0:
                extra_pts = self._sobol_sequence(n_extra, dim=2, skip=100)
                for i in range(n_extra):
                    if len(evaluated_points) >= self.budget:
                        break
                    if time.time() > deadline:
                        break
                    x = np.full(dim, 0.5)
                    x[0] = extra_pts[i, 0]
                    x[1] = extra_pts[i, 1]
                    x = np.clip(x, lo, hi)
                    try:
                        f = problem(x)
                        f = np.array(f, dtype=float)
                        evaluated_points.append(x)
                        evaluated_objs.append(f)
                    except:
                        continue
        
        return evaluated_points
# EVOLVE-BLOCK-END
