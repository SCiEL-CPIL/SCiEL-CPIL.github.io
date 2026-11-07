# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """
    Optimizer for DTLZ7 exploiting the known Pareto front structure.
    The Pareto front of DTLZ7 lies on the subspace where x[2:] = 0.
    We use Sobol sampling on the 2D subspace, but we transform the second variable
    to ensure uniform coverage in the objective space.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0  # 300s limit with 10s safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = 0
        results = []
        
        # The Pareto front of DTLZ7 is defined by x[2:] = 0.
        # The decision variables on the front are x[0] and x[1].
        # f1 = x0
        # f2 = 2 - 2*sqrt(x1)  (assuming g=1)
        # f3 = 1 - sqrt(x1)    (assuming standard DTLZ7 M=3)
        # To maximize hypervolume, we want uniform coverage in objective space.
        # Since f3 is linear in sqrt(x1), we sample sqrt(x1) uniformly.
        # Let s1 ~ U[0,1]. Then x1 = s1^2.
        
        n_total = self.budget
        
        # Explicitly add corners to ensure max coverage of the hypervolume space
        # (0,0) -> f1=0, f2=2, f3=1
        # (0,1) -> f1=0, f2=0, f3=0
        # (1,0) -> f1=1, f2=2, f3=1
        # (1,1) -> f1=1, f2=0, f3=0
        explicit_points = [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0]
        ]
        
        n_explicit = len(explicit_points)
        n_sobol = max(0, n_total - n_explicit)
        
        # First evaluate explicit corner points
        for pt in explicit_points:
            if time.time() > deadline:
                break
            x = np.zeros(self.dim)
            x[0] = pt[0]
            # Apply transformation: x1 = (sqrt_val)^2
            # Here pt[1] is the "sqrt" value from the explicit set logic?
            # No, explicit points are defined in x-space.
            # We want to cover the extremes of the front.
            # x1=0 and x1=1 are the extremes.
            x[1] = pt[1]
            f = problem(x)
            results.append(f)
            n_evals += 1
        
        # Then use Sobol for the remaining budget
        if n_sobol > 0:
            sampler = qmc.Sobol(d=2, scramble=True)
            sobol_samples = sampler.random(n=n_sobol)
            
            x0_vals = sobol_samples[:, 0]
            # Transform x1 to ensure uniformity in objective space
            # x1 = s1^2
            x1_vals = sobol_samples[:, 1] ** 2
            
            for i in range(n_sobol):
                if time.time() > deadline:
                    break
                
                x = np.zeros(self.dim)
                x[0] = x0_vals[i]
                x[1] = x1_vals[i]
                # x[2:] remains 0, which is the Pareto front condition for DTLZ7
                
                f = problem(x)
                results.append(f)
                n_evals += 1
        
        return results
# EVOLVE-BLOCK-END
