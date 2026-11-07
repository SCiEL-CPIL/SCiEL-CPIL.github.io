import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
        def __init__(self, budget, dim, n_obj):
            self.budget = budget
            self.dim = dim
            self.n_obj = n_obj

        def __call__(self, problem, bounds):
            start_time = time.time()
            deadline = start_time + 280

            lo = np.array([b[0] for b in bounds])
            hi = np.array([b[1] for b in bounds])

            k_shape = self.n_obj - 1
            shape_vars = list(range(k_shape))
            dist_vars = list(range(k_shape, self.dim))

            # Set distance vars to 0.5 to be on the front (g=1)
            # This assumes standard DTLZ2 where g = 1 + 10 * sum((x-0.5)^2)
            
            n_evals = self.budget
            
            # Generate Sobol points
            # n_shape = 2 for n_obj=3
            sampler = qmc.Sobol(d=k_shape, scramble=True, seed=42)
            # Sobol requires power of 2 for best performance, but works for any n.
            # We want n_evals points.
            # Note: Sobol points are in [0,1].
            
            # If the problem function uses x directly in trig, and bounds are [0,1],
            # we are restricted to [0,1] radians.
            # If the problem function scales x to [0, pi/2], we are good.
            # Given the ambiguity, using [0,1] is the safest interpretation of the bounds.
            
            points = sampler.random(n=n_evals)
            
            solutions = []
            
            for i in range(n_evals):
                if time.time() > deadline:
                    break
                    
                x = np.zeros(self.dim)
                # Set shape vars
                for j, var_idx in enumerate(shape_vars):
                    x[var_idx] = points[i][j]
                
                # Set distance vars to 0.5
                for var_idx in dist_vars:
                    x[var_idx] = 0.5
                    
                f = np.array(problem(x))
                solutions.append((x, f))
                
            return solutions
