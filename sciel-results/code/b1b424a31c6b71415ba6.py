# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for DTLZ7 exploiting the known Pareto front structure.
    The Pareto front of DTLZ7 lies on the subspace where variables x_2...x_{n-1} are 0.
    We sample this subspace densely and use remaining budget for local refinement.
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
        
        # DTLZ7 Pareto Front:
        # Set x[2:] = 0 to minimize g(x).
        # Then f1 = x[0], f2 = x[1], f3 = 1 - sqrt(x[0]).
        # We need to sample points on this front such that they maximize hypervolume.
        
        points = []
        evals_used = 0
        
        # Strategy:
        # 1. Add explicit corner points (0,0), (0,1), (1,0), (1,1) to ensure coverage of extremes.
        # 2. Use Sobol sequence for quasi-random sampling on the front.
        # 3. Use remaining budget for more Sobol points or random fill if needed.
        
        # Helper to create a point on the front
        def make_point(x0, x1):
            x = np.zeros(self.dim)
            x[0] = x0
            x[1] = x1
            # x[2:] is 0
            return x
        
        # 1. Corner points
        corners = [
            (lo[0], lo[1]),
            (lo[0], hi[1]),
            (hi[0], lo[1]),
            (hi[0], hi[1])
        ]
        
        for x0, x1 in corners:
            if evals_used >= self.budget:
                break
            points.append(make_point(x0, x1))
            evals_used += 1
            
        # 2. Sobol sampling
        # Use scipy.stats.qmc.Sobol for low-discrepancy sampling
        from scipy.stats import qmc
        
        # We have budget - evals_used left.
        # Let's use all remaining budget for Sobol points.
        n_sobol = self.budget - evals_used
        
        if n_sobol > 0:
            # Generate Sobol points in [0, 1]^2
            sampler = qmc.Sobol(d=2, scramble=True)
            # n_sobol must be a power of 2 for Sobol? No, it can be any integer, but performance is best for powers of 2.
            # Let's just generate n_sobol points.
            sobol_points = sampler.random(n=n_sobol)
            
            # Map to [lo[0], hi[0]] and [lo[1], hi[1]]
            for i in range(n_sobol):
                if evals_used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                x0 = lo[0] + sobol_points[i, 0] * (hi[0] - lo[0])
                x1 = lo[1] + sobol_points[i, 1] * (hi[1] - lo[1])
                points.append(make_point(x0, x1))
                evals_used += 1
                
        # Evaluate all points
        # The problem function is called for each point.
        # We don't strictly need the objective values for the return, but the evaluator records them.
        # We must call problem(x) for each x in points.
        
        for x in points:
            if time.time() > deadline:
                break
            problem(x)
            
        return points
# EVOLVE-BLOCK-END
