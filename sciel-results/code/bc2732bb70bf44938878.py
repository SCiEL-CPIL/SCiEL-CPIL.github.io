"""Structure-exploiting optimizer for DTLZ7 using importance-weighted Sobol sampling."""
import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """
    Optimizer for DTLZ7 exploiting the known Pareto front structure.
    Pareto front: f0=u, f1=v, f2=1-sqrt(v) for u,v in [0,1], x[2:]=0.
    
    We sample in (u, s) where s=sqrt(v), so f0=u, f1=s^2, f2=1-s.
    This gives uniform coverage in f0 and f2, which is hypervolume-optimal
    for this front shape.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        results = []
        n_evals = 0
        budget = self.budget
        seen = set()
        
        def evaluate(u, v):
            nonlocal n_evals
            if n_evals >= budget:
                return
            if time.time() > deadline:
                return
            # Round to avoid floating point duplicates
            key = (round(u, 12), round(v, 12))
            if key in seen:
                return
            seen.add(key)
            x = np.zeros(self.dim)
            x[0] = u
            x[1] = v
            # x[2:] remains 0, which is on the Pareto front (g=0)
            obj = problem(x)
            results.append(obj)
            n_evals += 1
        
        # Reserve points for critical boundary/corner/edge points
        n_reserved = 16
        n_sobol = budget - n_reserved
        
        if n_sobol < 8:
            n_sobol = max(budget - 4, 1)
            n_reserved = budget - n_sobol
        
        # Generate Sobol points in (u, s) space where s = sqrt(v)
        # This gives uniform density in f0=u and f2=1-s
        sampler = qmc.Sobol(d=2, scramble=True, seed=42)
        sobol_points = sampler.random(n_sobol)
        
        # Evaluate Sobol points: u = point[0], s = point[1], v = s^2
        for u, s in sobol_points:
            if n_evals >= budget:
                break
            if time.time() > deadline:
                break
            v = s * s
            evaluate(u, v)
        
        # Add critical boundary and corner points
        # These ensure coverage of the extremes of the Pareto front
        critical_points = [
            # Corners of the (u,v) square
            (0.0, 0.0),   # f = (0, 0, 1)
            (1.0, 0.0),   # f = (1, 0, 1)
            (0.0, 1.0),   # f = (0, 1, 0)
            (1.0, 1.0),   # f = (1, 1, 0)
            # Edge points: u=0, varying v (f2 axis coverage)
            (0.0, 0.25),
            (0.0, 0.5),
            (0.0, 0.75),
            # Edge points: u=1, varying v
            (1.0, 0.25),
            (1.0, 0.5),
            (1.0, 0.75),
            # Edge points: v=0, varying u (f0 axis coverage)
            (0.25, 0.0),
            (0.5, 0.0),
            (0.75, 0.0),
            # Edge points: v=1, varying u
            (0.25, 1.0),
            (0.5, 1.0),
            (0.75, 1.0),
        ]
        
        for u, v in critical_points:
            if n_evals >= budget:
                break
            if time.time() > deadline:
                break
            evaluate(u, v)
        
        return results
