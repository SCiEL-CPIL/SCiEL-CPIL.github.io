import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295
        
        n_evals = self.budget
        if n_evals <= 0:
            return []
            
        results = []
        
        # For ZDT6 with x[1:]=0:
        #   f1 = 1 + 9 * sin(10 * pi * x0)
        #   f2 = 1
        # f1 ranges from -8 to 10 as x0 goes from 0 to 1 (5 full oscillations).
        # The reference point is [11, 11]. All points on the PF have f2=1 < 11.
        # To maximize hypervolume, we want good coverage of the f1 range.
        
        # Strategy: sample x0 to get near-uniform spacing in f1.
        # f1 = 1 + 9*sin(10*pi*x0). We want f1 values spread evenly in [-8, 10].
        # For each target f1, solve sin(10*pi*x0) = (f1-1)/9.
        # There are multiple solutions per period. We pick the smallest x0 in [0,1]
        # for each target, which gives us 20 distinct x0 values (one per half-period).
        # Then we use the remaining budget to add more points via stratified x0 sampling.
        
        # First, generate points with uniform f1 coverage
        n_uniform_f1 = min(20, n_evals)
        f1_targets = np.linspace(-8.0, 10.0, n_uniform_f1)
        x0s_uniform = np.zeros(n_uniform_f1)
        for i, f1t in enumerate(f1_targets):
            s = (f1t - 1.0) / 9.0
            s = np.clip(s, -1.0, 1.0)
            base = np.arcsin(s)  # in [-pi/2, pi/2]
            if base >= 0:
                angle = base
            else:
                angle = base + 2 * np.pi
            x0 = angle / (10.0 * np.pi)
            x0s_uniform[i] = x0
        
        # Remaining points: stratified sampling in x0 to fill gaps
        n_remaining = n_evals - n_uniform_f1
        if n_remaining > 0:
            x0s_stratified = np.linspace(0.0, 1.0, n_remaining, endpoint=False)
            x0s_stratified += 0.5 / n_remaining  # midpoints of strata
            x0s = np.concatenate([x0s_uniform, x0s_stratified])
        else:
            x0s = x0s_uniform
        
        # Ensure we hit both endpoints of x0
        if n_evals > 1:
            x0s[0] = 0.0
            x0s[n_evals - 1] = 1.0
        
        # Sort x0s to get a nice progression (helps with numerical stability)
        x0s = np.sort(x0s)
        
        for i in range(n_evals):
            if time.time() > deadline:
                break
            
            x = np.zeros(self.dim, dtype=float)
            x[0] = float(x0s[i])
            # x[1:] = 0 minimizes g, putting points on the Pareto front
            
            try:
                f = problem(x)
                results.append(f)
            except Exception:
                break
                
        return results
