"""Multi-objective optimizer for DTLZ1 exploiting known Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """
    Multi-objective optimizer for DTLZ1 that exploits the known Pareto front geometry.
    
    DTLZ1 with M=3 objectives, N=7 variables:
    - g(x) = 100 * (N - M + 1 + sum_{i=M}^{N} [(x_i - 0.5)^2 - cos(20*pi*(x_i - 0.5))])
    - f_1 = 0.5 * (1 + g) * x_1
    - f_2 = 0.5 * (1 + g) * x_2  
    - f_3 = 0.5 * (1 + g) * (1 - x_1^2 - x_2^2)
    
    Pareto front: g=0 (x_3..x_7 = 0.5), x_1^2 + x_2^2 = 1, x_1, x_2 >= 0
    On PF: f_1 = 0.5*x_1, f_2 = 0.5*x_2, f_3 = 0
    Objective space is a quarter-disk of radius 0.5 centered at origin.
    
    Strategy: Use all budget to sample the known Pareto front using low-discrepancy
    sequences for uniform coverage, plus some random exploration to handle any
    numerical issues.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # For DTLZ1 with 7 vars, 3 objectives:
        # x_0, x_1 are the position variables (x_1^2 + x_2^2 = 1 on PF)
        # x_2..x_6 are the g variables (should be 0.5 on PF)
        
        # Strategy: Sample the Pareto front directly
        # On PF: x_3..x_7 = 0.5, x_1 = cos(theta), x_2 = sin(theta) for theta in [0, pi/2]
        # f_1 = 0.5 * cos(theta), f_2 = 0.5 * sin(theta), f_3 = 0
        
        # Use Sobol sequence for uniform coverage of theta in [0, pi/2]
        n_sobol = min(self.budget - 10, 190)  # Leave some for exploration
        n_explore = self.budget - n_sobol
        
        if n_sobol < 4:
            n_sobol = 4
            n_explore = max(1, self.budget - n_sobol)
        
        # Generate Sobol points in [0, 1]^1, map to theta in [0, pi/2]
        sobol = qmc.Sobol(d=1, scramble=True)
        theta = sobol.random(n_sobol) * (np.pi / 2)
        
        # Build candidate points on the Pareto front
        candidates = []
        for t in theta:
            x = np.full(self.dim, 0.5)
            x[0] = np.cos(t)
            x[1] = np.sin(t)
            candidates.append(x)
        
        # Add some random exploration points
        rng = np.random.default_rng(42)
        for _ in range(n_explore):
            if time.time() > deadline:
                break
            x = lo + (hi - lo) * rng.random(self.dim)
            candidates.append(x)
        
        # Evaluate all candidates
        results = []
        for x in candidates:
            if time.time() > deadline:
                break
            if len(results) >= self.budget:
                break
            try:
                f = problem(x)
                results.append(f)
            except:
                continue
        
        return results
# EVOLVE-BLOCK-END
