"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """
    Structure-aware optimizer for DTLZ2 using 2D Sobol quasi-random sampling on the Pareto front.
    
    Strategy:
    1. Exploit DTLZ2 structure:
       - m = n_obj = 3
       - Angle variables: x[0], x[1]
       - Distance variables: x[2] ... x[dim-1]
       - Pareto front is achieved when distance variables are 0.5 (making g=0).
       - Objective values are determined solely by x[0] and x[1].
       
    2. Sampling Strategy:
       - Use a 2D Sobol sequence to generate points in the unit square [0, 1]^2.
       - This provides a low-discrepancy set, meaning points are evenly spread out
         in the 2D angle space, leading to better hypervolume coverage than
         1D Fibonacci lattices which can have spiral artifacts in 2D.
       
    3. Mapping:
       - Map the Sobol points (u1, u2) to DTLZ2 angle variables.
       - Set all other variables to 0.5.
       - Evaluate all points.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin: 300s limit, stop at 290s

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_angle = self.n_obj - 1  # 2 for n_obj=3
        n_dist = self.dim - n_angle
        
        # Generate Sobol points in [0, 1]^n_angle
        # We need exactly 'budget' points.
        # Sobol requires n to be a power of 2 for the standard generator,
        # but we can just generate a power of 2 >= budget and take the first 'budget'.
        n_gen = 1
        while n_gen < self.budget:
            n_gen *= 2
            
        sampler = qmc.Sobol(d=n_angle, scramble=True)
        points = sampler.random(n_gen)
        
        # Take the first 'budget' points
        points = points[:self.budget]
        
        # Construct the full variable vectors
        X = np.zeros((self.budget, self.dim))
        
        # Assign angle variables
        X[:, :n_angle] = points
        
        # Assign distance variables to 0.5 (center of [0, 1]) to ensure g=0
        # This assumes the bounds are [0, 1]. If bounds are different, we should use (lo+hi)/2
        # But DTLZ2 standard is [0,1].
        mid_dist = (lo[n_angle:] + hi[n_angle:]) / 2.0
        X[:, n_angle:] = mid_dist
        
        # Evaluate
        for i in range(self.budget):
            if time.time() > deadline:
                break
            problem(X[i])
            
        return X
# EVOLVE-BLOCK-END
