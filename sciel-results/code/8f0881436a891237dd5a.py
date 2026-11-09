import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # 15s safety margin before 300s kill
        dim = self.dim
        budget = self.budget
        
        # Use Latin Hypercube Sampling for better space coverage
        # This is structural change: from local search to global space-filling search
        # f7 is a step function, so local gradients are useless.
        # We need to hit the hidden ellipsoid. LHS maximizes the chance of hitting
        # a small target region compared to random or random-walk methods.
        
        sampler = qmc.LatinHypercube(d=dim, seed=42)
        # Generate all points at once for efficiency
        points = sampler.random(n=budget)
        # Scale to [-4, 4]
        # qmc.scale scales to [low, high]
        points = qmc.scale(points, -4.0, 4.0)
        
        # Evaluate first point
        best_x = points[0]
        best_f = func(best_x)
        calls = 1
        
        # Evaluate remaining points
        for i in range(1, budget):
            if time.time() > deadline:
                break
            
            x = points[i]
            f = func(x)
            calls += 1
            
            if f < best_f:
                best_f = f
                best_x = x.copy()
                
        return best_x, best_f
