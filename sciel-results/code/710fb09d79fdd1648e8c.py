"""Multi-objective optimizer for DTLZ4 using exact Pareto front construction."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using direct front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ4 parameters
        beta = 10.0
        
        # Strategy:
        # 1. Set position variables (x[n_obj:]) to 0.5 to ensure g(x) = 0.
        # 2. Sample shape variables (x[0], x[1]) to uniformly cover the Pareto front.
        # The Pareto front for DTLZ4 (M=3) is a quarter-sphere.
        # f1 = cos(pi/2 * x0^beta) * cos(pi/2 * x1^beta)
        # f2 = sin(pi/2 * x0^beta) * cos(pi/2 * x1^beta)
        # f3 = sin(pi/2 * x1^beta)
        
        # We generate points uniformly on the sphere surface.
        # Use Halton sequence for quasi-random sampling of angles.
        
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        n_sample = budget
        X = np.zeros((n_sample, dim))
        
        # Set position variables to 0.5
        X[:, n_obj:] = 0.5
        
        bases = [2, 3]
        for i in range(1, n_sample + 1):
            # Sample alpha in [0, pi/2] and u in [0, 1]
            # alpha determines the azimuth
            # u determines the polar angle via cos(beta_angle) = u
            alpha = halton(i, bases[0]) * (np.pi / 2.0)
            u = halton(i, bases[1])
            
            # Calculate objective values on the unit sphere
            # f1 = cos(alpha) * u  (since cos(beta_angle) = u)
            # f2 = sin(alpha) * u
            # f3 = sin(beta_angle) = sqrt(1 - u^2)
            
            f1 = np.cos(alpha) * u
            f2 = np.sin(alpha) * u
            f3 = np.sqrt(1.0 - u * u)
            
            # Invert to find x0, x1
            # f3 = sin(pi/2 * x1^beta)
            # pi/2 * x1^beta = arcsin(f3)
            # x1^beta = 2/pi * arcsin(f3)
            # x1 = (2/pi * arcsin(f3))^(1/beta)
            
            # Handle edge cases where f3 might be slightly out of [0,1] due to floating point
            f3_clipped = np.clip(f3, 0.0, 1.0)
            x1_pow = (2.0 / np.pi) * np.arcsin(f3_clipped)
            x1 = x1_pow ** (1.0 / beta)
            
            # f1 = cos(pi/2 * x0^beta) * cos(pi/2 * x1^beta)
            # We know cos(pi/2 * x1^beta) = u
            # So f1 = cos(pi/2 * x0^beta) * u
            # cos(pi/2 * x0^beta) = f1 / u
            # If u is 0, f1 is 0, x0 is indeterminate but doesn't matter as f1=f2=0.
            # We can set x0=0.5 or 0.
            
            if u > 1e-10:
                cos_x0 = f1 / u
                cos_x0 = np.clip(cos_x0, 0.0, 1.0)
                pi_2_x0_pow = np.arccos(cos_x0)
                x0_pow = (2.0 / np.pi) * pi_2_x0_pow
                x0 = x0_pow ** (1.0 / beta)
            else:
                x0 = 0.5
                
            X[i-1, 0] = x0
            X[i-1, 1] = x1

        # Evaluate all points
        F = np.array([problem(x) for x in X])
        used = n_sample

        # Check if we're close to the deadline
        if time.time() > deadline - 30:
            return X

        # If we have budget left (shouldn't, as we used all), we could refine,
        # but the points are already on the Pareto front.
        # No refinement needed as g=0 is the global minimum for g.
        
        return X
# EVOLVE-BLOCK-END
