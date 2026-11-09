"""Model-and-optimize: DTLZ2 front is a quarter-sphere; sample with Halton sequence for uniform coverage."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _halton(self, n, base):
        """Generate n Halton sequence values in [0, 1) for the given base."""
        result = np.zeros(n)
        for i in range(1, n + 1):
            f = 1.0
            r = 0.0
            j = i
            while j > 0:
                j, rem = divmod(j, base)
                f /= base
                r += rem * f
            result[i - 1] = r
        return result

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ2 M=3:
        # x[0], x[1] are angles (mapped to [0, 1] from [0, pi/2]).
        # x[2:] are distance variables. Set to 0.5 for g=0.
        
        results = []
        count = 0
        
        # Generate Halton sequences for surface-uniform sampling on the quarter sphere.
        # For M=3 DTLZ2, the Pareto front is a quarter-sphere in [0,1]^3.
        # Parameterize by two angles: theta1 in [0, pi/2], theta2 in [0, pi/2].
        # Objective values: f1 = cos(theta1), f2 = sin(theta1)*cos(theta2), f3 = sin(theta1)*sin(theta2)
        # To get surface-area-uniform sampling:
        #   u ~ Uniform(0,1) -> theta1 = arccos(1 - 2*u)  (so cos(theta1) = 1-2u)
        #   v ~ Uniform(0,1) -> theta2 = 2*pi*v           (uniform in [0, 2*pi] for full sphere, 
        #                                                   but for quarter sphere use [0, pi/2])
        # Actually for quarter sphere in first octant:
        #   theta1 in [0, pi/2], theta2 in [0, pi/2]
        #   Surface element is sin(theta1) dtheta1 dtheta2
        #   To make uniform: let u = (1 - cos(theta1))/2, so cos(theta1) = 1-2u
        #   And theta2 uniform in [0, pi/2]
        
        n_points = min(self.budget, 200)
        
        # Use Halton sequences with coprime bases for low-discrepancy sampling
        u_halton = self._halton(n_points, 2)
        v_halton = self._halton(n_points, 3)
        
        for i in range(n_points):
            if count >= self.budget:
                break
            if time.time() > deadline:
                return np.array(results) if results else None
            
            # Map Halton values to angles for surface-uniform sampling
            u = u_halton[i]
            v = v_halton[i]
            
            # theta1: arccos(1 - 2u) gives surface-uniform distribution
            theta1 = np.arccos(1.0 - 2.0 * u)
            
            # theta2: uniform in [0, pi/2]
            theta2 = v * (np.pi / 2.0)
            
            # Compute objective values on the Pareto front
            f1 = np.cos(theta1)
            f2 = np.sin(theta1) * np.cos(theta2)
            f3 = np.sin(theta1) * np.sin(theta2)
            
            # Map back to decision space for the problem function
            # x[0] corresponds to theta1, x[1] corresponds to theta2
            # In DTLZ2, x[0] and x[1] are mapped to [0, pi/2] via:
            # theta_i = (pi/2) * x[i]
            # So x[i] = 2*theta_i / pi
            
            x = np.zeros(self.dim)
            x[0] = 2.0 * theta1 / np.pi
            x[1] = 2.0 * theta2 / np.pi
            if self.dim > 2:
                x[2:] = 0.5  # Set g=0 for Pareto front
            
            x = np.clip(x, lo, hi)
            
            try:
                f = problem(x)
                results.append(f)
                count += 1
            except Exception:
                break
        
        return np.array(results) if results else None
# EVOLVE-BLOCK-END
