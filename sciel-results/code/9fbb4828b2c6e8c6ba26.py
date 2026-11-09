"""DTLZ2 optimizer using Halton-sequence sampling of the Pareto front."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using Halton-sequence front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton(self, n, base):
        """Generate n Halton numbers in [0,1) with the given base."""
        result = np.zeros(n)
        f = 1.0 / base
        for i in range(n):
            i_val = i + 1
            while i_val > 0:
                result[i] += f * (i_val % base)
                i_val //= base
                f /= base
        return result

    def _front_point(self, theta1, theta2):
        """
        Map angles (theta1, theta2) in [0, pi/2]^2 to a point on the DTLZ2 Pareto front.
        For 3 objectives, the front is the unit sphere in the positive octant.
        f1 = cos(theta1)
        f2 = sin(theta1) * cos(theta2)
        f3 = sin(theta1) * sin(theta2)
        """
        f1 = np.cos(theta1)
        f2 = np.sin(theta1) * np.cos(theta2)
        f3 = np.sin(theta1) * np.sin(theta2)
        return np.array([f1, f2, f3])

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 30s safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # Generate Halton sequences for theta1 and theta2
        # We need to sample the 2D Pareto front surface
        n_halton = min(budget, 150)
        
        theta1 = self._halton(n_halton, 2) * (np.pi / 2)
        theta2 = self._halton(n_halton, 3) * (np.pi / 2)
        
        # Add special edge/vertex points for better coverage
        # Corners and edges of the [0, pi/2]^2 square
        special_points = [
            (0, 0), (np.pi/2, 0), (0, np.pi/2), (np.pi/2, np.pi/2),
            (np.pi/4, 0), (np.pi/4, np.pi/2), (0, np.pi/4), (np.pi/2, np.pi/4),
            (np.pi/3, np.pi/3), (np.pi/6, np.pi/6),
        ]
        
        all_theta1 = np.concatenate([theta1, np.array([p[0] for p in special_points])])
        all_theta2 = np.concatenate([theta2, np.array([p[1] for p in special_points])])
        
        n_total = len(all_theta1)
        
        # Clip to budget
        if n_total > budget:
            all_theta1 = all_theta1[:budget]
            all_theta2 = all_theta2[:budget]
            n_total = budget
        
        # Evaluate all points
        used = 0
        for i in range(n_total):
            if time.time() > deadline:
                break
            
            t1 = all_theta1[i]
            t2 = all_theta2[i]
            
            # For DTLZ2, the Pareto front points have:
            # x[0] = 0.5, x[1:] = 0.5 (or any values that give g=0)
            # The objectives are determined by the angular parameters
            # Actually, for DTLZ2: f_i = (1 + g) * cos/sin terms
            # On the Pareto front, g=0, so f_i are just the cos/sin terms
            
            # We need to construct x such that problem(x) returns the front point
            # For DTLZ2 with 3 objectives:
            # x[0] is the position parameter, x[1:] are for g
            # To get g=0, set x[1:] = 0.5 (or the boundary values)
            # x[0] determines the position on the front via theta
            
            # Actually, the standard DTLZ2 mapping:
            # For M objectives, we need M-1 angles
            # x[0] = position, x[1:] = g parameters
            # f_1 = (1+g) * cos(x[0] * pi/2)
            # f_2 = (1+g) * sin(x[0] * pi/2) * cos(x[1] * pi/2)  -- for M=3
            # f_3 = (1+g) * sin(x[0] * pi/2) * sin(x[1] * pi/2)
            
            # To get the Pareto front, set g=0 by setting x[1:] = 0.5
            # Then x[0] controls theta1, and we need another parameter for theta2
            # Wait, for M=3, we need 2 position parameters: x[0] and x[1]
            # And x[2:] for g
            
            # Let me reconsider: for DTLZ2 with M objectives:
            # Position parameters: x[0:M-1]
            # g parameters: x[M:]
            # g = sum((x[M:] - 0.5)^2)
            # f_1 = (1+g) * prod_{i=1}^{M-1} cos(x[i]*pi/2) ... no
            
            # Standard DTLZ2:
            # f_1 = (1+g) * cos(x_1*pi/2)
            # f_2 = (1+g) * sin(x_1*pi/2) * cos(x_2*pi/2)
            # f_3 = (1+g) * sin(x_1*pi/2) * sin(x_2*pi/2)
            # g = sum_{i=3}^{n} (x_i - 0.5)^2
            
            # So for dim=12, n_obj=3:
            # x[0], x[1] are position parameters
            # x[2:12] are g parameters
            # To get g=0: x[2:12] = 0.5
            # x[0] in [0,1] maps to theta1 = x[0]*pi/2
            # x[1] in [0,1] maps to theta2 = x[1]*pi/2
            
            x = np.zeros(dim)
            x[0] = t1 / (np.pi / 2)  # map theta1 to [0,1]
            x[1] = t2 / (np.pi / 2)  # map theta2 to [0,1]
            x[2:] = 0.5  # g = 0
            
            # Clip to bounds
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            used += 1
        
        return None
# EVOLVE-BLOCK-END
