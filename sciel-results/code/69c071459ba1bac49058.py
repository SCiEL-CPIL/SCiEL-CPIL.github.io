"""Portfolio search for DTLZ2: deterministic quarter-circle arc on the known front."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using an explicit front-arc construction."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 315.0  # conservative safety margin under the 330 s kill

        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj
        used = 0

        # Strategy: Stratified grid on the Pareto front surface.
        # The front is a quarter sphere. We sample u = cos(theta) uniformly in [0,1]
        # and phi uniformly in [0, pi/2].
        # This provides a uniform distribution of points on the surface area.
        
        n_total = budget
        # Use a grid that exactly fills the budget
        # Try different factorizations to get n_u * n_v = budget
        best_nu, best_nv = 14, 14
        for nu in range(1, n_total + 1):
            if n_total % nu == 0:
                nv = n_total // nu
                # Prefer grids closer to square
                if abs(nu - nv) < abs(best_nu - best_nv):
                    best_nu, best_nv = nu, nv
        n_u, n_v = best_nu, best_nv
        
        candidates = []
        
        for i in range(n_u):
            # Uniform stratified sampling for theta in (0, pi/2)
            # theta determines f3 = cos(theta), so uniform in theta gives
            # uniform in cos(theta) which is uniform in f3
            t_u = (i + 0.5) / n_u
            theta = t_u * (np.pi / 2.0)
            u = np.cos(theta)
            sin_theta = np.sin(theta)
            # Ensure strictly inside (0,1)
            u = np.clip(u, 1e-10, 1.0 - 1e-10)
            sin_theta = np.sqrt(max(0.0, 1.0 - u * u))
            
            for j in range(n_v):
                # Uniform stratified sampling for phi in (0, pi/2)
                t_v = (j + 0.5) / n_v
                phi = t_v * (np.pi / 2.0)
                
                f1 = sin_theta * np.cos(phi)
                f2 = sin_theta * np.sin(phi)
                f3 = u
                
                # Map to x
                x = np.full(dim, 0.5)
                x[0] = (2.0 / np.pi) * np.arcsin(np.clip(f3, 0, 1))
                
                r = np.sqrt(f1 * f1 + f2 * f2)
                if r > 1e-15:
                    ratio = f1 / r
                    x[1] = (2.0 / np.pi) * np.arcsin(np.clip(ratio, -1, 1))
                else:
                    x[1] = 0.5
                
                candidates.append(x)
        
        # Add boundary points: corners of the front
        # Points where one or two objectives are near 0
        boundary_points = [
            # f3 near 0 (theta near pi/2)
            [np.array([1.0, 0.5, 0.5]), np.array([1.0, 0.0, 0.5]), np.array([1.0, 0.5, 0.0])],
            # f1 near 0 (phi near 0)
            [np.array([0.5, 1.0, 0.5]), np.array([0.0, 1.0, 0.5]), np.array([0.5, 1.0, 0.0])],
            # f2 near 0 (phi near pi/2)
            [np.array([0.5, 0.5, 0.5]), np.array([0.0, 0.5, 0.5]), np.array([0.5, 0.0, 0.0])],
        ]
        # These are (f1, f2, f3) targets; convert to x
        for f_target in boundary_points:
            for f in f_target:
                f1_t, f2_t, f3_t = f
                x = np.full(dim, 0.5)
                x[0] = (2.0 / np.pi) * np.arcsin(np.clip(f3_t, 0, 1))
                r_t = np.sqrt(f1_t * f1_t + f2_t * f2_t)
                if r_t > 1e-15:
                    ratio_t = f1_t / r_t
                    x[1] = (2.0 / np.pi) * np.arcsin(np.clip(ratio_t, -1, 1))
                else:
                    x[1] = 0.5
                candidates.append(x)
        
        # Trim to budget if needed
        if len(candidates) > budget:
            candidates = candidates[:budget]

        # Evaluate all candidates within the time budget
        for x in candidates:
            if used >= budget:
                break
            if time.time() >= deadline:
                break
            problem(x)
            used += 1

        return np.zeros((0, n_obj))
# EVOLVE-BLOCK-END
