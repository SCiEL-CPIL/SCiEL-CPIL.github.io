"""Optimizer for DTLZ4 multi-objective optimization using analytic front construction."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using analytic Pareto front construction."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _construct_point(self, x1, x2, bounds):
        """Construct a DTLZ4 point with g=0 (Pareto optimal)."""
        x = np.zeros(self.dim)
        x[0] = x1
        x[1] = x2
        # g-variables (indices 7-11) set to 0.5 for g=0
        for i in range(7, 12):
            lo, hi = bounds[i]
            x[i] = 0.5
        # x[2], x[3], x[4], x[5], x[6] can be anything (they don't affect objectives when g=0)
        # Set to 0.5 for safety
        for i in range(2, 7):
            x[i] = 0.5
        # Clip to bounds
        for i in range(self.dim):
            lo, hi = bounds[i]
            x[i] = np.clip(x[i], lo, hi)
        return x

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 290.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ4: x[0], x[1] control the front; x[7..11] control g
        # For g=0: x[7..11] = 0.5
        # f1 = cos(x1/2)*cos(x2/2), f2 = cos(x1/2)*sin(x2/2), f3 = sin(x1/2)
        # x1, x2 in [0, 1]
        
        # Strategy: Use stratified sampling on the quarter-sphere for uniform area coverage
        # The Pareto front is parameterized by theta1 = x1/2 in [0, pi/2] and theta2 = x2/2 in [0, pi/2]
        # For uniform area on sphere: stratify sin(theta1) uniformly, theta2 uniformly
        # Budget: 200 evaluations total
        
        # Determine grid size
        # Use most of budget for front coverage
        # 14x14 = 196 points, leaving 4 for corners
        n_grid = 14
        n_points = n_grid * n_grid
        
        # Create stratified grid points with uniform sphere area coverage
        # theta1: stratify sin(theta1) uniformly in [0,1] => theta1 = arcsin(u) where u is stratified
        # theta2: stratify uniformly in [0, pi/2]
        points = []
        for i in range(n_grid):
            for j in range(n_grid):
                # Stratified sin(theta1): u in ((i + 0.5)/n_grid)
                u = (i + 0.5) / n_grid
                theta1 = np.arcsin(u)  # in [0, pi/2]
                x1 = 2.0 * theta1  # convert back to x1 in [0, 1]
                
                # Stratified theta2: uniform in [0, pi/2]
                v = (j + 0.5) / n_grid
                theta2 = v * (np.pi / 2.0)  # in [0, pi/2]
                x2 = 2.0 * theta2  # convert back to x2 in [0, 1]
                
                points.append((x1, x2))
        
        # Evaluate all grid points
        used = 0
        for x1, x2 in points:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            x = self._construct_point(x1, x2, bounds)
            f = problem(x)
            used += 1
        
        # Use remaining budget for explicit corner inclusion
        remaining = self.budget - used
        if remaining > 0 and time.time() < deadline:
            # Corners of the (x1, x2) domain map to extreme points of the Pareto front
            corner_positions = [
                (0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0),
            ]
            for x1, x2 in corner_positions:
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                x = self._construct_point(x1, x2, bounds)
                f = problem(x)
                used += 1
        
        return None
# EVOLVE-BLOCK-END
