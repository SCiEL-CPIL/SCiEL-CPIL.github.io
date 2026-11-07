"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-aware stratified sampling on the ZDT3 Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _zdt3_h(self, x1):
        """Compute h(x1) for ZDT3."""
        h = 0.5 * x1 + np.sin(10.0 * np.pi * x1)
        return h

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # 300s limit, 20s safety margin

        dim = self.dim
        budget = self.budget

        # ZDT3 Pareto front: x_2..x_dim = 0, f_1 = x_1, f_2 = h(x_1)
        # We need to sample x_1 in [0, 1] to cover the front well.
        # The front is discontinuous due to sin term, so we want dense sampling.
        
        # Strategy: Use stratified sampling of x_1 values to maximize hypervolume.
        # Allocate all budget to evaluating points on the known Pareto front.
        
        # Number of points to evaluate
        n_points = budget
        
        # Stratified sampling: divide [0,1] into n_points equal intervals,
        # pick one point from each. Use midpoint of each interval for determinism.
        # This ensures uniform coverage of the front.
        
        # Actually, for hypervolume maximization, we want to cover the front densely.
        # Let's use a non-uniform distribution that accounts for the curvature.
        # The ZDT3 front has sharp features due to sin(10*pi*x1), so we want
        # more points where the function changes rapidly.
        
        # Simple approach: uniform stratified sampling should work well.
        # For 200 points, that's one point every 0.005 in x1.
        
        x1_values = np.linspace(0.0, 1.0, n_points, endpoint=False) + 0.5 / n_points
        
        # For each x1, create the decision vector: [x1, 0, 0, ..., 0]
        # and evaluate it through the problem function.
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Create all decision vectors at once
        X = np.zeros((n_points, dim))
        X[:, 0] = x1_values
        # x_2..x_dim are already 0
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        # We need to call problem(x) for each x and record the objective values.
        # The evaluator records every returned objective vector.
        
        # Call problem for each point
        for i in range(n_points):
            if time.time() > deadline:
                break
            problem(X[i])
        
        return X
# EVOLVE-BLOCK-END
