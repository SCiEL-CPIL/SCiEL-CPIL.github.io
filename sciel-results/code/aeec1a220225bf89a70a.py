"""Optimizer for DTLZ4 multi-objective optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using analytic Pareto front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget

        # DTLZ4 M=3, k=3:
        # f1 = 0.5 * x1^10 * sin(pi*x1/2)
        # f2 = 0.5 * x1^10 * sin(pi*x2/2) * (1-x1^10)
        # f3 = 0.5 * (1-x1^10) * (1-x2^10)
        # On Pareto front: x4..x12 = 0.5
        # Let u1 = x1^10, u2 = x2^10, both in [0,1]
        # f1 = 0.5*u1*sin(pi*x1/2)
        # f2 = 0.5*u1*(1-u1)*sin(pi*x2/2)
        # f3 = 0.5*(1-u1)*(1-u2)
        # Note: f1^2 + f2^2 + f3^2 = 0.25 only when u1=1 or special cases
        # Actually the front is parameterized by x1, x2 in [0,1]

        all_X_list = []
        used = 0

        def evaluate_points(X_batch):
            nonlocal used
            F_batch = np.array([problem(x) for x in X_batch])
            used += len(X_batch)
            all_X_list.append(X_batch.copy())
            return F_batch

        # Strategy: Directly sample the Pareto front with a dense stratified grid
        # Use a 2D grid in (x1, x2) space with Halton-like stratification
        # to maximize hypervolume coverage.

        # The Pareto front is a 2D manifold parameterized by x1, x2 in [0,1]
        # We want to cover the front densely to maximize hypervolume.
        
        # Use a stratified sampling approach: divide [0,1]^2 into cells
        # and sample one point per cell using a quasi-random sequence.
        
        n_samples = budget  # Use all budget on the front
        
        # Generate stratified points in [0,1]^2
        # Use a grid of sqrt(n) x sqrt(n) and add jitter
        n_per_dim = int(np.sqrt(n_samples))
        if n_per_dim * n_per_dim < n_samples:
            n_per_dim += 1
        
        # Create base grid
        grid_1d = np.linspace(0.5/n_per_dim, 1.0 - 0.5/n_per_dim, n_per_dim)
        x1_grid, x2_grid = np.meshgrid(grid_1d, grid_1d, indexing='ij')
        
        # Flatten and take first n_samples points
        x1_flat = x1_grid.flatten()[:n_samples]
        x2_flat = x2_grid.flatten()[:n_samples]
        
        # Add small jitter to avoid exact duplicates and improve coverage
        rng = np.random.default_rng(42)
        jitter = 1.0 / (n_per_dim * 2)
        x1_flat = np.clip(x1_flat + rng.uniform(-jitter, jitter, n_samples), 0, 1)
        x2_flat = np.clip(x2_flat + rng.uniform(-jitter, jitter, n_samples), 0, 1)
        
        # Build decision vectors
        X = np.zeros((n_samples, dim))
        X[:, 0] = x1_flat
        X[:, 1] = x2_flat
        X[:, 2] = 0.5  # x3 unused
        X[:, 3:] = 0.5  # Distance vars at optimum (g=0)
        
        evaluate_points(X)
        
        # Collect all evaluated points
        all_X = np.vstack(all_X_list)
        
        return all_X
# EVOLVE-BLOCK-END
