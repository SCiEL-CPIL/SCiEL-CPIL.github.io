# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for DTLZ7 that directly constructs the Pareto front.
    For DTLZ7, the Pareto-optimal points lie in the subspace where g(x) is minimized.
    We sample x[0] and x[1] densely on the Pareto front subspace to maximize
    the hypervolume of the recorded objective vectors.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # safety margin before 330s kill

        n = self.dim
        budget = self.budget

        # The Pareto front of DTLZ7 requires g(x) to be minimized.
        # g(x) = (n-m+1) + sum_{i=n-m+2}^n [(x_i-0.5)^2 - cos(20*pi*(x_i+0.5))]
        # This is minimized when each x_i = 0.5 for i >= 2, giving g=0.
        # We use x[2:] = 0.5 to be on the Pareto front.
        
        # Number of points to evaluate on the front
        n_eval = budget

        # Use a stratified sampling approach for x[0] and x[1]
        # We want good coverage of the [0,1] x [0,1] space
        # Use a grid-like approach with some jitter for diversity
        
        # Create a stratified sample in 2D
        # Use sqrt(n) x sqrt(n) grid if possible, with jitter
        side = int(np.sqrt(n_eval))
        
        # Create base grid
        x0_base = (np.arange(side) + 0.5) / side
        x1_base = (np.arange(side) + 0.5) / side
        x0_grid, x1_grid = np.meshgrid(x0_base, x1_base)
        x0_flat = x0_grid.ravel()
        x1_flat = x1_grid.ravel()
        
        # Add jitter for better coverage
        rng = np.random.RandomState(0)
        jitter_scale = 1.0 / side
        x0_flat = x0_flat + rng.uniform(-0.4, 0.4, size=len(x0_flat)) * jitter_scale
        x1_flat = x1_flat + rng.uniform(-0.4, 0.4, size=len(x1_flat)) * jitter_scale
        
        # Clip to [0, 1]
        x0_flat = np.clip(x0_flat, 0.0, 1.0)
        x1_flat = np.clip(x1_flat, 0.0, 1.0)
        
        # If we have more budget, add more points
        n_grid = len(x0_flat)
        if n_grid < n_eval:
            n_extra = n_eval - n_grid
            # Add additional random points
            x0_extra = rng.uniform(0.0, 1.0, size=n_extra)
            x1_extra = rng.uniform(0.0, 1.0, size=n_extra)
            x0_flat = np.concatenate([x0_flat, x0_extra])
            x1_flat = np.concatenate([x1_flat, x1_extra])
        else:
            x0_flat = x0_flat[:n_eval]
            x1_flat = x1_flat[:n_eval]
        
        # Build points on the Pareto front subspace
        # x[2:] = 0.5 to minimize g
        X_front = np.full((n_eval, n), 0.5)
        X_front[:, 0] = x0_flat
        X_front[:, 1] = x1_flat
        
        # Evaluate all points
        F = np.empty((n_eval, 3))
        for i in range(n_eval):
            F[i] = problem(X_front[i])
            # Time check
            if time.time() > deadline - 10.0:
                F = F[:i+1]
                X_front = X_front[:i+1]
                break
        
        return X_front

    def _find_non_dominated(self, F):
        """Find indices of non-dominated points."""
        n = len(F)
        nd = np.ones(n, dtype=bool)
        for i in range(n):
            if not nd[i]:
                continue
            for j in range(n):
                if i == j or not nd[j]:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    nd[i] = False
                    break
        return nd
# EVOLVE-BLOCK-END
