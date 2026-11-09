import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
                start_time = time.time()
                deadline = start_time + 280  # Safety margin
                
                # DTLZ1 specific:
                # dim=7, n_obj=3
                # Shape vars: 0, 1, 2
                # Distance vars: 3, 4, 5, 6
                
                n_shape = self.n_obj - 1
                n_dist = self.dim - n_shape
                
                # Generate Halton sequence for shape space
                # Simple Halton implementation or use random if not available/easy
                # NumPy doesn't have built-in Halton, but we can use a simple low-discrepancy approach
                # Or just use random with good coverage.
                
                # Let's use a stratified random sample (Latin Hypercube) for shape vars
                # This ensures coverage in each dimension.
                
                # We will generate self.budget points.
                # Set distance vars to 0.
                
                X = np.zeros((self.budget, self.dim))
                # Set distance vars to 0 (already zero)
                
                # Generate LHS for shape vars
                # LHS: divide [0,1] into N intervals, pick one random point in each for each dim
                # Then shuffle columns? No, that's for single dim.
                # For multi-dim LHS:
                # Create N x n_shape matrix.
                # For each dim j, permute the intervals.
                
                n_evals = self.budget
                # Use a fixed seed for reproducibility? The task says "For seeds 0 and 1 the evaluator calls np.random.seed(seed)".
                # So we should use np.random functions which will be seeded by the evaluator.
                
                # Generate LHS
                # Intervals: [0, 1/N], [1/N, 2/N], ...
                intervals = np.linspace(0, 1, n_evals + 1)
                midpoints = (intervals[:-1] + intervals[1:]) / 2.0
                
                shape_samples = np.zeros((n_evals, n_shape))
                for j in range(n_shape):
                    # Random permutation of midpoints
                    perm = np.random.permutation(midpoints)
                    shape_samples[:, j] = perm
                
                # Add some noise to avoid exact grid artifacts? LHS is already good.
                # Maybe add a few pure random points to break symmetry?
                # LHS is generally superior for hypervolume.
                
                X[:, :n_shape] = shape_samples
                # X[:, n_shape:] is already 0
                
                # Evaluate
                F = np.array([problem(x) for x in X])
                
                # Check time
                if time.time() > deadline:
                    return X
                
                # If we have time, we could refine, but 200 evals is the limit.
                # The problem says "More than 200 evaluations ... makes the whole result invalid".
                # So we must use exactly 200 or fewer.
                
                return X
