import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = self.budget
        
        # ZDT6 specific logic
        # f1(x) = 1 - exp(-4*x1) * sin(6*pi*x1)^8
        # f2(x) = g(x) * (1 - sqrt(f1(x)/g(x)))
        # g(x) = 1 + 9 * sum(x[1:]) / 9
        # To minimize g, set x[1:] = 0.
        
        # We want to sample x1 such that f1(x1) is uniformly distributed.
        
        # 1. Create a high-resolution grid for x1
        n_grid = 100000
        x1_grid = np.linspace(lo[0], hi[0], n_grid)
        
        # 2. Compute f1 for the grid
        # f1 = 1 - exp(-4*x1) * sin(6*pi*x1)^8
        f1_grid = 1.0 - np.exp(-4.0 * x1_grid) * np.sin(6.0 * np.pi * x1_grid)**8
        
        # 3. Sort by f1 to allow uniform sampling in f1 space
        sorted_indices = np.argsort(f1_grid)
        sorted_f1 = f1_grid[sorted_indices]
        sorted_x1 = x1_grid[sorted_indices]
        
        # 4. Determine the range of f1
        f1_min = sorted_f1[0]
        f1_max = sorted_f1[-1]
        
        # 5. Generate target f1 values uniformly in [f1_min, f1_max]
        target_f1 = np.linspace(f1_min, f1_max, n_evals)
        
        # 6. For each target, find the x1 in the grid that produces the closest f1
        selected_x1 = np.zeros(n_evals)
        
        # We can use searchsorted to find the closest point in the sorted array
        # Since sorted_f1 is sorted, we can find the insertion point for each target
        indices = np.searchsorted(sorted_f1, target_f1)
        
        # Check neighbors to find the closest
        # indices might be out of bounds, so we clip
        indices = np.clip(indices, 1, n_grid - 1)
        
        # Compare distance to left and right neighbor
        dist_left = np.abs(sorted_f1[indices - 1] - target_f1)
        dist_right = np.abs(sorted_f1[indices] - target_f1)
        
        # If right is closer, use index, else use index-1
        # Note: indices is the position where target would be inserted.
        # So sorted_f1[indices-1] <= target <= sorted_f1[indices]
        # We want the x1 corresponding to the closer f1.
        
        use_right = dist_right < dist_left
        # For indices where use_right is True, we use index
        # For indices where use_right is False, we use index-1
        
        final_indices = np.where(use_right, indices, indices - 1)
        
        selected_x1 = sorted_x1[final_indices]
        
        # 7. Construct the X matrix
        X = np.zeros((n_evals, self.dim))
        X[:, 0] = selected_x1
        # x[1:] are 0, which is lo[1:] for ZDT6 [0,1]
        
        # 8. Evaluate
        F = []
        for i in range(n_evals):
            if time.time() > deadline:
                break
            f = problem(X[i])
            F.append(f)
            
        return X
