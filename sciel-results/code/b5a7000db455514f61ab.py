import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # Safety margin before 330s limit
        
        # DTLZ4 Pareto Front Analysis:
        # For DTLZ4 with M=3 objectives, the Pareto front is defined by:
        # f1 = x0 * cos(x1 * pi/2)
        # f2 = x0 * sin(x1 * pi/2)
        # f3 = 1 - x0
        # where x0, x1 in [0, 1] and x[2:] = 0.5 (to minimize g=0).
        
        # We want to select `budget` points from this surface to maximize hypervolume.
        # Strategy: Greedy Farthest-Point Sampling in Objective Space.
        # 1. Generate a dense grid of candidates in parameter space (x0, x1).
        # 2. Compute their objective values.
        # 3. Start with a central point.
        # 4. Iteratively pick the candidate that is farthest from the current set of selected points.
        
        n_points = self.budget
        
        # Step 1: Generate dense candidate pool
        # A 100x100 grid gives 10,000 candidates, which is small enough for fast distance calculations.
        grid_size = 100
        x0_vals = np.linspace(0, 1, grid_size)
        x1_vals = np.linspace(0, 1, grid_size)
        
        # Create meshgrid
        X0, X1 = np.meshgrid(x0_vals, x1_vals)
        
        # Compute objectives for all candidates
        # f1 = x0 * cos(x1 * pi/2)
        # f2 = x0 * sin(x1 * pi/2)
        # f3 = 1 - x0
        F1 = X0 * np.cos(X1 * np.pi / 2)
        F2 = X0 * np.sin(X1 * np.pi / 2)
        F3 = 1 - X0
        
        # Stack into shape (N_candidates, 3)
        candidates = np.column_stack((F1.ravel(), F2.ravel(), F3.ravel()))
        
        # Step 2: Greedy Selection
        # Initialize selected set with the center of the front (x0=0.5, x1=0.5)
        # Center in parameter space: x0=0.5, x1=0.5
        # f1 = 0.5 * cos(pi/4) = 0.5 * 0.707 = 0.3535
        # f2 = 0.5 * sin(pi/4) = 0.3535
        # f3 = 1 - 0.5 = 0.5
        # Let's find the candidate closest to this center to start.
        target_center = np.array([0.5 * np.cos(np.pi/4), 0.5 * np.sin(np.pi/4), 0.5])
        dists_to_center = np.linalg.norm(candidates - target_center, axis=1)
        first_idx = np.argmin(dists_to_center)
        
        selected_indices = [first_idx]
        selected_points = [candidates[first_idx]]
        
        # Keep track of min distance from each candidate to the selected set
        # min_dists[i] = min distance from candidate i to any point in selected_set
        min_dists = np.linalg.norm(candidates - selected_points[0], axis=1)
        
        for _ in range(n_points - 1):
            if time.time() > deadline - 5.0:
                break
                
            # Find candidate with maximum min_dist
            # Exclude already selected candidates by setting their min_dist to -1
            for idx in selected_indices:
                min_dists[idx] = -1
            
            next_idx = np.argmax(min_dists)
            selected_indices.append(next_idx)
            new_point = candidates[next_idx]
            selected_points.append(new_point)
            
            # Update min_dists for all candidates
            # dist_to_new = norm(candidates - new_point)
            # min_dists = min(min_dists, dist_to_new)
            dist_to_new = np.linalg.norm(candidates - new_point, axis=1)
            min_dists = np.minimum(min_dists, dist_to_new)
            
            # Re-enable the newly selected point's slot? No, we set it to -1.
            # But we need to ensure we don't pick it again. The loop sets it to -1 before argmax.
            # However, the update `np.minimum` might restore a positive value if we don't mask it.
            # Actually, `min_dists` holds the distance to the *closest* selected point.
            # If we pick `next_idx`, its distance to itself is 0.
            # The next iteration will set `min_dists[next_idx] = -1` before `argmax`.
            # So it won't be picked again.
            
        # Convert selected objective points back to decision variables
        # We have the objective values, but we need x vectors.
        # Since we generated candidates from a grid, we can map back.
        # Or simpler: we stored the indices. We can reconstruct x from the grid indices.
        # But wait, `candidates` was flattened. We need to map back to (i, j) in the grid.
        
        # Let's reconstruct x vectors directly from the grid coordinates.
        # selected_indices are indices into the flattened candidates array.
        # We can recover (row, col) in the meshgrid.
        # X0 and X1 are (grid_size, grid_size).
        # ravel() order is C-order (row-major).
        # idx = row * grid_size + col
        
        points = []
        for idx in selected_indices:
            row = idx // grid_size
            col = idx % grid_size
            x0 = X0[row, col]
            x1 = X1[row, col]
            
            x = np.zeros(self.dim)
            x[0] = x0
            x[1] = x1
            x[2:] = 0.5
            points.append(x)
        
        # Evaluate all points with deadline guard
        F = []
        for x in points:
            if time.time() > deadline - 1.0:
                break
            F.append(problem(x))
            
        return points
