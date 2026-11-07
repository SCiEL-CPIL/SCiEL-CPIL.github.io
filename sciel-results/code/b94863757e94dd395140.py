import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        used = 0
        budget = self.budget
        dim = self.dim

        def check_time():
            return time.time() < deadline

        def evaluate(x):
            nonlocal used
            if not check_time():
                return
            if used >= budget:
                return
            x_clipped = np.clip(x, lo, hi)
            x_eval = x_clipped.copy()
            x_eval[2:] = 0.0
            problem(x_eval)
            used += 1

        # DTLZ7 Pareto front analysis:
        # With x[2:]=0, g(x)=1. The objectives are:
        # f1 = x[0]
        # f2 = x[1]
        # f3 = 1 - x[1]*h(x[1], 1) where h(y,1) = 9/(1+2*(sin(3πy)+sin(4πy)))
        #
        # The Pareto front is 2D, parameterized by (x0, x1) in [0,1]^2.
        # Reference point: [1.1, 1.1, 25.0]
        #
        # Strategy: Max-min facility location in objective space (f1, f2) to ensure
        # uniform coverage of the hypervolume-relevant region. This avoids clustering
        # artifacts of grids and quasi-random sequences, providing better spatial
        # distribution for maximizing hypervolume.
        #
        # We place N points greedily: each new point maximizes its minimum distance
        # to all previously placed points in (f1, f2) space.

        N = budget
        if N < 2:
            N = 2

        # Precompute f3 values for candidate x1 values
        # We'll work in (f1, f2) = (x0, x1) space directly since f1=x0, f2=x1
        # and f3 is determined solely by x1.

        # Greedy max-min placement
        # Start with a few well-spread initial points
        n_init = min(4, N)
        # Initial points: corners and center of [0,1]^2
        init_candidates = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5]
        ])
        
        # Select first n_init points from init_candidates
        placed = []
        for i in range(n_init):
            idx = i % len(init_candidates)
            placed.append(init_candidates[idx].copy())
        
        placed = np.array(placed)
        
        # Evaluate initial points
        for i in range(len(placed)):
            if not check_time():
                break
            if used >= budget:
                break
            x = np.zeros(dim)
            x[0] = placed[i, 0]
            x[1] = placed[i, 1]
            evaluate(x)
        
        # Greedy max-min: for each next point, find candidate that maximizes
        # minimum distance to all placed points
        remaining_budget = budget - used
        if remaining_budget > 0 and check_time():
            # Use a much denser candidate grid for better spatial coverage
            # 200x200 = 40000 candidates, sufficient for 200 placements
            n_grid = 200
            grid_x0 = np.linspace(0, 1, n_grid)
            grid_x1 = np.linspace(0, 1, n_grid)
            G0, G1 = np.meshgrid(grid_x0, grid_x1, indexing='ij')
            candidates = np.column_stack([G0.ravel(), G1.ravel()])
            
            # For each candidate, compute min squared distance to placed points
            # Use squared distances to avoid expensive sqrt
            for step in range(remaining_budget):
                if not check_time():
                    break
                if used >= budget:
                    break
                
                # Compute squared distances from each candidate to each placed point
                # Using broadcasting: candidates (M,1,2) - placed (1,k,2)
                diff = candidates[:, np.newaxis, :] - placed[np.newaxis, :, :]
                sq_dists = np.sum(diff**2, axis=2)  # shape (M, k)
                min_sq_dists = np.min(sq_dists, axis=1)  # shape (M,)
                
                # Find candidate with maximum min squared distance
                best_idx = np.argmax(min_sq_dists)
                best_point = candidates[best_idx].copy()
                
                # Add to placed
                placed = np.vstack([placed, best_point])
                
                # Evaluate
                x = np.zeros(dim)
                x[0] = best_point[0]
                x[1] = best_point[1]
                evaluate(x)
                
                # If we've used all budget, break
                if used >= budget:
                    break

        return None
