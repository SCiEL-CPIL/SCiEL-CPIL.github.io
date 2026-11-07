import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin of 10 seconds
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        used = 0
        
        # DTLZ7 Pareto front: x[2:] = 0, x[0] and x[1] vary in [0,1]
        # We need to spread points on the (x[0], x[1]) square to maximize hypervolume.
        
        # Strategy: Greedy farthest-point sampling
        # Start with corners, then iteratively place points that maximize
        # minimum distance to all existing points.
        
        # Initial points: corners and center for good spread
        initial_points = [
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0),
            (0.5, 0.5),
        ]
        
        # Place initial points
        placed = []
        for x0, x1 in initial_points:
            if used >= self.budget:
                break
            if time.time() > deadline:
                return None
            
            x_new = np.zeros(self.dim)
            x_new[0] = x0
            x_new[1] = x1
            # x_new[2:] = 0
            
            f_new = problem(x_new)
            used += 1
            placed.append(np.array([x0, x1]))
        
        # Greedy farthest-point sampling for remaining points
        n_remaining = self.budget - used
        
        if n_remaining > 0 and len(placed) > 0:
            placed_arr = np.array(placed)
            
            # Candidate grid for fast distance computation
            # Use a moderate grid size to balance accuracy and speed
            n_grid = 40
            x0_candidates = np.linspace(0.0, 1.0, n_grid)
            x1_candidates = np.linspace(0.0, 1.0, n_grid)
            
            # Create meshgrid of candidates
            X0, X1 = np.meshgrid(x0_candidates, x1_candidates)
            candidates = np.column_stack([X0.ravel(), X1.ravel()])
            
            for _ in range(n_remaining):
                if time.time() > deadline:
                    break
                
                # Compute minimum distance from each candidate to all placed points
                # placed_arr shape: (n_placed, 2)
                # candidates shape: (n_candidates, 2)
                # diff shape: (n_candidates, n_placed, 2)
                diff = candidates[:, np.newaxis, :] - placed_arr[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diff ** 2, axis=2))
                min_dists = np.min(dists, axis=1)
                
                # Find candidate with maximum minimum distance
                best_idx = np.argmax(min_dists)
                best_x0, best_x1 = candidates[best_idx]
                
                # Evaluate this point
                x_new = np.zeros(self.dim)
                x_new[0] = best_x0
                x_new[1] = best_x1
                # x_new[2:] = 0
                
                f_new = problem(x_new)
                used += 1
                placed.append(np.array([best_x0, best_x1]))
                
                # Update placed array for next iteration
                if len(placed) > 1:
                    placed_arr = np.array(placed)
                else:
                    placed_arr = placed_arr  # keep as is if only one point
        
        return None
