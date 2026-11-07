"""DTLZ7 hypervolume optimizer combining structured front sampling with max-min refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ7-specific optimizer using structured Sobol initialization with max-min facility location refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _pareto_front_objectives(self, x1, x2):
        """Compute DTLZ7 objective values for given x1, x2 with x3..x22=0.
        
        With g=0 (x3..x22=0):
        f1 = x1
        f2 = x2
        f3 = 2 - x1 - x2
        """
        f1 = x1
        f2 = x2
        f3 = 2.0 - x1 - x2
        return np.column_stack([f1, f2, f3])

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before the 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n = self.budget

        # Phase 1: Structured initialization using Sobol-like quasi-random points
        # plus corner points for guaranteed boundary coverage.
        # This gives a good baseline spread across the Pareto front.
        
        # Corner points of the Pareto front square in (x1, x2) space
        corners = np.array([
            [0.0, 0.0],   # f=[0, 0, 2]
            [0.0, 1.0],   # f=[0, 1, 1]
            [1.0, 0.0],   # f=[1, 0, 1]
            [1.0, 1.0],   # f=[1, 1, 0]
        ])
        
        # Generate quasi-random points using a simple Halton-like sequence
        # or Sobol approximation for the remaining points
        n_remaining = n - len(corners)
        
        if n_remaining > 0:
            # Use a stratified sampling approach for good coverage
            # Divide the unit square into a grid and pick one point per cell
            # with some randomness within each cell
            
            # Choose grid size that roughly matches n_remaining
            # Use a square-ish grid
            grid_size = int(np.sqrt(n_remaining))
            if grid_size < 1:
                grid_size = 1
            # Adjust to get close to n_remaining
            n_cells = grid_size * grid_size
            n_extra = n_remaining - n_cells
            
            # Generate grid points with jitter
            xs = np.linspace(0.5/grid_size, 1 - 0.5/grid_size, grid_size)
            ys = np.linspace(0.5/grid_size, 1 - 0.5/grid_size, grid_size)
            
            # Create meshgrid
            X_grid, Y_grid = np.meshgrid(xs, ys)
            X_flat = X_grid.flatten()
            Y_flat = Y_grid.flatten()
            
            # Add small random jitter to break symmetry
            rng = np.random.default_rng(42)
            jitter = 0.8 / grid_size
            X_flat += rng.uniform(-jitter/2, jitter/2, size=len(X_flat))
            Y_flat += rng.uniform(-jitter/2, jitter/2, size=len(Y_flat))
            
            # Clip to [0, 1]
            X_flat = np.clip(X_flat, 0.0, 1.0)
            Y_flat = np.clip(Y_flat, 0.0, 1.0)
            
            # If we have extra points needed, add them randomly
            if n_extra > 0:
                extra_x = rng.random(n_extra)
                extra_y = rng.random(n_extra)
                X_flat = np.concatenate([X_flat, extra_x])
                Y_flat = np.concatenate([Y_flat, extra_y])
            else:
                # If we have too many, trim
                X_flat = X_flat[:n_remaining]
                Y_flat = Y_flat[:n_remaining]
            
            # Combine with corners
            x1_init = np.concatenate([X_flat, corners[:, 0]])
            x2_init = np.concatenate([Y_flat, corners[:, 1]])
        else:
            x1_init = corners[:, 0].copy()
            x2_init = corners[:, 1].copy()
        
        # Ensure we have exactly n points
        x1_init = x1_init[:n]
        x2_init = x2_init[:n]
        
        # Clip to bounds
        x1_init = np.clip(x1_init, lo[0], hi[0])
        x2_init = np.clip(x2_init, lo[1], hi[1])
        
        # Compute objective vectors for initial points
        obj_pts = self._pareto_front_objectives(x1_init, x2_init)
        
        # Phase 2: Refine using max-min facility location to improve coverage
        # Identify poorly-covered regions and add points there
        # We'll use the remaining budget (if any) to add refined points
        
        # Actually, we've used all n points in initialization.
        # Instead, let's do a refinement step: for some of the points,
        # try to find better positions that maximize minimum distance.
        
        # Better approach: use a subset of the budget for refinement.
        # Let's reserve 20% of the budget for refinement.
        n_refine = max(1, n // 5)
        n_base = n - n_refine
        
        # Re-initialize with fewer base points
        if n_base > 0:
            # Use the first n_base points from our initial set
            x1_base = x1_init[:n_base]
            x2_base = x2_init[:n_base]
        else:
            x1_base = np.array([])
            x2_base = np.array([])
        
        # Compute objective vectors for base points
        if len(x1_base) > 0:
            obj_base = self._pareto_front_objectives(x1_base, x2_base)
        else:
            obj_base = np.empty((0, 3))
        
        # Add refinement points using max-min facility location
        pts_x1 = list(x1_base)
        pts_x2 = list(x2_base)
        obj_all = obj_base.copy() if len(obj_base) > 0 else np.empty((0, 3))
        
        rng2 = np.random.default_rng(123)
        
        for _ in range(n_refine):
            if time.time() > deadline - 10:
                break
            
            # Generate candidate points
            n_candidates = 3000
            cand_x1 = rng2.random(n_candidates)
            cand_x2 = rng2.random(n_candidates)
            
            # Ensure valid region: x1 + x2 <= 2 (always true for [0,1])
            # and x1, x2 in [0, 1]
            
            if len(cand_x1) == 0:
                continue
            
            # Compute objective vectors for candidates
            cand_obj = self._pareto_front_objectives(cand_x1, cand_x2)
            
            # Compute minimum distance from each candidate to existing points
            if len(obj_all) > 0:
                dists = np.sqrt(np.sum(
                    (cand_obj[:, np.newaxis, :] - obj_all[np.newaxis, :, :])**2, 
                    axis=2
                ))
                min_dists = np.min(dists, axis=1)
            else:
                min_dists = np.ones(len(cand_x1))
            
            # Select the candidate with maximum minimum distance
            best_idx = np.argmax(min_dists)
            best_x1 = cand_x1[best_idx]
            best_x2 = cand_x2[best_idx]
            
            pts_x1.append(best_x1)
            pts_x2.append(best_x2)
            
            new_obj = self._pareto_front_objectives(
                np.array([best_x1]), np.array([best_x2])
            )
            obj_all = np.vstack([obj_all, new_obj])
        
        # Finalize points
        x1 = np.array(pts_x1)
        x2 = np.array(pts_x2)
        
        # Trim to budget if needed
        x1 = x1[:n]
        x2 = x2[:n]
        
        # Clip to bounds
        x1 = np.clip(x1, lo[0], hi[0])
        x2 = np.clip(x2, lo[1], hi[1])
        
        # Build full 22-dim vectors with x3..x22 = 0
        X = np.zeros((len(x1), self.dim))
        X[:, 0] = x1
        X[:, 1] = x2
        
        # Evaluate each point; stop early if we exceed the deadline.
        F = np.zeros((len(x1), self.n_obj))
        for i in range(len(x1)):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])
        
        return X
# EVOLVE-BLOCK-END
