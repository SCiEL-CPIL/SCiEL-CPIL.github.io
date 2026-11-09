"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-Objective Particle Swarm Optimization (MOPSO) for ZDT6."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        # MOPSO Parameters
        pop_size = 20
        archive_size = 50
        w = 0.7  # Inertia weight
        c1 = 1.5  # Cognitive coefficient
        c2 = 1.5  # Social coefficient
        v_max = 0.2 * (hi - lo)  # Max velocity per dimension
        
        # Initialize Population
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        V = np.zeros((pop_size, dim))
        
        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Personal Bests
        Pbest_X = X.copy()
        Pbest_F = F.copy()
        
        # External Archive
        # Store (x, f) pairs
        archive_X = []
        archive_F = []
        
        def dominates(f1, f2):
            """Check if f1 dominates f2 (minimization)."""
            return np.all(f1 <= f2) and np.any(f1 < f2)
            
        def update_archive(new_x, new_f):
            """Add new solution to archive if non-dominated."""
            non_dominated = True
            for i in range(len(archive_F)):
                if dominates(archive_F[i], new_f):
                    non_dominated = False
                    break
            
            if not non_dominated:
                return
            
            # Remove dominated solutions from archive
            new_archive_X = []
            new_archive_F = []
            for i in range(len(archive_F)):
                if not dominates(new_f, archive_F[i]):
                    new_archive_X.append(archive_X[i])
                    new_archive_F.append(archive_F[i])
            
            new_archive_X.append(new_x)
            new_archive_F.append(new_f)
            
            # If archive exceeds size, remove dominated or use grid-based truncation
            # For simplicity and speed, we just keep the non-dominated set.
            # If it grows too large, we can implement a grid-based truncation.
            # Given budget 200, archive size should be manageable.
            # However, to be safe against memory issues or slow selection, we limit it.
            if len(new_archive_F) > archive_size:
                # Simple truncation: keep the 'best' based on a random selection from non-dominated
                # or use a grid. Let's use a simple grid-based selection for gbest.
                # For now, just keep the list. The selection of gbest will handle diversity.
                pass
            
            archive_X[:] = new_archive_X
            archive_F[:] = new_archive_F

        # Initialize archive
        for i in range(pop_size):
            update_archive(X[i], F[i])
            
        def select_gbest():
            """Select gbest from archive using grid-based crowding."""
            if not archive_F:
                return Pbest_X[0]
            
            # If only one point
            if len(archive_F) == 1:
                return archive_X[0]
            
            # Define grid
            # We need to map objective values to grid indices.
            # We don't know the exact range, so we use min/max of current archive.
            f_min = np.min(archive_F, axis=0)
            f_max = np.max(archive_F, axis=0)
            
            # Avoid division by zero
            f_range = f_max - f_min
            f_range[f_range == 0] = 1.0
            
            # Number of hypercubes per dimension
            # For 2 objectives, 10x10 grid is reasonable
            n_grid = 10
            grid_counts = np.zeros((n_grid, n_grid), dtype=int)
            grid_points = {}
            
            for i in range(len(archive_F)):
                # Normalize to [0, n_grid-1]
                idxs = []
                for k in range(self.n_obj):
                    norm_val = (archive_F[i][k] - f_min[k]) / f_range[k]
                    idx = int(norm_val * (n_grid - 1))
                    if idx >= n_grid:
                        idx = n_grid - 1
                    idxs.append(idx)
                
                key = tuple(idxs)
                grid_counts[idxs[0], idxs[1]] += 1
                if key not in grid_points:
                    grid_points[key] = []
                grid_points[key].append(i)
            
            # Find the cell with the minimum count
            min_count = np.min(grid_counts)
            # Get all cells with min count
            candidates = []
            for i in range(n_grid):
                for j in range(n_grid):
                    if grid_counts[i, j] == min_count:
                        for idx in grid_points.get((i, j), []):
                            candidates.append(idx)
            
            if not candidates:
                return archive_X[0]
            
            # Randomly select from candidates
            gbest_idx = candidates[rng.integers(0, len(candidates))]
            return archive_X[gbest_idx]

        # Main Loop
        while used < budget:
            gbest = select_gbest()
            
            for i in range(pop_size):
                if used >= budget:
                    break
                    
                r1 = rng.random(dim)
                r2 = rng.random(dim)
                
                # Update velocity
                V[i] = w * V[i] + c1 * r1 * (Pbest_X[i] - X[i]) + c2 * r2 * (gbest - X[i])
                
                # Clip velocity
                V[i] = np.clip(V[i], -v_max, v_max)
                
                # Update position
                X[i] = X[i] + V[i]
                
                # Clip position to bounds
                X[i] = np.clip(X[i], lo, hi)
                
                # Evaluate
                F_new = problem(X[i])
                used += 1
                
                # Update Personal Best
                if dominates(F_new, Pbest_F[i]):
                    Pbest_F[i] = F_new
                    Pbest_X[i] = X[i].copy()
                elif not dominates(Pbest_F[i], F_new):
                    # Non-dominated: Keep old Pbest to maintain stability, 
                    # or replace with probability? 
                    # Standard MOPSO often keeps the non-dominated one. 
                    # If we replace, we might lose the "best" so far.
                    # Let's keep the old one if it's non-dominated to be conservative,
                    # unless we want more exploration. 
                    # Actually, if they are non-dominated, both are valid.
                    # We'll stick to the old one to avoid noise in Pbest.
                    pass
                
                # Update Global Archive
                update_archive(X[i], F_new)

        return X
# EVOLVE-BLOCK-END
