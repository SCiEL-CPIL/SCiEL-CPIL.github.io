import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        limit_time = 320  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Archive for non-dominated solutions
        archive_X = []
        archive_F = []
        
        # Helper to add to archive
        def add_to_archive(x, f):
            nonlocal archive_X, archive_F
            if len(archive_F) > 0:
                existing_F = np.array(archive_F)
                # Check if x is dominated by any existing solution
                dominated_mask = np.all(existing_F <= f, axis=1) & np.any(existing_F < f, axis=1)
                if np.any(dominated_mask):
                    return False
                
                # Remove existing solutions dominated by new solution
                dominated_by_new_mask = np.all(f <= existing_F, axis=1) & np.any(f < existing_F, axis=1)
                if np.any(dominated_by_new_mask):
                    indices_to_keep = np.where(~dominated_by_new_mask)[0]
                    archive_X = [archive_X[i] for i in indices_to_keep]
                    archive_F = [archive_F[i] for i in indices_to_keep]
            
            archive_X.append(x)
            archive_F.append(f)
            return True

        used = 0

        # Phase 1: Latin Hypercube Sampling for broad coverage
        # We use a portion of the budget for LHS to ensure we explore the space well.
        # 200 evaluations total. Let's use 100 for LHS.
        n_lhs = 100
        
        # Generate LHS points
        # For each dimension, we split the range into n_lhs intervals.
        # We permute the indices for each dimension to ensure one point per interval per dim.
        # This creates a set of points that cover the space well.
        
        # Create the grid
        lhs_points = np.zeros((n_lhs, self.dim))
        for d in range(self.dim):
            # Create intervals [0, 1/n], [1/n, 2/n], ...
            # Pick a random point in each interval
            # Then permute the order
            intervals = np.linspace(0, 1, n_lhs + 1)
            # Random point in each interval
            points_in_intervals = (intervals[:-1] + np.random.rand(n_lhs) * (1.0 / n_lhs))
            # Permute
            permuted = np.random.permutation(points_in_intervals)
            lhs_points[:, d] = permuted
        
        # Map to bounds
        lhs_points = lo + lhs_points * span
        
        # Evaluate LHS points
        for x in lhs_points:
            if used >= self.budget:
                break
            f = problem(x)
            used += 1
            add_to_archive(x, f)
            
            if time.time() > limit_time:
                return archive_X

        # Phase 2: Local search / refinement
        # We have budget - n_lhs left.
        # We will perform a simple local search around the archive points.
        # Or, we can use a differential evolution style approach.
        
        # Let's try a simple strategy:
        # Pick a random point from the archive.
        # Generate a mutation using a random direction and step size.
        # If it's non-dominated, add it.
        
        # To make it more effective, let's use a population-based approach with a small population.
        # But since we already have a good archive, let's just refine.
        
        # Alternative: Use the remaining budget to do a random search with a smaller step size,
        # biased towards the archive.
        
        # Let's try a simple local search:
        # For each remaining evaluation:
        # 1. Pick a random point from the archive.
        # 2. Generate a new point by adding Gaussian noise.
        # 3. Clip to bounds.
        # 4. Evaluate and add to archive.
        
        # This is essentially a random walk around the best solutions.
        # To avoid getting stuck, we occasionally do a global jump.
        
        while used < self.budget:
            if time.time() > limit_time:
                break
            
            # 10% chance of global exploration
            if np.random.rand() < 0.1 and len(archive_X) > 0:
                # Global jump: pick a random point in the space
                x = lo + np.random.rand(self.dim) * span
            else:
                # Local refinement: pick a random point from the archive and add noise
                if len(archive_X) == 0:
                    x = lo + np.random.rand(self.dim) * span
                else:
                    idx = np.random.randint(0, len(archive_X))
                    parent = archive_X[idx]
                    # Gaussian noise
                    # Step size: start small, maybe increase slightly?
                    # Let's use a fixed small step relative to the span.
                    # 0.05 * span seems reasonable.
                    noise = np.random.randn(self.dim) * 0.05 * span
                    x = parent + noise
            
            x = np.clip(x, lo, hi)
            f = problem(x)
            used += 1
            add_to_archive(x, f)

        return archive_X
