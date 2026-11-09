"""Rewritten program using Evolutionary Strategy with External Archive for multi-objective optimization."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-objective optimizer using Evolutionary Strategy with External Archive."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        budget = self.budget
        
        # Archive of non-dominated solutions
        # Each entry is (x, f)
        archive_x = []
        archive_f = []
        
        used = 0
        
        def is_dominated(f1, f2):
            """Check if f1 dominates f2. f1 <= f2 and f1 != f2."""
            if np.all(f1 <= f2) and np.any(f1 < f2):
                return True
            return False

        def add_to_archive(x, f):
            """Add point to archive if non-dominated, removing dominated points."""
            non_local = True
            # Check if dominated by any existing point
            for ef in archive_f:
                if is_dominated(ef, f):
                    non_local = False
                    break
            
            if not non_local:
                return False
            
            # Remove points dominated by new point
            keep_indices = []
            for i, ef in enumerate(archive_f):
                if is_dominated(f, ef):
                    # New point dominates existing point i
                    continue
                keep_indices.append(i)
            
            # If we removed some, update lists
            if len(keep_indices) < len(archive_f):
                archive_x[:] = [archive_x[i] for i in keep_indices]
                archive_f[:] = [archive_f[i] for i in keep_indices]
            
            archive_x.append(x)
            archive_f.append(f)
            return True

        def prune_archive(max_size=50):
            """Prune archive if it exceeds max_size using crowding distance."""
            if len(archive_f) <= max_size:
                return
            
            # Calculate crowding distance
            # Normalize objectives to [0, 1] for distance calculation
            f_arr = np.array(archive_f)
            f_min = f_arr.min(axis=0)
            f_max = f_arr.max(axis=0)
            f_range = f_max - f_min
            f_range[f_range == 0] = 1.0
            f_norm = (f_arr - f_min) / f_range
            
            n = len(f_norm)
            crowd = np.zeros(n)
            
            for m in range(self.n_obj):
                # Sort by objective m
                order = np.argsort(f_norm[:, m])
                crowd[order[0]] = np.inf
                crowd[order[-1]] = np.inf
                for i in range(1, n - 1):
                    idx_prev = order[i-1]
                    idx_next = order[i+1]
                    dist = f_norm[idx_next, m] - f_norm[idx_prev, m]
                    crowd[order[i]] += dist / (f_range[m] if f_range[m] != 0 else 1.0)
            
            # Remove point with smallest crowding distance
            # We need to remove until size <= max_size
            while len(archive_f) > max_size:
                min_crowd_idx = np.argmin(crowd)
                archive_x.pop(min_crowd_idx)
                archive_f.pop(min_crowd_idx)
                # Recalculate crowd or just update? For small sizes, recalc is fine.
                if len(archive_f) > max_size:
                    f_arr = np.array(archive_f)
                    f_min = f_arr.min(axis=0)
                    f_max = f_arr.max(axis=0)
                    f_range = f_max - f_min
                    f_range[f_range == 0] = 1.0
                    f_norm = (f_arr - f_min) / f_range
                    n = len(f_norm)
                    crowd = np.zeros(n)
                    for m in range(self.n_obj):
                        order = np.argsort(f_norm[:, m])
                        if n > 2:
                            crowd[order[0]] = np.inf
                            crowd[order[-1]] = np.inf
                            for i in range(1, n - 1):
                                idx_prev = order[i-1]
                                idx_next = order[i+1]
                                dist = f_norm[idx_next, m] - f_norm[idx_prev, m]
                                crowd[order[i]] += dist / (f_range[m] if f_range[m] != 0 else 1.0)
                        else:
                            crowd[:] = np.inf

        # Phase 1: Initial Population
        n_init = min(30, budget // 5)
        if n_init < 10:
            n_init = 10
            
        for _ in range(n_init):
            x = lo + rng.random(dim) * span
            f = problem(x)
            used += 1
            add_to_archive(x, f)
            
        prune_archive(50)

        # Phase 2: Evolutionary Loop
        # Adaptive step size
        base_sigma = 0.2
        min_sigma = 0.01
        
        while used < budget:
            # Select parent
            if len(archive_x) == 0:
                # Fallback
                x = lo + rng.random(dim) * span
            else:
                # Select parent with largest crowding distance? Or random?
                # Random selection from archive is simple.
                # To encourage exploration, let's pick the point with the largest "gap" to neighbors.
                # But that's expensive. Let's just pick a random point from the archive.
                # Or, pick the point that is furthest from the centroid of the archive in objective space.
                if len(archive_f) > 1:
                    f_arr = np.array(archive_f)
                    centroid = f_arr.mean(axis=0)
                    dists = np.linalg.norm(f_arr - centroid, axis=1)
                    # Pick the one furthest from centroid to promote diversity
                    parent_idx = np.argmax(dists)
                else:
                    parent_idx = 0
                
                parent_x = archive_x[parent_idx].copy()
            
            # Adapt step size
            progress = used / budget
            sigma = base_sigma * (1.0 - progress) + min_sigma * progress
            # Occasionally take a larger step to escape local optima
            if rng.random() < 0.1:
                sigma *= 3.0
                
            # Generate child
            mutation = rng.standard_normal(dim) * sigma
            child_x = parent_x + mutation
            child_x = np.clip(child_x, lo, hi)
            
            # Evaluate
            child_f = problem(child_x)
            used += 1
            
            # Add to archive
            add_to_archive(child_x, child_f)
            
            # Prune if necessary
            if len(archive_f) > 50:
                prune_archive(50)

        # Return all evaluated points? The task says "Every returned objective vector is recorded".
        # The return value is ignored, but we should return something valid.
        # Usually, we return the archive or the best points.
        # Since the evaluator records *all* calls to problem(), the return value doesn't matter for scoring,
        # but we must return *something* if the interface expects it.
        # The prompt says "The return value of __call__ is ignored".
        # So we can return the archive points.
        return archive_x

# EVOLVE-BLOCK-END
