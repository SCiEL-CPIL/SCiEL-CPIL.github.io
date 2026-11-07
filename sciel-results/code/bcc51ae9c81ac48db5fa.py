"""Multi-objective optimizer for WFG4 using hypervolume-guided local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-guided coordinate descent with diverse restarts for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hypervolume_contribution(self, f, archive_F):
        """Compute hypervolume contribution of point f to archive.
        
        For 3 objectives with reference point [3, 5, 7], the contribution
        of a point is the volume it adds to the dominated region.
        """
        ref = np.array([3.0, 5.0, 7.0])
        if np.any(f >= ref):
            return 0.0
        
        # A point contributes if it's not dominated by any archive point
        # and its dominated region isn't fully covered
        # Simplified: compute the box from f to ref, subtract overlaps with archive
        # For efficiency, just check if f is non-dominated in archive + f
        
        # Quick dominance check against archive
        for af in archive_F:
            if np.all(af <= f) and np.any(af < f):
                return 0.0  # dominated by archive
        
        # Compute volume contribution (approximate for 3D)
        # The contribution is the volume of the box [f, ref] minus parts covered by archive
        # For simplicity, use the full box volume as upper bound
        # This is an overestimate but preserves ordering for selection
        box_vol = np.prod(ref - f)
        
        # Subtract approximate overlap with archive points that are better in at least one obj
        # For a more accurate computation in 3D:
        # Contribution = volume not dominated by any archive point
        # Use inclusion-exclusion or just return box_vol as proxy
        
        return box_vol

    def _is_non_dominated(self, f, archive_F):
        """Check if f is non-dominated with respect to archive_F."""
        for af in archive_F:
            if np.all(af <= f) and np.any(af < f):
                return False
        return True

    def _update_archive(self, archive_X, archive_F, x, f):
        """Add point to archive if non-dominated, remove dominated points."""
        if not self._is_non_dominated(f, archive_F):
            return archive_X, archive_F, False
        
        # Remove points dominated by new point
        new_X = []
        new_F = []
        for i in range(len(archive_X)):
            if np.all(f <= archive_F[i]) and np.any(f < archive_F[i]):
                continue  # dominated by new point
            new_X.append(archive_X[i])
            new_F.append(archive_F[i])
        
        new_X.append(x)
        new_F.append(f)
        return np.array(new_X), np.array(new_F), True

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random  # Use evaluator's seeded RNG

        dim = self.dim
        budget = self.budget
        used = 0

        # Archive of non-dominated solutions
        archive_X = []
        archive_F = []

        # Phase 1: Generate diverse initial points (use ~40% of budget)
        n_init = min(80, budget // 4)
        
        for i in range(n_init):
            if time.time() > deadline or used >= budget:
                break
            x = lo + (hi - lo) * rng.random_sample(dim)
            f = np.array(problem(x), dtype=np.float64)
            used += 1
            archive_X, archive_F, _ = self._update_archive(archive_X, archive_F, x, f)
        
        archive_X = np.array(archive_X) if archive_X else np.empty((0, dim))
        archive_F = np.array(archive_F) if archive_F else np.empty((0, self.n_obj))

        # Phase 2: Hypervolume-guided local search on archive points
        # For each archive point, try coordinate-wise perturbations
        max_iterations = 500
        step_sizes = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
        
        iteration = 0
        while used < budget and time.time() < deadline and iteration < max_iterations:
            iteration += 1
            
            if len(archive_X) == 0:
                # No archive yet, generate random point
                x = lo + (hi - lo) * rng.random_sample(dim)
                f = np.array(problem(x), dtype=np.float64)
                used += 1
                archive_X, archive_F, _ = self._update_archive(archive_X, archive_F, x, f)
                archive_X = np.array(archive_X) if archive_X else np.empty((0, dim))
                archive_F = np.array(archive_F) if archive_F else np.empty((0, self.n_obj))
                continue

            # Select a point from archive to refine
            # Prefer points with higher hypervolume contribution
            if len(archive_F) > 1:
                # Compute approximate HV contribution for each archive point
                # Use simple heuristic: prefer points that are more "central" in the front
                # or have larger potential improvement
                # For now, cycle through archive points
                idx = iteration % len(archive_X)
            else:
                idx = 0
            
            base_x = archive_X[idx].copy()
            base_f = archive_F[idx].copy()
            
            # Try coordinate-wise perturbations
            best_x = base_x
            best_f = base_f
            improved = False
            
            for d in range(dim):
                if used >= budget or time.time() > deadline:
                    break
                
                # Try several step sizes in both directions
                for step in step_sizes:
                    for direction in [1, -1]:
                        if used >= budget or time.time() > deadline:
                            break
                        
                        x_try = base_x.copy()
                        x_try[d] = base_x[d] + direction * step * (hi[d] - lo[d])
                        x_try = np.clip(x_try, lo, hi)
                        
                        # Skip if too close to base
                        if np.allclose(x_try, base_x, atol=1e-10):
                            continue
                        
                        f_try = np.array(problem(x_try), dtype=np.float64)
                        used += 1
                        
                        # Check if this is an improvement (non-dominated and adds HV)
                        if self._is_non_dominated(f_try, archive_F):
                            # Check if it improves over base
                            if np.any(f_try < base_f) and not np.any(f_try > base_f):
                                best_x = x_try
                                best_f = f_try
                                improved = True
                            elif np.any(f_try < base_f):
                                # Mixed improvement, still consider it
                                best_x = x_try
                                best_f = f_try
                                improved = True
                            break  # Found improvement in this direction, try next dimension
                    else:
                        continue
                    break  # Found improvement, break out of step loop
            
            # Update archive with best found
            if improved:
                archive_X, archive_F, added = self._update_archive(archive_X, archive_F, best_x, best_f)
                archive_X = np.array(archive_X) if archive_X else np.empty((0, dim))
                archive_F = np.array(archive_F) if archive_F else np.empty((0, self.n_obj))
            
            # Occasionally add a new random point to maintain diversity
            if iteration % 10 == 0 and used < budget - 20:
                x_new = lo + (hi - lo) * rng.random_sample(dim)
                f_new = np.array(problem(x_new), dtype=np.float64)
                used += 1
                archive_X, archive_F, _ = self._update_archive(archive_X, archive_F, x_new, f_new)
                archive_X = np.array(archive_X) if archive_X else np.empty((0, dim))
                archive_F = np.array(archive_F) if archive_F else np.empty((0, self.n_obj))

        return archive_X if len(archive_X) > 0 else np.empty((1, dim))
# EVOLVE-BLOCK-END
