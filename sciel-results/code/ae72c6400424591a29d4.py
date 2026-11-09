"""Multi-objective optimizer for RE37 using hypervolume-driven adaptive sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-driven adaptive sampling optimizer for RE37 (4 vars, 3 objectives)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _dominates(self, a, b):
        """Check if a dominates b (a <= b in all objectives, a < b in at least one)."""
        return np.all(a <= b) and np.any(a < b)

    def _marginal_hv_contribution(self, f_new, archive_F, ref):
        """
        Estimate marginal hypervolume contribution of f_new to the archive.
        Uses a simplified 3D HV calculation: the volume of the box defined by
        f_new and ref that is not already covered by the archive.
        """
        # Box dimensions from f_new to ref
        box_dims = ref - f_new
        if np.any(box_dims <= 0):
            return 0.0  # f_new is not better than ref in all objectives
        
        # Total box volume
        total_vol = np.prod(box_dims)
        
        # Subtract volume already covered by archive points that dominate f_new
        # or are in the same region
        covered_vol = 0.0
        for f_arch in archive_F:
            # Check if f_arch is within the box (f_arch <= f_new is wrong; we need overlap)
            # The box is [f_new, ref]. A point f_arch covers part of this box if
            # f_arch <= f_new in all objectives (i.e., f_arch dominates f_new)
            # But actually, we want to subtract the volume that f_arch "claims"
            # within the box [f_new, ref].
            # A point f_arch with f_arch <= ref contributes to HV as the box [f_arch, ref].
            # The overlap between [f_new, ref] and [f_arch, ref] is [max(f_new, f_arch), ref]
            # if max(f_new, f_arch) < ref in all dimensions.
            overlap_lower = np.maximum(f_new, f_arch)
            overlap_dims = ref - overlap_lower
            if np.all(overlap_dims > 0):
                covered_vol += np.prod(overlap_dims)
            # Note: This over-counts if multiple archive points overlap, but it's a heuristic
        
        # Simple estimate: total_vol minus average covered (crude but fast)
        # For a better estimate, we'd need proper union-of-boxes, but that's expensive.
        # Use a simple heuristic: if many points are close, the marginal contribution is small.
        # Count how many archive points are "close" to f_new
        n_close = 0
        for f_arch in archive_F:
            if np.all(f_arch <= ref):  # Only consider valid points
                dist = np.max(np.abs(f_new - f_arch))
                if dist < 0.1:
                    n_close += 1
        
        # Penalize if too many close points
        penalty = min(1.0, n_close / 10.0)
        
        return total_vol * (1.0 - penalty)

    def _hypervolume_driven_proposal(self, archive_X, archive_F, bounds, ref):
        """
        Propose a new point by maximizing expected marginal hypervolume contribution.
        Uses a combination of:
        1. Interpolation between pairs of non-dominated points (to fill gaps)
        2. Local perturbations of existing points (to refine the front)
        3. Random sampling in under-explored regions
        """
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        n_arch = len(archive_F)
        if n_arch == 0:
            return lo + (hi - lo) * np.random.random(dim)
        
        # Strategy 1: Interpolate between pairs of non-dominated points
        # This fills gaps in the objective space
        if n_arch >= 2 and np.random.random() < 0.5:
            # Pick two random archive points
            i1 = np.random.randint(n_arch)
            i2 = np.random.randint(n_arch)
            if i1 == i2:
                i2 = (i1 + 1) % n_arch
            
            # Interpolate in decision space with some noise
            alpha = np.random.random()
            child = alpha * archive_X[i1] + (1 - alpha) * archive_X[i2]
            # Add small noise to avoid exact duplicates
            noise_scale = 0.02
            child += np.random.randn(dim) * noise_scale * (hi - lo)
            child = np.clip(child, lo, hi)
            return child
        
        # Strategy 2: Local perturbation of a random archive point
        elif np.random.random() < 0.8:
            i = np.random.randint(n_arch)
            child = archive_X[i].copy()
            # Perturb 1-2 dimensions
            n_perturb = np.random.randint(1, 3)
            dims_to_perturb = np.random.choice(dim, size=n_perturb, replace=False)
            for d in dims_to_perturb:
                step = np.random.uniform(0.01, 0.08) * (hi[d] - lo[d])
                direction = np.random.choice([-1, 1])
                child[d] += direction * step
            child = np.clip(child, lo, hi)
            return child
        
        # Strategy 3: Random sampling
        else:
            return lo + (hi - lo) * np.random.random(dim)

    def _update_archive(self, archive_X, archive_F, x_new, f_new, ref):
        """
        Update the archive with a new point if it is non-dominated.
        Returns updated archive and a boolean indicating if the point was added.
        """
        n_obj = self.n_obj
        
        # Check if f_new is better than ref in all objectives
        if not np.all(f_new < ref):
            return archive_X, archive_F, False
        
        # Check if f_new is dominated by any archive point
        for f_arch in archive_F:
            if np.all(f_arch <= f_new) and np.any(f_arch < f_new):
                return archive_X, archive_F, False
        
        # Remove archive points dominated by f_new
        new_X_list = []
        new_F_list = []
        for x_arch, f_arch in zip(archive_X, archive_F):
            if not (np.all(f_new <= f_arch) and np.any(f_new < f_arch)):
                new_X_list.append(x_arch)
                new_F_list.append(f_arch)
        
        # Add the new point
        new_X_list.append(x_new)
        new_F_list.append(f_new)
        
        return np.array(new_X_list), np.array(new_F_list), True

    def _local_refinement(self, problem, bounds, x_center, f_center, ref, budget_remaining, deadline):
        """
        Perform local refinement around a point using multi-scale search.
        Returns the best point found and the number of evaluations used.
        """
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        
        best_x = x_center.copy()
        best_f = f_center.copy()
        used = 0
        
        # Multi-scale local search
        step_sizes = [0.05, 0.03, 0.02, 0.01, 0.005]
        
        for step_frac in step_sizes:
            if used >= budget_remaining or time.time() > deadline:
                break
            
            # Try perturbations in each dimension
            for d in range(dim):
                if used >= budget_remaining or time.time() > deadline:
                    break
                
                for direction in [-1, 1]:
                    if used >= budget_remaining or time.time() > deadline:
                        break
                    
                    trial_x = best_x.copy()
                    step = step_frac * (hi[d] - lo[d])
                    trial_x[d] += direction * step
                    trial_x = np.clip(trial_x, lo, hi)
                    
                    # Skip if too close to current best
                    if np.allclose(trial_x, best_x, atol=1e-6):
                        continue
                    
                    trial_f = np.array(problem(trial_x))
                    used += 1
                    
                    # Check if trial dominates best
                    if np.all(trial_f <= best_f) and np.any(trial_f < best_f):
                        best_x = trial_x.copy()
                        best_f = trial_f.copy()
        
        return best_x, best_f, used

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # 300s limit, 15s safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj
        ref = np.array([1.1, 1.3, 1.1])

        # Initialize archive with a diverse set of random points
        init_size = 20
        archive_X = lo + (hi - lo) * np.random.random((init_size, dim))
        archive_F = np.array([problem(x) for x in archive_X])
        used = init_size

        # Reserve budget for local refinement
        reserve_fraction = 0.3
        reserve = int(budget * reserve_fraction)
        exploration_budget = budget - reserve

        # Main loop: hypervolume-driven adaptive sampling
        while used < exploration_budget:
            if time.time() > deadline:
                break

            # Propose a new point
            x_new = self._hypervolume_driven_proposal(archive_X, archive_F, bounds, ref)
            f_new = np.array(problem(x_new))
            used += 1

            # Update archive
            archive_X, archive_F, added = self._update_archive(archive_X, archive_F, x_new, f_new, ref)

            # If the point was added to the archive, consider local refinement
            if added and used < exploration_budget and time.time() < deadline:
                # Spend some budget on local refinement around this new point
                refinement_budget = min(5, exploration_budget - used)
                if refinement_budget > 0:
                    best_x, best_f, used_ref = self._local_refinement(
                        problem, bounds, x_new, f_new, ref, refinement_budget, deadline
                    )
                    used += used_ref
                    
                    # Update archive with the refined point
                    archive_X, archive_F, _ = self._update_archive(
                        archive_X, archive_F, best_x, best_f, ref
                    )

        # Final refinement phase: polish the best points in the archive
        if used < budget and time.time() < deadline:
            # Sort archive by hypervolume contribution (approximate)
            # Points with larger gaps to neighbors are more valuable
            n_arch = len(archive_F)
            if n_arch > 0:
                contributions = np.zeros(n_arch)
                for i in range(n_arch):
                    min_dist = np.inf
                    for j in range(n_arch):
                        if i == j:
                            continue
                        d = np.max(np.abs(archive_F[i] - archive_F[j]))
                        min_dist = min(min_dist, d)
                    contributions[i] = min_dist
                
                # Process points in order of contribution (most isolated first)
                order = np.argsort(-contributions)
                
                remaining_budget = budget - used
                budget_per_point = max(1, remaining_budget // n_arch)
                
                for oi in order:
                    if used >= budget or time.time() > deadline:
                        break
                    
                    idx = order[oi]
                    x_center = archive_X[idx]
                    f_center = archive_F[idx]
                    
                    # Local refinement
                    best_x, best_f, used_ref = self._local_refinement(
                        problem, bounds, x_center, f_center, ref, budget_per_point, deadline
                    )
                    used += used_ref
                    
                    # Update archive
                    archive_X, archive_F, _ = self._update_archive(
                        archive_X, archive_F, best_x, best_f, ref
                    )

        return archive_X
# EVOLVE-BLOCK-END
