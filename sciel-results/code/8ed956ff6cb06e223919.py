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
        self.ref_point = np.array([3.0, 5.0, 7.0])

    def _hypervolume_contribution(self, f, archive_F):
        """Compute hypervolume contribution of point f to archive.
        
        For 3 objectives with reference point [3, 5, 7], the contribution
        of a point is the volume it adds to the dominated region.
        """
        if len(archive_F) == 0:
            # Volume dominated by f alone
            if any(f[i] >= self.ref_point[i] for i in range(self.n_obj)):
                return 0.0
            return np.prod(np.maximum(self.ref_point - f, 0))

        # Check if f is dominated by any point in archive
        for af in archive_F:
            if np.all(af <= f) and np.any(af < f):
                return 0.0

        # Compute volume dominated by f but not by any archive point
        # This is complex; use a simpler approximation:
        # Volume dominated by f minus volume dominated by f AND by archive
        vol_f = np.prod(np.maximum(self.ref_point - f, 0)) if all(f[i] < self.ref_point[i] for i in range(self.n_obj)) else 0.0
        
        # Subtract overlapping volume with archive points
        # For simplicity, approximate by checking against each archive point
        overlap = 0.0
        for af in archive_F:
            if any(f[i] >= self.ref_point[i] for i in range(self.n_obj)):
                break
            # Overlap region is dominated by both f and af
            upper = np.minimum(self.ref_point, np.maximum(f, af))
            lower = np.minimum(f, af)
            if all(upper[i] > lower[i] for i in range(self.n_obj)):
                overlap = max(overlap, np.prod(upper - lower))
        
        return max(0.0, vol_f - overlap)

    def _ternary_search_dim(self, problem, bounds, x, dim_idx, archive_F, max_evals):
        """Perform ternary search along one dimension to maximize hypervolume contribution."""
        lo = bounds[dim_idx][0]
        hi = bounds[dim_idx][1]
        
        # Initial evaluations at 1/3 and 2/3 points
        m1 = lo + (hi - lo) / 3.0
        m2 = lo + 2.0 * (hi - lo) / 3.0
        
        x1 = x.copy()
        x2 = x.copy()
        x1[dim_idx] = m1
        x2[dim_idx] = m2
        
        f1 = problem(x1)
        f2 = problem(x2)
        
        hv1 = self._hypervolume_contribution(f1, archive_F)
        hv2 = self._hypervolume_contribution(f2, archive_F)
        
        evals_used = 2
        tol = 1e-6 * (hi - lo)
        
        while (hi - lo) > tol and evals_used < max_evals:
            if hv1 < hv2:
                lo = m1
                m1 = m2
                f1 = f2
                hv1 = hv2
                m1 = lo + (hi - lo) / 3.0
                m2 = lo + 2.0 * (hi - lo) / 3.0
                x1 = x.copy()
                x1[dim_idx] = m1
                f1 = problem(x1)
                hv1 = self._hypervolume_contribution(f1, archive_F)
                evals_used += 1
            else:
                hi = m2
                m2 = m1
                f2 = f1
                hv2 = hv1
                m1 = lo + (hi - lo) / 3.0
                m2 = lo + 2.0 * (hi - lo) / 3.0
                x2 = x.copy()
                x2[dim_idx] = m2
                f2 = problem(x2)
                hv2 = self._hypervolume_contribution(f2, archive_F)
                evals_used += 1
        
        # Return best point found
        if hv1 >= hv2:
            return x1, f1, hv1, evals_used
        else:
            return x2, f2, hv2, evals_used

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random  # Use evaluator's seeded RNG

        dim = self.dim
        budget = self.budget
        used = 0

        # Archive of non-dominated points
        archive_X = []
        archive_F = []

        def add_to_archive(x, f):
            """Add point to archive if it's non-dominated and not dominated."""
            nonlocal used
            # Check if f is dominated by any archive point
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    return False
            # Remove archive points dominated by f
            new_archive_X = []
            new_archive_F = []
            for ax, af in zip(archive_X, archive_F):
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive_X.append(ax)
                    new_archive_F.append(af)
            new_archive_X.append(x)
            new_archive_F.append(f)
            archive_X[:] = new_archive_X
            archive_F[:] = new_archive_F
            return True

        # Phase 1: Diverse initialization
        n_init = min(30, budget // 4)
        for _ in range(n_init):
            if used >= budget or time.time() > deadline:
                break
            x = lo + (hi - lo) * rng.random_sample(dim)
            f = problem(x)
            used += 1
            add_to_archive(x.copy(), f.copy())

        # Phase 2: Coordinate descent with ternary search
        # Iterate over archive points and improve each via coordinate descent
        max_iter = 20
        for iteration in range(max_iter):
            if time.time() > deadline or used >= budget:
                break
            
            # Get current archive size
            n_arch = len(archive_F)
            if n_arch == 0:
                break
            
            # For each point in archive, try to improve via coordinate descent
            # Process in random order to avoid bias
            order = rng.permutation(n_arch)
            
            improved = False
            for idx in order:
                if time.time() > deadline or used >= budget:
                    break
                
                x = archive_X[idx].copy()
                f = archive_F[idx].copy()
                
                # Try each dimension
                for d in range(dim):
                    if time.time() > deadline or used >= budget:
                        break
                    
                    # Reserve some evaluations for this search
                    max_evals = min(10, budget - used)
                    if max_evals < 3:
                        continue
                    
                    x_new, f_new, hv_new, evals_used = self._ternary_search_dim(
                        problem, bounds, x, d, archive_F, max_evals
                    )
                    used += evals_used
                    
                    # Check if improvement
                    hv_old = self._hypervolume_contribution(f, [af for i, af in enumerate(archive_F) if i != idx])
                    if hv_new > hv_old + 1e-8:
                        # Replace the point
                        archive_X[idx] = x_new
                        archive_F[idx] = f_new
                        improved = True
                        break  # Restart from improved point

            if not improved and iteration > 3:
                # No improvement, try adding new random points
                n_new = min(5, budget - used)
                for _ in range(n_new):
                    if used >= budget or time.time() > deadline:
                        break
                    x = lo + (hi - lo) * rng.random_sample(dim)
                    f = problem(x)
                    used += 1
                    add_to_archive(x.copy(), f.copy())

        # Phase 3: Final polish - small perturbations around best points
        if used < budget and time.time() < deadline:
            n_polish = min(20, budget - used)
            for _ in range(n_polish):
                if used >= budget or time.time() > deadline:
                    break
                
                # Pick a random archive point
                idx = rng.randint(len(archive_F))
                x = archive_X[idx].copy()
                
                # Small Gaussian perturbation
                sigma = 0.05 * (hi - lo)
                x_pert = x + sigma * rng.standard_normal(dim)
                x_pert = np.clip(x_pert, lo, hi)
                
                f_pert = problem(x_pert)
                used += 1
                add_to_archive(x_pert, f_pert)

        # Return the archive (will be ignored, but points are recorded)
        if len(archive_X) > 0:
            return np.array(archive_X)
        else:
            return np.array([lo + (hi - lo) * 0.5])
# EVOLVE-BLOCK-END
