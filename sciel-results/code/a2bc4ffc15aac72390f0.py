"""Multi-objective optimizer for RE21 combining stratified grid, hypervolume archive, and coordinate-descent polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-based multi-objective optimizer for RE21 with stratified init and local polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 275.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        ref_point = np.array([3000.0, 0.06])

        # Archive: list of (x, f) pairs that are non-dominated
        archive_X = []
        archive_F = []
        used = 0

        def dominates(a, b):
            """Check if a dominates b (a <= b in all objectives, a < b in at least one)."""
            return np.all(a <= b) and np.any(a < b)

        def compute_hypervolume_2d(points, ref_pt):
            """Compute exact hypervolume for 2D points with given reference point."""
            valid = [p for p in points if p[0] < ref_pt[0] and p[1] < ref_pt[1]]
            if not valid:
                return 0.0
            valid.sort(key=lambda p: p[0])
            hv = 0.0
            prev_y = ref_pt[1]
            for p in valid:
                if p[1] < prev_y:
                    hv += (ref_pt[0] - p[0]) * (prev_y - p[1])
                    prev_y = p[1]
            return hv

        def add_to_archive(x, f):
            """Add a point to the archive if it's not dominated. Remove dominated points."""
            nonlocal archive_X, archive_F
            if f[0] >= ref_point[0] or f[1] >= ref_point[1]:
                return False
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    return False
            new_X = []
            new_F = []
            for i, (ax, af) in enumerate(zip(archive_X, archive_F)):
                if not (np.all(f <= af) and np.any(f < af)):
                    new_X.append(ax)
                    new_F.append(af)
            new_X.append(x.copy())
            new_F.append(f.copy())
            archive_X = new_X
            archive_F = new_F
            return True

        def evaluate_and_add(x):
            """Evaluate x and add to archive if non-dominated."""
            nonlocal used
            f = problem(x)
            f = np.array(f, dtype=np.float64)
            used += 1
            add_to_archive(x.copy(), f)
            return f

        def get_hv_contributions():
            """Compute hypervolume contribution of each archive point in 2D."""
            n = len(archive_F)
            if n == 0:
                return np.array([])
            # Sort by f[0] ascending
            sorted_idx = np.argsort([f[0] for f in archive_F])
            sorted_F = [archive_F[i] for i in sorted_idx]
            
            contribs_sorted = np.zeros(n)
            for i in range(n - 1):
                # Width in f[0] direction
                w = sorted_F[i+1][0] - sorted_F[i][0]
                # Height in f[1] direction
                h = ref_point[1] - sorted_F[i][1]
                contribs_sorted[i] = max(0.0, w) * max(0.0, h)
            
            # Last point
            w = ref_point[0] - sorted_F[-1][0]
            h = ref_point[1] - sorted_F[-1][1]
            contribs_sorted[-1] = max(0.0, w) * max(0.0, h)
            
            # Map back to original indices
            contribs = np.zeros(n)
            for orig_idx, sort_idx in enumerate(sorted_idx):
                pass # Need to map correctly
            # Actually, simpler: create array of contributions in original order
            contribs = np.zeros(n)
            for i in range(n):
                orig_idx = sorted_idx[i]
                contribs[orig_idx] = contribs_sorted[i]
            return contribs

        def prune_archive_by_hv(max_size):
            """Prune archive to limit size by removing points with lowest HV contribution."""
            nonlocal archive_X, archive_F
            if len(archive_X) <= max_size:
                return
            
            while len(archive_X) > max_size and time.time() < deadline - 5:
                n = len(archive_X)
                if n <= 1:
                    break
                contribs = get_hv_contributions()
                min_idx = np.argmin(contribs)
                archive_X.pop(min_idx)
                archive_F.pop(min_idx)

        # Phase 1: Stratified initialization
        # Use a grid to ensure coverage. 4 dims, so 2^4 = 16 points for a coarse grid.
        # Or use random but stratified. Let's do a mix: 16 grid points + random fill.
        # Grid: 2 points per dimension -> 2^4 = 16 points.
        n_grid = 16
        grid_points = []
        for i in range(n_grid):
            bits = [(i >> j) & 1 for j in range(self.dim)]
            x = np.array([lo[j] + bits[j] * (hi[j] - lo[j]) for j in range(self.dim)])
            grid_points.append(x)
        
        # Shuffle grid points to avoid order bias
        np.random.shuffle(grid_points)
        
        for x in grid_points:
            if used >= self.budget:
                break
            evaluate_and_add(x)
            
        # Fill remaining initial budget with random points
        n_init_remaining = max(0, 30 - used)
        for i in range(n_init_remaining):
            x = lo + (hi - lo) * np.random.random(self.dim)
            evaluate_and_add(x)

        # Phase 2: Evolutionary search with hypervolume-based selection
        max_archive_size = 50
        
        while used < self.budget:
            if time.time() > deadline - 15:
                break
            
            if not archive_F:
                x = lo + (hi - lo) * np.random.random(self.dim)
                evaluate_and_add(x)
                continue
            
            # Select parent based on hypervolume contribution
            contribs = get_hv_contributions()
            if np.sum(contribs) > 0:
                probs = contribs / np.sum(contribs)
                # Ensure probs are valid
                if np.any(probs < 0):
                    probs = np.ones(len(archive_F)) / len(archive_F)
            else:
                probs = np.ones(len(archive_F)) / len(archive_F)
            
            idx = np.random.choice(len(archive_F), p=probs)
            parent = archive_X[idx]
            
            # Mutation
            progress = used / self.budget
            # Adaptive step size: start larger, decrease over time
            step_scale = 0.4 * (1.0 - progress) + 0.05
            sigma = step_scale * (hi - lo)
            
            child = parent + np.random.standard_normal(self.dim) * sigma
            child = np.clip(child, lo, hi)
            
            evaluate_and_add(child)
            
            # Prune archive if it gets too large
            if len(archive_X) > max_archive_size:
                prune_archive_by_hv(max_archive_size)
            
            # Occasionally add a random point to maintain diversity
            if used % 10 == 0 and used < self.budget - 5:
                x = lo + (hi - lo) * np.random.random(self.dim)
                evaluate_and_add(x)

        # Phase 3: Local search polish
        # Use remaining budget to polish the best points in the archive.
        # Select top K points by HV contribution and perform coordinate descent.
        
        if archive_F and used < self.budget:
            # Sort archive by HV contribution descending
            contribs = get_hv_contributions()
            sorted_indices = np.argsort(-contribs)
            
            # We will polish the top points, using a fraction of the remaining budget
            # Let's reserve 50% of the remaining budget for polish, or just use all if little left.
            # Actually, let's just iterate through the top 10 points and spend a few evals on each.
            
            top_k = min(10, len(archive_F))
            polish_budget_per_point = max(2, (self.budget - used) // (top_k * 4)) # 4 dims
            
            for k in range(top_k):
                if used >= self.budget or time.time() > deadline - 5:
                    break
                
                idx = sorted_indices[k]
                # Get current best x and f for this point
                # Note: archive indices might have shifted if we removed points? 
                # No, we don't remove during polish, only evaluate.
                # But wait, evaluate_and_add might add new points, shifting indices?
                # No, add_to_archive creates new lists. But we are iterating over sorted_indices 
                # which are indices into the CURRENT archive_F.
                # If we add points, the archive changes. We should copy the points first.
                
                x_current = archive_X[idx].copy()
                f_current = archive_F[idx].copy()
                
                # Coordinate descent
                for dim in range(self.dim):
                    if used >= self.budget or time.time() > deadline - 5:
                        break
                    
                    # Try +/- perturbations
                    # Step size for local search is small
                    step_size = 0.05 * (hi[dim] - lo[dim])
                    
                    # Try positive
                    x_test = x_current.copy()
                    x_test[dim] += step_size
                    if x_test[dim] <= hi[dim]:
                        f_test = problem(x_test)
                        f_test = np.array(f_test, dtype=np.float64)
                        used += 1
                        
                        # Check if this improves hypervolume
                        # Simple check: does it dominate current? Or is it non-dominated?
                        # For polish, we want to improve the HV.
                        # Let's just add to archive if it's non-dominated.
                        # But we want to update x_current if it's better in some way.
                        # Let's use a simple rule: if it dominates f_current, or is closer to ideal?
                        # Actually, just add to archive. If it's good, it will stay.
                        # And update x_current to the new point if it's non-dominated.
                        
                        # Check if dominated by f_current
                        if not (np.all(f_current <= f_test) and np.any(f_current < f_test)):
                            # Not dominated by current
                            # Check if it dominates current
                            if np.all(f_test <= f_current) and np.any(f_test < f_current):
                                x_current = x_test
                                f_current = f_test
                            # Add to archive
                            add_to_archive(x_test.copy(), f_test.copy())
                            
                    # Try negative
                    x_test = x_current.copy()
                    x_test[dim] -= step_size
                    if x_test[dim] >= lo[dim]:
                        f_test = problem(x_test)
                        f_test = np.array(f_test, dtype=np.float64)
                        used += 1
                        
                        if not (np.all(f_current <= f_test) and np.any(f_current < f_test)):
                            if np.all(f_test <= f_current) and np.any(f_test < f_current):
                                x_current = x_test
                                f_current = f_test
                            add_to_archive(x_test.copy(), f_test.copy())
                            
                    # Try smaller step
                    step_size = 0.01 * (hi[dim] - lo[dim])
                    
                    x_test = x_current.copy()
                    x_test[dim] += step_size
                    if x_test[dim] <= hi[dim]:
                        f_test = problem(x_test)
                        f_test = np.array(f_test, dtype=np.float64)
                        used += 1
                        
                        if not (np.all(f_current <= f_test) and np.any(f_current < f_test)):
                            if np.all(f_test <= f_current) and np.any(f_test < f_current):
                                x_current = x_test
                                f_current = f_test
                            add_to_archive(x_test.copy(), f_test.copy())
                            
                    x_test = x_current.copy()
                    x_test[dim] -= step_size
                    if x_test[dim] >= lo[dim]:
                        f_test = problem(x_test)
                        f_test = np.array(f_test, dtype=np.float64)
                        used += 1
                        
                        if not (np.all(f_current <= f_test) and np.any(f_current < f_test)):
                            if np.all(f_test <= f_current) and np.any(f_test < f_current):
                                x_current = x_test
                                f_current = f_test
                            add_to_archive(x_test.copy(), f_test.copy())

        return archive_X if archive_X else [lo + (hi - lo) * 0.5]
# EVOLVE-BLOCK-END
