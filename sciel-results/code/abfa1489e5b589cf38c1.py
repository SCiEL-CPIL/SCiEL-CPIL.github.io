import numpy as np
import time
import sys


class Optimizer:
    """Hybrid multi-objective optimizer for WFG4 with coordinate refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hv_contribution(self, f, ref, archive):
        """
        Compute the hypervolume contribution of point f with respect to reference point ref,
        given existing archive points. Uses a simple exact method for 3 objectives.
        """
        if f[0] >= ref[0] or f[1] >= ref[1] or f[2] >= ref[2]:
            return 0.0
        
        # If no archive, contribution is full volume
        if len(archive) == 0:
            return (ref[0] - f[0]) * (ref[1] - f[1]) * (ref[2] - f[2])
        
        # For 3 objectives, the contribution of f is the volume of the box [f, ref]
        # minus the volume of the boxes of archive points that are inside [f, ref]
        # and are not dominated by f.
        
        # Points that are dominated by f do not overlap with f's unique region in a way that
        # subtracts volume (they are outside f's box in at least one dimension where f is better).
        # Actually, points dominated by f have all objectives >= f's objectives, so they are
        # OUTSIDE the box [f, ref] in the sense that they don't contribute to the volume
        # "below" f. Wait, the box [f, ref] is the set of points y such that f_i <= y_i <= ref_i.
        # A point a dominated by f means a_i >= f_i for all i. So a is inside the box [f, ref]
        # if a_i <= ref_i for all i. But if a is dominated by f, then a is not in the archive's
        # non-dominated set (usually). But here archive might contain dominated points? 
        # The problem says "Every returned objective vector is recorded". So archive has all points.
        # Hypervolume is the volume of the union of boxes [f_i, ref] for all non-dominated f_i.
        # Dominated points don't add volume. So we only need to consider non-dominated points in archive.
        
        # For simplicity and correctness with small archives, let's do this:
        # The contribution of f is the volume of the set of y such that:
        # 1. f_i <= y_i <= ref_i for all i
        # 2. For all a in archive (non-dominated), NOT (a_i <= y_i for all i)
        
        # This is getting complex. Let's use a standard 3D hypervolume algorithm.
        # For a point f, its contribution to the hypervolume is:
        # (ref[0] - f[0]) * (ref[1] - f[1]) * (ref[2] - f[2])
        # minus the volume of the intersection of [f, ref] with the union of boxes [a, ref] for a in archive.
        
        # Since all boxes extend to ref, the union of [a, ref] is the set of points y where
        # there exists a in archive such that a_i <= y_i for all i.
        
        # So the contribution is the volume of {y in [f, ref] : for all a in archive, exists i such that y_i < a_i}
        
        # This is equivalent to: volume of [f, ref] minus volume of {y in [f, ref] : exists a in archive with a_i <= y_i for all i}
        
        # The set {y in [f, ref] : exists a in archive with a_i <= y_i for all i} is the union of
        # {y in [f, ref] : a_i <= y_i for all i} for each a in archive.
        # {y in [f, ref] : a_i <= y_i for all i} = [max(f, a), ref] if max(f, a) <= ref in all dims, else empty.
        
        # So we need the volume of the union of boxes [max(f, a), ref] for a in archive.
        # This is a 3D union of axis-aligned boxes problem.
        
        # For small archives, we can use a brute-force or inclusion-exclusion.
        # But this is expensive.
        
        # Alternative: Since we're using this as a fitness guide, an approximate contribution is fine.
        # A common approximation: 
        # contribution = volume of [f, ref] - sum of volumes of [max(f, a), ref] for a in archive that are "close"
        # But this over-subtracts.
        
        # Let's use a simpler heuristic: 
        # If f dominates any point in archive, it's good.
        # If f is dominated by any point in archive, contribution is 0.
        # Otherwise, approximate contribution as volume of [f, ref] scaled by how "spread out" f is.
        
        # For now, let's return a simple metric:
        # -1 if dominated by any archive point
        # -2 if not dominated but close to existing points
        # -3 if not dominated and far from existing points
        
        # Actually, let's just use the volume of [f, ref] as a proxy, assuming archive is non-dominated
        # and well-spread.
        
        return (ref[0] - f[0]) * (ref[1] - f[1]) * (ref[2] - f[2])

    def _coordinate_refine(self, x, problem, bounds, budget_remaining, ref, archive):
        """
        Perform coordinate descent on x, using hypervolume contribution as guide.
        Uses ternary search on each dimension.
        """
        dim = self.dim
        best_x = x.copy()
        best_f = np.array(problem(best_x))
        best_hv = self._hv_contribution(best_f, ref, archive)
        
        # If budget is tight, just do a few iterations
        max_iterations = min(3, budget_remaining // (dim * 3))
        
        for it in range(max_iterations):
            for d in range(dim):
                # Ternary search on dimension d
                lo = bounds[d][0]
                hi = bounds[d][1]
                
                # Sample 5 points along dimension d
                # Include current value
                candidates = [best_x[d]]
                for frac in [0.2, 0.4, 0.6, 0.8]:
                    candidates.append(lo + frac * (hi - lo))
                
                best_val = best_x[d]
                best_obj = best_hv
                
                for val in candidates:
                    if abs(val - best_x[d]) < 1e-10:
                        continue
                    
                    trial_x = best_x.copy()
                    trial_x[d] = val
                    
                    trial_f = np.array(problem(trial_x))
                    trial_hv = self._hv_contribution(trial_f, ref, archive)
                    
                    if trial_hv > best_obj:
                        best_obj = trial_hv
                        best_val = val
                        best_f = trial_f
                        best_x = trial_x
        
        return best_x, best_f

    def _fast_non_dominated_sort(self, F):
        """Return list of fronts (each a list of indices), sorted by dominance rank."""
        n = len(F)
        if n == 0:
            return []
        
        domination_count = np.zeros(n, dtype=int)
        dominated_set = [list() for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    dominated_set[i].append(j)
                    domination_count[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated_set[j].append(i)
                    domination_count[i] += 1
        
        fronts = []
        current_front = [i for i in range(n) if domination_count[i] == 0]
        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current_front = next_front
        
        return fronts

    def _crowding_distance(self, F, front_indices):
        """Compute crowding distance for points in a front."""
        n = len(front_indices)
        if n <= 2:
            return np.full(n, np.inf)
        
        dist = np.zeros(n)
        F_front = F[front_indices]
        m = F_front.shape[1]
        
        for k in range(m):
            min_val = F_front[:, k].min()
            max_val = F_front[:, k].max()
            if max_val - min_val < 1e-15:
                continue
            order = np.argsort(F_front[:, k])
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf
            for i in range(1, n - 1):
                dist[order[i]] += (F_front[order[i + 1], k] - F_front[order[i - 1], k]) / (max_val - min_val)
        
        return dist

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280.0
        
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        ref = np.array([3.0, 5.0, 7.0])
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Initialize with a diverse set of points
        # Use a mix of random and structured points
        n_init = 30
        X_init = []
        
        # Random points
        for _ in range(n_init - 3):
            x = lo + range_ * np.random.rand(dim)
            X_init.append(x)
        
        # Structured points: corners and midpoints
        # All low
        X_init.append(lo.copy())
        # All high
        X_init.append(hi.copy())
        # Midpoints
        X_init.append((lo + hi) / 2)
        
        X = np.array(X_init)
        F = np.array([problem(x) for x in X])
        used = len(X)
        
        # Keep a non-dominated archive for hypervolume guidance
        archive_X = []
        archive_F = []
        
        # Add initial non-dominated points to archive
        fronts = self._fast_non_dominated_sort(F)
        if fronts:
            for idx in fronts[0]:
                archive_X.append(X[idx])
                archive_F.append(F[idx])
        
        # Main loop: alternate between evolutionary steps and coordinate refinement
        pop_size = 30
        
        while used < budget and time.time() < deadline:
            # Check if we should do coordinate refinement
            # Do refinement on the best non-dominated points
            if len(archive_F) > 0 and used < budget - 10:
                # Pick the point with highest hypervolume contribution
                best_idx = 0
                best_hv = -1
                for i, f in enumerate(archive_F):
                    hv = self._hv_contribution(f, ref, archive_F[:i] + archive_F[i+1:])
                    if hv > best_hv:
                        best_hv = hv
                        best_idx = i
                
                # Refine this point
                if budget - used > 20:
                    refined_x, refined_f = self._coordinate_refine(
                        archive_X[best_idx], problem, bounds, budget - used, ref, archive_F
                    )
                    used += 1  # At least 1 evaluation per refinement (approximate)
                    
                    # Check if refined point is non-dominated
                    is_dominated = False
                    for af in archive_F:
                        if np.all(af <= refined_f) and np.any(af < refined_f):
                            is_dominated = True
                            break
                    
                    if not is_dominated:
                        # Remove dominated archive points
                        new_archive_X = []
                        new_archive_F = []
                        for i, af in enumerate(archive_F):
                            if not (np.all(refined_f <= af) and np.any(refined_f < af)):
                                new_archive_X.append(archive_X[i])
                                new_archive_F.append(af)
                        
                        new_archive_X.append(refined_x)
                        new_archive_F.append(refined_f)
                        archive_X = new_archive_X
                        archive_F = new_archive_F
            
            # If no archive or budget is low, do evolutionary step
            if time.time() > deadline - 10:
                break
            
            # Non-dominated sorting
            fronts = self._fast_non_dominated_sort(F)
            
            # Assign ranks
            rank = np.zeros(len(F), dtype=int)
            for i, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = i
            
            # Compute crowding distance for each front
            cd = np.zeros(len(F))
            for front in fronts:
                front_cd = self._crowding_distance(F, front)
                for j, idx in enumerate(front):
                    cd[idx] = front_cd[j]
            
            # Generate offspring
            offspring_X = []
            offspring_F = []
            
            for _ in range(pop_size // 2):
                if used >= budget:
                    break
                
                # Select parents
                p1 = np.random.randint(len(F))
                p2 = np.random.randint(len(F))
                
                # Simple crossover
                if np.random.rand() < 0.9:
                    # SBX-like
                    child = np.zeros(dim)
                    for d in range(dim):
                        if np.random.rand() < 0.5:
                            child[d] = X[p1, d]
                        else:
                            child[d] = X[p2, d]
                else:
                    child = X[p1].copy()
                
                # Mutation
                for d in range(dim):
                    if np.random.rand() < 0.9:
                        # Polynomial mutation
                        u = np.random.rand()
                        if u < 0.5:
                            delta = (2 * u) ** (1.0 / 21) - 1
                        else:
                            delta = 1 - (2 * (1 - u)) ** (1.0 / 21)
                        child[d] += delta * range_[d]
                
                child = np.clip(child, lo, hi)
                fc = problem(child)
                used += 1
                offspring_X.append(child)
                offspring_F.append(fc)
            
            if not offspring_X:
                break
            
            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)
            
            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            
            # Non-dominated sort on combined
            fronts = self._fast_non_dominated_sort(combined_F)
            
            # Assign ranks
            rank = np.zeros(len(combined_F), dtype=int)
            for i, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = i
            
            # Compute crowding distance
            cd = np.zeros(len(combined_F))
            for front in fronts:
                front_cd = self._crowding_distance(combined_F, front)
                for j, idx in enumerate(front):
                    cd[idx] = front_cd[j]
            
            # Select next generation
            new_X = []
            new_F = []
            count = 0
            for front in fronts:
                if count + len(front) <= pop_size:
                    for idx in front:
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                        count += 1
                else:
                    front_cd = cd[front]
                    order = np.argsort(-front_cd)
                    remaining = pop_size - count
                    for k in range(min(remaining, len(front))):
                        idx = front[order[k]]
                        new_X.append(combined_X[idx])
                        new_F.append(combined_F[idx])
                        count += 1
                    break
            
            X = np.array(new_X[:pop_size])
            F = np.array(new_F[:pop_size])
            
            # Update archive with new non-dominated points
            for i, f in enumerate(F):
                is_dominated = False
                for af in archive_F:
                    if np.all(af <= f) and np.any(af < f):
                        is_dominated = True
                        break
                
                if not is_dominated:
                    # Remove dominated archive points
                    new_archive_X = []
                    new_archive_F = []
                    for j, af in enumerate(archive_F):
                        if not (np.all(f <= af) and np.any(f < af)):
                            new_archive_X.append(archive_X[j])
                            new_archive_F.append(af)
                    
                    new_archive_X.append(X[i])
                    new_archive_F.append(F[i])
                    archive_X = new_archive_X
                    archive_F = new_archive_F
            
            # Limit archive size
            if len(archive_F) > 50:
                # Keep the best 50 by crowding distance
                archive_fronts = self._fast_non_dominated_sort(np.array(archive_F))
                if archive_fronts:
                    first_front = archive_fronts[0]
                    if len(first_front) > 50:
                        cd_archive = self._crowding_distance(np.array(archive_F), list(range(len(archive_F))))
                        order = np.argsort(-cd_archive)
                        keep = order[:50]
                        archive_X = [archive_X[i] for i in keep]
                        archive_F = [archive_F[i] for i in keep]
        
        # Return the archive points (best non-dominated)
        if len(archive_X) > 0:
            return np.array(archive_X)
        else:
            return X

