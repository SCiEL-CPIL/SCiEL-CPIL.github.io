"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE37 using refined evolutionary search with diversity."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _fast_non_dominated_sort(self, F):
        """Non-dominated sorting. Returns list of fronts (arrays of indices)."""
        n = len(F)
        if n == 0:
            return []
        domination_count = np.zeros(n, dtype=int)
        dominated_set = [[] for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    dominated_set[i].append(j)
                    domination_count[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated_set[j].append(i)
                    domination_count[i] += 1
        
        fronts = []
        current_front = np.where(domination_count == 0)[0]
        while len(current_front) > 0:
            fronts.append(current_front)
            next_front = set()
            for i in current_front:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.add(j)
            current_front = np.array(sorted(next_front), dtype=int)
        
        return fronts

    def _crowding_distance(self, F, indices):
        """Compute crowding distance for given indices."""
        n = len(indices)
        if n == 0:
            return np.array([])
        if n == 1:
            return np.array([np.inf])
        
        cd = np.zeros(n)
        F_sub = F[indices]
        for m in range(F_sub.shape[1]):
            fvals = F_sub[:, m]
            fmin, fmax = fvals.min(), fvals.max()
            rng_m = fmax - fmin
            if rng_m < 1e-15:
                cd[:] = np.inf
                continue
            order = np.argsort(fvals)
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            for i in range(1, n - 1):
                cd[order[i]] += (fvals[order[i+1]] - fvals[order[i-1]]) / rng_m
        
        return cd

    def _select_parents(self, F, n_parents, rng):
        """Select parents using non-dominated sort + crowding distance."""
        n = len(F)
        fronts = self._fast_non_dominated_sort(F)
        
        # Build a ranked list with crowding distances
        ranked = []
        for front in fronts:
            cd = self._crowding_distance(F, front)
            # Sort by crowding distance descending
            order = np.argsort(-cd)
            for idx in order:
                ranked.append(front[idx])
        
        # Select n_parents from ranked list (with replacement if needed)
        if len(ranked) <= n_parents:
            return ranked
        # Tournament-style: pick randomly from top half
        top_half = ranked[:max(1, len(ranked)//2)]
        return [top_half[rng.integers(len(top_half))] for _ in range(n_parents)]

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310  # 330s kill, 20s safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)
        
        pop_size = min(40, self.budget)
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Archive of all non-dominated points seen
        archive_X = []
        archive_F = []
        for i in range(pop_size):
            archive_X.append(X[i].copy())
            archive_F.append(F[i].copy())
        
        n_children = 4  # Multi-child per generation
        
        while used + n_children <= self.budget:
            if time.time() > deadline:
                break
            
            # Generate children
            parents = self._select_parents(F, n_children, rng)
            
            # Progress-based mutation scaling
            progress = used / max(1, self.budget)
            
            children_X = []
            children_F = []
            
            for p_idx in parents:
                # Adaptive mutation: mix of large and small steps
                r = rng.random()
                if r < 0.15:
                    # Large exploration jump
                    scale = 0.35 * (1.0 - 0.4 * progress)
                elif r < 0.6:
                    # Medium step
                    scale = 0.15 * (1.0 - 0.5 * progress)
                else:
                    # Small refinement step
                    scale = 0.05 * (1.0 - 0.3 * progress)
                
                child = X[p_idx] + scale * (hi - lo) * rng.standard_normal(self.dim)
                child = np.clip(child, lo, hi)
                children_X.append(child)
            
            # Evaluate children
            for child in children_X:
                if time.time() > deadline:
                    break
                fc = problem(child)
                used += 1
                children_F.append(fc)
            
            if not children_F:
                break
            
            children_F = np.array(children_F)
            
            # Combine population and children, select new population
            combined_X = np.vstack([X, np.array(children_X)])
            combined_F = np.vstack([F, children_F])
            
            # Non-dominated sort on combined
            fronts = self._fast_non_dominated_sort(combined_F)
            
            new_X = np.zeros((pop_size, self.dim))
            new_F = np.zeros((pop_size, self.n_obj))
            
            for front in fronts:
                if len(front) <= pop_size - sum(1 for f in fronts if len(f) <= len(front) and f is not front):
                    # This front fits entirely
                    pass
                break
            
            # Simpler approach: rank all by (front, crowding) and take top pop_size
            ranked_indices = []
            for front in fronts:
                cd = self._crowding_distance(combined_F, front)
                order = np.argsort(-cd)
                for idx in order:
                    ranked_indices.append(front[idx])
            
            for i in range(min(pop_size, len(ranked_indices))):
                new_X[i] = combined_X[ranked_indices[i]]
                new_F[i] = combined_F[ranked_indices[i]]
            
            X, F = new_X, new_F
            
            # Update archive with new non-dominated points
            for i in range(len(children_F)):
                new_point = children_F[i]
                # Check if it dominates any archive point
                dominated_archive = []
                for j, af in enumerate(archive_F):
                    if np.all(new_point <= af) and np.any(new_point < af):
                        dominated_archive.append(j)
                
                # Check if it's dominated by any archive point
                is_dominated = False
                for j, af in enumerate(archive_F):
                    if j not in dominated_archive:
                        if np.all(af <= new_point) and np.any(af < new_point):
                            is_dominated = True
                            break
                
                if not is_dominated:
                    for j in sorted(dominated_archive, reverse=True):
                        archive_X.pop(j)
                        archive_F.pop(j)
                    archive_X.append(children_X[i].copy())
                    archive_F.append(new_point.copy())
            
            # Periodic random injection for diversity
            if used % 20 < n_children and used + 1 <= self.budget:
                if time.time() > deadline:
                    break
                rand_x = lo + (hi - lo) * rng.random(self.dim)
                rand_f = problem(rand_x)
                used += 1
                # Add to archive if non-dominated
                is_dominated = False
                for af in archive_F:
                    if np.all(af <= rand_f) and np.any(af < rand_f):
                        is_dominated = True
                        break
                if not is_dominated:
                    dominated_archive = [j for j, af in enumerate(archive_F) 
                                         if np.all(rand_f <= af) and np.any(rand_f < af)]
                    for j in sorted(dominated_archive, reverse=True):
                        archive_X.pop(j)
                        archive_F.pop(j)
                    archive_X.append(rand_x.copy())
                    archive_F.append(rand_f.copy())
        
        # Final polish: local search on best archive points
        remaining = self.budget - used
        if remaining > 0 and archive_F:
            archive_F_arr = np.array(archive_F)
            # Sort archive by sum of objectives (lower is better)
            sums = np.sum(archive_F_arr, axis=1)
            top_k = min(5, len(archive_F))
            top_idx = np.argsort(sums)[:top_k]
            
            for _ in range(min(remaining, 10)):
                if time.time() > deadline:
                    break
                if used >= self.budget:
                    break
                # Pick a random top point and do very fine local search
                base_idx = top_idx[rng.integers(len(top_idx))]
                base_x = archive_X[base_idx]
                # Very small step
                step = 0.01 * (hi - lo) * rng.standard_normal(self.dim)
                candidate = np.clip(base_x + step, lo, hi)
                cf = problem(candidate)
                used += 1
                
                # Add to archive if non-dominated
                is_dominated = False
                for af in archive_F:
                    if np.all(af <= cf) and np.any(af < cf):
                        is_dominated = True
                        break
                if not is_dominated:
                    dominated_archive = [j for j, af in enumerate(archive_F) 
                                         if np.all(cf <= af) and np.any(cf < af)]
                    for j in sorted(dominated_archive, reverse=True):
                        archive_X.pop(j)
                        archive_F.pop(j)
                    archive_X.append(candidate.copy())
                    archive_F.append(cf.copy())
        
        return X
# EVOLVE-BLOCK-END
