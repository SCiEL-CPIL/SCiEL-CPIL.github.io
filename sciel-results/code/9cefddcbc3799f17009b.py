"""Multi-objective optimizer on RE34 combining Differential Evolution with
hypervolume-aware selection, Pareto archive, and adaptive local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using DE with Pareto archive and hypervolume-guided selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 315.0  # Safety margin before 330s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo

        # Use evaluator's seeded RNG for reproducibility
        rng = np.random

        def dominates(a, b):
            """Returns True if a dominates b (a <= b in all objectives, a < b in at least one)."""
            return np.all(a <= b) and np.any(a < b)

        # Structured initialization: corners + midpoints + random for diversity
        pop_size = min(40, budget // 5)
        
        # Generate corner points (2^5 = 32 corners for 5D)
        corners = []
        for i in range(2 ** dim):
            point = np.array([hi[j] if (i >> j) & 1 else lo[j] for j in range(dim)])
            corners.append(point)
        
        # Add center point
        corners.append((lo + hi) / 2)
        
        # Add random points to fill population
        while len(corners) < pop_size:
            corners.append(lo + span * rng.random(dim))
        
        X = np.array(corners[:pop_size], dtype=np.float64)
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # Reference point for hypervolume
        ref_point = np.array([1705.0, 12.0, 0.3], dtype=np.float64)

        # Pareto archive
        archive_X = [X[0].copy()]
        archive_F = [F[0].copy()]

        # Add remaining initial points to archive
        for i in range(1, pop_size):
            if not any(dominates(af, F[i]) for af in archive_F):
                # Remove dominated from archive
                new_X = []
                new_F = []
                for j in range(len(archive_X)):
                    if not dominates(F[i], archive_F[j]):
                        new_X.append(archive_X[j])
                        new_F.append(archive_F[j])
                new_X.append(X[i].copy())
                new_F.append(F[i].copy())
                archive_X, archive_F = new_X, new_F

        def update_archive(x, f):
            """Add point to archive if non-dominated, remove dominated points."""
            nonlocal archive_X, archive_F
            # Check if dominated by existing archive points
            for af in archive_F:
                if dominates(af, f):
                    return
            # Remove points dominated by new point
            new_X = []
            new_F = []
            for i in range(len(archive_X)):
                if not dominates(f, archive_F[i]):
                    new_X.append(archive_X[i])
                    new_F.append(archive_F[i])
            new_X.append(x.copy())
            new_F.append(f.copy())
            archive_X, archive_F = new_X, new_F

        def fast_non_dominated_sort(F_arr):
            """Return list of fronts, each a list of indices."""
            n = len(F_arr)
            if n == 0:
                return []
            domination_count = np.zeros(n, dtype=int)
            dominated_set = [[] for _ in range(n)]
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        dominated_set[i].append(j)
                        domination_count[j] += 1
            
            current_front = [i for i in range(n) if domination_count[i] == 0]
            fronts = []
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in dominated_set[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(j)
                current_front = next_front
            
            # Add any remaining
            assigned = set()
            for front in fronts:
                for idx in front:
                    assigned.add(idx)
            if len(assigned) < n:
                remaining = [i for i in range(n) if i not in assigned]
                fronts.append(remaining)
            
            return fronts

        def crowding_distance(F_arr, indices):
            """Compute crowding distances for given indices."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            dist = np.zeros(n)
            for m in range(n_obj):
                vals = F_arr[indices, m]
                min_val = vals.min()
                max_val = vals.max()
                if max_val - min_val < 1e-15:
                    continue
                order = np.argsort(vals)
                dist[order[0]] = np.inf
                dist[order[-1]] = np.inf
                for k in range(1, n - 1):
                    dist[order[k]] += (vals[order[k + 1]] - vals[order[k - 1]]) / (max_val - min_val)
            return dist

        def hypervolume_contribution(f, archive_F_arr):
            """Compute approximate hypervolume contribution of point f to archive."""
            if len(archive_F_arr) == 0:
                vol = 1.0
                for m in range(n_obj):
                    diff = ref_point[m] - f[m]
                    if diff <= 0:
                        return 0.0
                    vol *= diff
                return vol
            
            # Check if dominated
            for af in archive_F_arr:
                if dominates(af, f):
                    return 0.0
            
            # Approximate: volume of box dominated by f but not by any archive point
            # Simple approach: product of (ref - f) minus overlap
            vol = 1.0
            for m in range(n_obj):
                diff = ref_point[m] - f[m]
                if diff <= 0:
                    return 0.0
                vol *= diff
            return vol

        def find_worst_to_replace(F_arr, ref_pt):
            """Find worst individual to replace using dominance and hypervolume."""
            n = len(F_arr)
            if n == 0:
                return 0
            
            # Check dominance
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                for j in range(n):
                    if i != j and dominates(F_arr[j], F_arr[i]):
                        dominated[i] = True
                        break
            
            if np.any(dominated):
                dominated_indices = np.where(dominated)[0]
                scores = F_arr[dominated_indices].sum(axis=1)
                return dominated_indices[np.argmax(scores)]
            
            # No dominated points: replace least hypervolume contributor
            contribs = np.ones(n)
            for m in range(n_obj):
                contribs *= np.maximum(ref_pt[m] - F_arr[:, m], 0)
            
            # Add small noise for diversity
            noise = rng.random(n) * 0.05 * (contribs.max() if contribs.max() > 0 else 1)
            scores = -contribs + noise
            return int(np.argmax(scores))

        # DE parameters
        F_factor = 0.8
        CR = 0.9
        # Adaptive mutation scale
        base_mutation_scale = 0.15

        # Main optimization loop
        while used < budget:
            if time.time() - start_time > deadline:
                break

            # Get non-dominated fronts from current population
            fronts = fast_non_dominated_sort(F)
            
            # Select parents from first front (Pareto optimal)
            if fronts and len(fronts[0]) > 0:
                parent_pool = fronts[0]
            else:
                parent_pool = list(range(pop_size))
            
            # Differential Evolution: select 3 distinct random indices from population
            n_pop = len(X)
            if n_pop >= 3:
                idx1, idx2, idx3 = rng.choice(n_pop, size=3, replace=False)
            else:
                idx1, idx2, idx3 = 0, 1, 2
            
            # Create trial vector using DE mutation
            r = rng.random(dim)
            trial = X[idx1] + F_factor * (X[idx2] - X[idx3])
            
            # Binomial crossover with parent from Pareto front
            parent_idx = parent_pool[rng.integers(len(parent_pool))]
            trial = np.where(r < CR, trial, X[parent_idx])
            
            # Adaptive mutation: scale decreases as we approach budget limit
            progress = used / budget
            mutation_scale = base_mutation_scale * (1.0 - 0.5 * progress)
            noise = mutation_scale * span * rng.standard_normal(dim)
            trial = trial + noise
            trial = np.clip(trial, lo, hi)
            
            # Evaluate trial
            ft = problem(trial)
            used += 1
            
            # Update archive
            update_archive(trial, ft)
            
            # Decide replacement using hypervolume-aware strategy
            # Compare trial with worst in population
            worst_idx = find_worst_to_replace(F, ref_point)
            
            # Replace if trial is non-dominated or better than worst
            trial_is_dominated = any(dominates(af, ft) for af in archive_F)
            worst_is_dominated = any(dominates(af, F[worst_idx]) for af in archive_F)
            
            if not trial_is_dominated:
                # Trial is non-dominated: replace worst
                X[worst_idx] = trial
                F[worst_idx] = ft
            elif worst_is_dominated:
                # Both dominated: replace if trial has better hypervolume contribution
                trial_contrib = hypervolume_contribution(ft, archive_F)
                worst_contrib = hypervolume_contribution(F[worst_idx], archive_F)
                if trial_contrib > worst_contrib:
                    X[worst_idx] = trial
                    F[worst_idx] = ft
            else:
                # Both non-dominated: replace if trial dominates worst
                if dominates(ft, F[worst_idx]):
                    X[worst_idx] = trial
                    F[worst_idx] = ft
                elif dominates(F[worst_idx], ft):
                    # Worst dominates trial: keep worst, but add trial to archive if possible
                    pass
                else:
                    # Incomparable: use crowding distance to decide
                    # Keep the one with lower crowding distance (more isolated)
                    front_idx = 0
                    for fi, front in enumerate(fronts):
                        if worst_idx in front:
                            front_idx = fi
                            break
                    if len(fronts) > front_idx and len(fronts[front_idx]) > 2:
                        cd = crowding_distance(F, fronts[front_idx])
                        # Find position of worst_idx in front
                        pos_in_front = fronts[front_idx].index(worst_idx)
                        worst_cd = cd[pos_in_front]
                        # Trial would be in same front if added
                        # Simple heuristic: replace if worst has high crowding distance
                        if worst_cd > np.inf * 0.5:  # Not at boundary
                            X[worst_idx] = trial
                            F[worst_idx] = ft

            # Periodic local refinement: every 20 evaluations, do a local search
            if used % 20 == 0 and used < budget - 5:
                # Select best point from archive for local refinement
                if archive_X:
                    # Pick point with best hypervolume contribution
                    best_archive_idx = 0
                    best_contrib = 0
                    for i in range(len(archive_X)):
                        contrib = hypervolume_contribution(archive_F[i], archive_F)
                        if contrib > best_contrib:
                            best_contrib = contrib
                            best_archive_idx = i
                    
                    base_x = archive_X[best_archive_idx].copy()
                    base_f = archive_F[best_archive_idx].copy()
                    
                    # Local search: small perturbations in random directions
                    n_local = min(5, budget - used)
                    for _ in range(n_local):
                        if used >= budget or time.time() - start_time > deadline:
                            break
                        # Small Gaussian perturbation
                        local_scale = 0.05 * span
                        local_x = base_x + local_scale * rng.standard_normal(dim)
                        local_x = np.clip(local_x, lo, hi)
                        
                        local_f = problem(local_x)
                        used += 1
                        
                        # Update archive
                        update_archive(local_x, local_f)
                        
                        # If local point is better, use it as new base
                        if not any(dominates(af, local_f) for af in archive_F):
                            # Check if it dominates the base
                            if dominates(local_f, base_f):
                                base_x = local_x.copy()
                                base_f = local_f.copy()
                            # Also consider replacing worst in population
                            worst_idx = find_worst_to_replace(F, ref_point)
                            if dominates(local_f, F[worst_idx]):
                                X[worst_idx] = local_x
                                F[worst_idx] = local_f

        return X
# EVOLVE-BLOCK-END
