"""Multi-objective optimizer on WFG4 with hypervolume-guided island model."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-guided island model for WFG4 multi-objective optimization."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _hypervolume_2d(self, points, ref):
        """Exact 2D hypervolume calculation."""
        if len(points) == 0:
            return 0.0
        # Sort by first objective
        pts = sorted(points, key=lambda p: p[0])
        hv = 0.0
        max_y = -np.inf
        for p in pts:
            if p[0] >= ref[0] or p[1] >= ref[1]:
                continue
            if p[1] < max_y:
                continue
            width = min(ref[0], p[0]) - p[0] if p[0] < ref[0] else 0
            # Actually need to be more careful
            pass
        
        # Simpler: use inclusion-exclusion for 2D
        pts = [p for p in points if p[0] < ref[0] and p[1] < ref[1]]
        if not pts:
            return 0.0
        pts.sort(key=lambda p: p[0])
        hv = 0.0
        prev_y = ref[1]
        for p in pts:
            if p[1] < prev_y:
                hv += (ref[0] - p[0]) * (prev_y - p[1])
                prev_y = p[1]
        return hv

    def _estimate_hv_contribution(self, F, ref_point):
        """Estimate hypervolume contribution of each point (approximate)."""
        n = len(F)
        if n == 0:
            return np.array([])
        
        # Filter points strictly better than reference
        valid_mask = np.all(F < ref_point, axis=1)
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            return np.zeros(n)
        
        # For each valid point, estimate its contribution
        # A point's contribution is roughly the volume it uniquely covers
        # Approximation: for each point, compute the box it would add if removed
        contributions = np.zeros(n)
        
        for i in valid_indices:
            # Check if this point is dominated by others
            is_dominated = False
            for j in valid_indices:
                if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    is_dominated = True
                    break
            
            if is_dominated:
                contributions[i] = 0.0
            else:
                # Estimate contribution based on gaps to neighbors
                # For simplicity, use a heuristic based on objective values
                # Points with lower objective values tend to contribute more
                contributions[i] = 1.0 / (1.0 + np.sum(F[i]))
        
        return contributions

    def _fast_non_dominated_sort(self, F):
        """Fast non-dominated sorting. Returns list of fronts (lists of indices)."""
        n = len(F)
        if n == 0:
            return []
        S = [[] for _ in range(n)]
        nd = np.zeros(n, dtype=int)
        fronts = []
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    nd[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    nd[i] += 1
        front0 = [i for i in range(n) if nd[i] == 0]
        fronts.append(front0)
        while front0:
            next_front = []
            for i in front0:
                for j in S[i]:
                    nd[j] -= 1
                    if nd[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            front0 = next_front
        return fronts

    def _crowding_distance(self, F, indices):
        """Compute crowding distance for points at given indices."""
        m = len(indices)
        if m <= 2:
            return np.full(m, np.inf)
        dist = np.zeros(m)
        n_obj = F.shape[1]
        for k in range(n_obj):
            vals = F[indices, k]
            v_min, v_max = vals.min(), vals.max()
            if v_max == v_min:
                continue
            order = np.argsort(vals)
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf
            for idx in range(1, m - 1):
                dist[order[idx]] += (vals[order[idx + 1]] - vals[order[idx - 1]]) / (v_max - v_min)
        return dist

    def _lhs_sample(self, rng, n, dim, lo, hi):
        """Latin Hypercube Sampling for better initial coverage."""
        samples = np.zeros((n, dim))
        for d in range(dim):
            perm = rng.permutation(n)
            samples[:, d] = (perm + rng.random(n)) / n
        return lo + (hi - lo) * samples

    def _local_polish(self, problem, bounds, x, lo, hi, rng, max_evals=10):
        """Local gradient-free polish around a point using adaptive step sizes."""
        dim = len(x)
        best_x = x.copy()
        best_f = np.array(problem(best_x))
        evals_used = 1
        step_size = 0.05

        while evals_used < max_evals:
            improved = False
            for d in range(dim):
                if evals_used >= max_evals:
                    break
                for sign in [1, -1]:
                    if evals_used >= max_evals:
                        break
                    trial = best_x.copy()
                    trial[d] += sign * step_size * (hi[d] - lo[d])
                    trial = np.clip(trial, lo, hi)
                    ft = problem(trial)
                    evals_used += 1
                    if np.all(ft <= best_f) and np.any(ft < best_f):
                        best_x = trial
                        best_f = ft
                        improved = True
                        break
                if improved:
                    break
            if improved:
                step_size *= 0.7
            else:
                step_size *= 0.5
                if step_size < 0.001:
                    break
        return best_x, best_f, evals_used

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin before 330s kill
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim, n_obj = self.dim, self.n_obj
        budget = self.budget
        
        # Reference point for hypervolume
        ref_point = np.array([3.0, 5.0, 7.0])
        
        # Island model: 3 islands with different weight vectors
        # Each island focuses on a different region of the Pareto front
        n_islands = 3
        island_pop = 10  # 30 total
        weights = [
            np.array([0.7, 0.15, 0.15]),  # Focus on obj1
            np.array([0.15, 0.7, 0.15]),  # Focus on obj2
            np.array([0.15, 0.15, 0.7])   # Focus on obj3
        ]
        
        # DE parameters per island (higher mutation for more exploratory islands)
        K_values = [0.3, 0.5, 0.7]
        CR_values = [0.8, 0.9, 1.0]
        
        # Initialize islands
        islands_X = []
        islands_F = []
        used = 0
        
        for i in range(n_islands):
            X_i = self._lhs_sample(rng, island_pop, dim, lo, hi)
            F_i = np.array([problem(x) for x in X_i])
            used += island_pop
            islands_X.append(X_i)
            islands_F.append(F_i)
        
        # Main evolution loop
        while used < budget - 15:  # Reserve 15 for polish
            if time.time() > deadline:
                break
            
            # Select island to evolve (round-robin with some randomness)
            island_idx = rng.integers(n_islands)
            
            X_i = islands_X[island_idx]
            F_i = islands_F[island_idx]
            K = K_values[island_idx]
            CR = CR_values[island_idx]
            w = weights[island_idx]
            
            # Non-dominated sorting within island
            fronts = self._fast_non_dominated_sort(F_i)
            
            # Select parents from best front
            if fronts and len(fronts[0]) >= 2:
                parent_pool = fronts[0]
            else:
                parent_pool = list(range(island_pop))
            
            # Pick 3 parents for DE
            if len(parent_pool) >= 3:
                p1, p2, p3 = rng.choice(parent_pool, size=3, replace=False)
            else:
                p1 = rng.choice(parent_pool)
                p2 = rng.choice(parent_pool)
                p3 = rng.choice(parent_pool)
            
            # DE mutation with weight-guided bias
            v = X_i[p1] + K * (X_i[p2] - X_i[p3])
            
            # Add small perturbation in direction guided by weight vector
            # This helps explore different regions
            perturbation = rng.random(dim) * 0.02
            v = v + perturbation
            
            # DE crossover
            jrand = rng.integers(dim)
            trial = np.empty(dim)
            for j in range(dim):
                if rng.random() < CR or j == jrand:
                    trial[j] = v[j]
                else:
                    trial[j] = X_i[p1][j]
            
            trial = np.clip(trial, lo, hi)
            
            # Evaluate
            ft = problem(trial)
            used += 1
            
            # Weighted scalarization to guide selection
            # Lower weighted sum is better
            trial_score = np.dot(ft, w)
            
            # Find dominated points in this island
            dominated = []
            for i in range(island_pop):
                if np.all(ft <= F_i[i]) and np.any(ft < F_i[i]):
                    dominated.append(i)
            
            if dominated:
                # Replace the worst dominated point (highest weighted sum)
                scores = np.array([np.dot(F_i[i], w) for i in dominated])
                worst_idx = dominated[np.argmax(scores)]
                X_i[worst_idx] = trial
                F_i[worst_idx] = ft
            else:
                # Check if trial is dominated
                dominates_trial = []
                for i in range(island_pop):
                    if np.all(F_i[i] <= ft) and np.any(F_i[i] < ft):
                        dominates_trial.append(i)
                
                if dominates_trial:
                    # Replace worst point in last front
                    if fronts and len(fronts) > 1:
                        last_front = fronts[-1]
                        if last_front:
                            cd = self._crowding_distance(F_i, last_front)
                            worst_pos = np.argmin(cd)
                            worst_idx = last_front[worst_pos]
                            X_i[worst_idx] = trial
                            F_i[worst_idx] = ft
                else:
                    # Non-dominated: replace point with lowest crowding distance in last front
                    if fronts and len(fronts) > 1:
                        last_front = fronts[-1]
                        if len(last_front) > 1:
                            cd = self._crowding_distance(F_i, last_front)
                            worst_pos = np.argmin(cd)
                            worst_idx = last_front[worst_pos]
                            X_i[worst_idx] = trial
                            F_i[worst_idx] = ft
                    else:
                        # Replace worst by weighted sum
                        scores = np.array([np.dot(F_i[i], w) for i in range(island_pop)])
                        worst_idx = np.argmax(scores)
                        X_i[worst_idx] = trial
                        F_i[worst_idx] = ft
        
        # Migration: exchange best individuals between islands
        if used < budget - 10 and time.time() < deadline:
            for i in range(n_islands):
                for j in range(n_islands):
                    if i != j:
                        # Find best point in island j by weighted sum with island i's weight
                        F_j = islands_F[j]
                        w_i = weights[i]
                        scores_j = np.array([np.dot(F_j[k], w_i) for k in range(island_pop)])
                        best_in_j = np.argmin(scores_j)
                        
                        # Find worst point in island i
                        F_i = islands_F[i]
                        scores_i = np.array([np.dot(F_i[k], w_i) for k in range(island_pop)])
                        worst_in_i = np.argmax(scores_i)
                        
                        # Exchange if beneficial
                        if scores_j[best_in_j] < scores_i[worst_in_i]:
                            islands_X[i][worst_in_i] = islands_X[j][best_in_j].copy()
                            islands_F[i][worst_in_i] = islands_F[j][best_in_j].copy()
        
        # Final local polish on non-dominated points from all islands
        if used < budget and time.time() < deadline:
            # Collect all non-dominated points
            all_X = np.vstack(islands_X)
            all_F = np.vstack(islands_F)
            fronts = self._fast_non_dominated_sort(all_F)
            front0 = fronts[0] if fronts else []
            
            # Polish top diverse non-dominated points
            n_polish = min(len(front0), max(3, 15 // 5))
            if len(front0) > n_polish:
                cd = self._crowding_distance(all_F, front0)
                top_idx = np.argsort(-cd)[:n_polish]
                polish_idx = [front0[i] for i in top_idx]
            else:
                polish_idx = list(front0)
            
            for idx in polish_idx:
                if used >= budget or time.time() > deadline:
                    break
                remaining = budget - used
                if remaining < 5:
                    break
                max_evals = min(5, remaining)
                new_x, new_f, evals_used = self._local_polish(
                    problem, bounds, all_X[idx], lo, hi, rng, max_evals
                )
                used += evals_used - 1
                if np.all(new_f <= all_F[idx]) and np.any(new_f < all_F[idx]):
                    all_X[idx] = new_x
                    all_F[idx] = new_f
        
        return all_X
# EVOLVE-BLOCK-END
