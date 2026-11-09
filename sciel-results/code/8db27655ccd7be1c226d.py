"""Multi-objective optimizer on RE37 with structured search and local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """Multi-objective optimizer using LHS + adaptive local search with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _fast_non_dominated_sort(self, F):
        """Return list of fronts (list of indices)."""
        n = len(F)
        if n == 0:
            return []
        
        S = [[] for _ in range(n)]
        nd = np.zeros(n, dtype=int)
        
        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                    S[p].append(q)
                elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                    nd[p] += 1
        
        front = [i for i in range(n) if nd[i] == 0]
        if not front:
            front = [0]
        
        fronts = [front]
        i = 0
        while front:
            next_front = []
            for p in front:
                for q in S[p]:
                    nd[q] -= 1
                    if nd[q] == 0:
                        next_front.append(q)
            if next_front:
                fronts.append(next_front)
            front = next_front
            i += 1
        return fronts

    def _crowding_distance(self, F, indices):
        """Calculate crowding distance for a front."""
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)
        
        dist = np.zeros(n)
        F_sub = F[indices]
        
        for m in range(self.n_obj):
            order = np.argsort(F_sub[:, m])
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf
            
            min_val = F_sub[order[0], m]
            max_val = F_sub[order[-1], m]
            if max_val == min_val:
                continue
                
            for k in range(1, n - 1):
                dist[order[k]] += (F_sub[order[k + 1], m] - F_sub[order[k - 1], m]) / (max_val - min_val)
        return dist

    def _hypervolume_contribution(self, F, ref_point):
        """Compute hypervolume contribution of each point (approximate)."""
        n = len(F)
        if n == 0:
            return np.array([])
        
        # Filter points that are better than reference
        mask = np.all(F < ref_point, axis=1)
        F_valid = F[mask]
        n_valid = len(F_valid)
        
        if n_valid == 0:
            return np.zeros(n)
        
        # Simple approximation: contribution based on volume of box dominated by point
        # that is not dominated by other points
        contrib = np.zeros(n)
        
        for i in range(n):
            if not mask[i]:
                continue
            # Volume of box from F[i] to ref_point
            box_vol = np.prod(ref_point - F[i])
            # Subtract overlap with other points (approximate)
            overlap = 0.0
            for j in range(n):
                if i == j or not mask[j]:
                    continue
                # Overlap volume
                overlap_box = np.minimum(ref_point, F[i]) - np.maximum(F[i], F[j])
                overlap_box = np.maximum(overlap_box, 0)
                overlap += np.prod(overlap_box)
            contrib[i] = max(0, box_vol - overlap)
        
        return contrib

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        ref_point = np.array([1.1, 1.3, 1.1])
        
        # Phase 1: Latin Hypercube Sampling
        n_lhs = 100
        permuted_indices = []
        for d in range(self.dim):
            perm = np.random.permutation(n_lhs)
            permuted_indices.append(perm)
        
        X_lhs = np.zeros((n_lhs, self.dim))
        for d in range(self.dim):
            X_lhs[:, d] = lo[d] + (permuted_indices[d] + 0.5) * (hi[d] - lo[d]) / n_lhs
            
        F_lhs = np.array([problem(x) for x in X_lhs])
        used = n_lhs
        
        if time.time() > deadline:
            return F_lhs
        
        # Phase 2: Select seeds for local refinement
        fronts = self._fast_non_dominated_sort(F_lhs)
        
        if fronts:
            front_indices = list(fronts[0])
        else:
            front_indices = list(range(n_lhs))
        
        # Select diverse seeds using hypervolume contribution
        if len(front_indices) > 15:
            hv_contrib = self._hypervolume_contribution(F_lhs, ref_point)
            # Combine with crowding distance for diversity
            dist = self._crowding_distance(F_lhs, front_indices)
            # Normalize and combine
            hv_norm = hv_contrib / (np.max(hv_contrib) + 1e-10)
            dist_norm = dist / (np.max(dist) + 1e-10)
            combined_score = 0.7 * hv_norm + 0.3 * dist_norm
            order = np.argsort(combined_score)[::-1]
            front_indices = [front_indices[i] for i in order[:15]]
        
        # Phase 3: Adaptive local search with coarse-to-fine strategy
        new_X = []
        new_F = []
        remaining_budget = self.budget - used
        points_to_refine = len(front_indices)
        
        if points_to_refine == 0:
            return F_lhs
        
        # Allocate evaluations: use adaptive step sizes
        # For each seed, do multiple steps with decreasing step sizes
        n_steps_per_seed = max(2, remaining_budget // points_to_refine)
        
        for idx in front_indices:
            if time.time() > deadline - 10:
                break
            
            x_seed = X_lhs[idx].copy()
            f_seed = F_lhs[idx].copy()
            
            # Coarse-to-fine: start with larger steps, decrease
            base_step = 0.15 * span
            step_sizes = [base_step * (0.5 ** i) for i in range(n_steps_per_seed)]
            
            best_x = x_seed.copy()
            best_f = f_seed.copy()
            
            for step_i, step_size in enumerate(step_sizes):
                if time.time() > deadline - 5:
                    break
                if used >= self.budget:
                    break
                
                # Try axis-aligned moves (exploit structure)
                for axis in range(self.dim):
                    if time.time() > deadline - 5 or used >= self.budget:
                        break
                    
                    # Try positive and negative moves along this axis
                    for direction in [1, -1]:
                        if time.time() > deadline - 5 or used >= self.budget:
                            break
                        
                        x_try = best_x.copy()
                        x_try[axis] += direction * step_size
                        x_try = np.clip(x_try, lo, hi)
                        
                        # Skip if too close to boundary (already tried)
                        if np.allclose(x_try, best_x):
                            continue
                        
                        f_try = problem(x_try)
                        used += 1
                        new_X.append(x_try)
                        new_F.append(f_try)
                        
                        # Check if this improves (in any objective without worsening others too much)
                        if np.any(f_try < best_f) and not np.any(f_try > best_f + 0.1 * np.abs(best_f)):
                            best_x = x_try.copy()
                            best_f = f_try.copy()
                
                # Also try a random perturbation for exploration
                if used < self.budget and time.time() < deadline - 5:
                    x_rand = best_x + step_size * 0.5 * np.random.standard_normal(self.dim)
                    x_rand = np.clip(x_rand, lo, hi)
                    if not np.allclose(x_rand, best_x):
                        f_rand = problem(x_rand)
                        used += 1
                        new_X.append(x_rand)
                        new_F.append(f_rand)
                        
                        if np.any(f_rand < best_f) and not np.any(f_rand > best_f + 0.1 * np.abs(best_f)):
                            best_x = x_rand.copy()
                            best_f = f_rand.copy()
            
            # Add the best found point for this seed
            if new_F:
                new_X.append(best_x)
                new_F.append(best_f)
                used += 1
        
        # Combine all results
        if new_F:
            F_combined = np.vstack([F_lhs, np.array(new_F)])
        else:
            F_combined = F_lhs
        
        return F_combined
# EVOLVE-BLOCK-END
