"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Evolutionary multi-objective search with deadline guard, diversity maintenance, and local polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _compute_crowding_distance(self, F_nd):
        """Compute crowding distance for non-dominated points."""
        n = len(F_nd)
        if n <= 2:
            return np.full(n, np.inf)
        
        cd = np.zeros(n)
        for m in range(self.n_obj):
            order = np.argsort(F_nd[:, m])
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            if F_nd[order[-1], m] - F_nd[order[0], m] > 0:
                for i in range(1, n - 1):
                    cd[order[i]] += (F_nd[order[i + 1], m] - F_nd[order[i - 1], m]) / (F_nd[order[-1], m] - F_nd[order[0], m])
        return cd

    def _compute_hv_contribution(self, F, ref_point):
        """Compute hypervolume contribution of each point (approximate for small sets)."""
        n = len(F)
        if n == 0:
            return np.array([])
        
        # Filter to only points better than reference
        mask = np.all(F < ref_point, axis=1)
        F_valid = F[mask]
        n_valid = len(F_valid)
        
        if n_valid == 0:
            return np.zeros(n)
        
        # For small sets, compute contribution exactly by checking if removing point changes HV
        # This is O(n * 2^n) which is too slow, so use a simpler approximation
        # Use the idea: contribution is proportional to how much space the point "owns"
        # Approximate by: for each point, compute the volume of the box it defines with ref point
        # that isn't covered by other points
        
        # Simpler: use normalized distance from reference
        ranges = ref_point - F_valid
        # Normalize each objective
        max_ranges = np.max(ranges, axis=0)
        max_ranges[max_ranges == 0] = 1
        normalized = ranges / max_ranges
        
        # Contribution approximation: product of normalized values, adjusted by dominance
        # Points that are more extreme in some objective get higher weight
        contrib = np.prod(normalized, axis=1)
        
        # Boost points that are non-dominated
        le = np.all(F_valid[:, np.newaxis, :] <= F_valid[np.newaxis, :, :], axis=2)
        lt = np.any(F_valid[:, np.newaxis, :] < F_valid[np.newaxis, :, :], axis=2)
        dominates = le & lt
        np.fill_diagonal(dominates, False)
        is_dominated = np.any(dominates, axis=0)
        
        contrib[~is_dominated] *= 2.0
        
        result = np.zeros(n)
        result[mask] = contrib
        return result

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270  # Extra safety margin

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        ref_point = np.array([3.0, 5.0, 7.0])

        # Larger population for better diversity
        pop = int(min(40, self.budget // 4))
        X = lo + range_ * rng.random((pop, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop

        # Track all evaluated points for final result
        all_X = [X.copy()]
        all_F = [F.copy()]

        while used < self.budget:
            if time.time() > deadline:
                break

            n = len(F)
            
            # Fast vectorized dominance check
            le = np.all(F[:, np.newaxis, :] <= F[np.newaxis, :, :], axis=2)
            lt = np.any(F[:, np.newaxis, :] < F[np.newaxis, :, :], axis=2)
            dominates = le & lt
            np.fill_diagonal(dominates, False)
            is_dominated = np.any(dominates, axis=0)

            # Parent selection: prefer non-dominated, use tournament among them
            if np.any(~is_dominated):
                nd_mask = ~is_dominated
                parents_idx = np.where(nd_mask)[0]
                
                # Tournament selection: pick 2 random parents, choose the one with better HV contribution
                if len(parents_idx) > 1:
                    idx1 = parents_idx[rng.integers(len(parents_idx))]
                    idx2 = parents_idx[rng.integers(len(parents_idx))]
                    # Use sum as simple fitness (lower is better)
                    if np.sum(F[idx1]) < np.sum(F[idx2]):
                        parent_idx = idx1
                    else:
                        parent_idx = idx2
                else:
                    parent_idx = parents_idx[0]
            else:
                parent_idx = rng.integers(n)

            parent = X[parent_idx]

            # Adaptive mutation: mix of Gaussian and occasional larger jumps
            progress = used / self.budget
            base_step = 0.12 - 0.09 * progress
            
            # 80% Gaussian, 20% larger exploration
            if rng.random() < 0.8:
                mutation = base_step * range_ * rng.standard_normal(self.dim)
            else:
                mutation = (base_step * 2.5) * range_ * rng.standard_normal(self.dim)
            
            child = parent + mutation
            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1

            # Replacement: use crowding distance for diversity-aware selection
            if np.any(is_dominated):
                # Replace a dominated point
                dom_idx = np.where(is_dominated)[0]
                
                # Among dominated, pick the one with worst crowding distance (if any ND) or highest sum
                if np.any(~is_dominated):
                    nd_idx = np.where(~is_dominated)[0]
                    cd_nd = self._compute_crowding_distance(F[nd_idx])
                    # Pick dominated point to replace: the one closest to ND front in sum
                    worst_dom = dom_idx[np.argmin(np.sum(F[dom_idx], axis=1))]
                else:
                    worst_dom = dom_idx[np.argmax(np.sum(F[dom_idx], axis=1))]
                
                X[worst_dom] = child
                F[worst_dom] = fc
            else:
                # No dominated points: use crowding distance to pick which to replace
                cd = self._compute_crowding_distance(F)
                # Replace the point with smallest crowding distance (most crowded)
                worst = int(np.argmin(cd))
                X[worst] = child
                F[worst] = fc

            # Periodically store snapshot
            if used % 20 == 0:
                all_X.append(X.copy())
                all_F.append(F.copy())

        # Final polish: local search on best non-dominated points
        if time.time() < deadline - 30:
            n = len(F)
            le = np.all(F[:, np.newaxis, :] <= F[np.newaxis, :, :], axis=2)
            lt = np.any(F[:, np.newaxis, :] < F[np.newaxis, :, :], axis=2)
            dominates = le & lt
            np.fill_diagonal(dominates, False)
            is_dominated = np.any(dominates, axis=0)
            
            nd_idx = np.where(~is_dominated)[0]
            
            # Polish up to 5 best ND points
            polish_count = min(5, len(nd_idx), self.budget - used)
            if polish_count > 0:
                # Sort ND points by sum
                nd_sums = np.sum(F[nd_idx], axis=1)
                polish_order = nd_idx[np.argsort(nd_sums)[:polish_count]]
                
                for idx in polish_order:
                    if used >= self.budget or time.time() > deadline:
                        break
                    
                    # Local search with decreasing step
                    best_x = X[idx].copy()
                    best_f = F[idx].copy()
                    
                    for polish_iter in range(10):
                        if used >= self.budget or time.time() > deadline:
                            break
                        
                        step = 0.02 * (0.8 ** polish_iter)
                        for _ in range(3):  # Try 3 random directions per iteration
                            if used >= self.budget or time.time() > deadline:
                                break
                            
                            candidate = best_x + step * range_ * rng.standard_normal(self.dim)
                            candidate = np.clip(candidate, lo, hi)
                            cf = problem(candidate)
                            used += 1
                            
                            # Accept if better in at least one objective and not worse in others
                            if np.all(cf <= best_f) and np.any(cf < best_f):
                                best_x = candidate
                                best_f = cf
                            elif np.sum(cf) < np.sum(best_f) * 0.95:
                                # Accept if significantly better in sum
                                best_x = candidate
                                best_f = cf
                    
                    X[idx] = best_x
                    F[idx] = best_f

        # Return the best non-dominated points from final population
        n = len(F)
        le = np.all(F[:, np.newaxis, :] <= F[np.newaxis, :, :], axis=2)
        lt = np.any(F[:, np.newaxis, :] < F[np.newaxis, :, :], axis=2)
        dominates = le & lt
        np.fill_diagonal(dominates, False)
        is_dominated = np.any(dominates, axis=0)
        
        nd_idx = np.where(~is_dominated)[0]
        if len(nd_idx) > 0:
            return X[nd_idx]
        else:
            return X

# EVOLVE-BLOCK-END
