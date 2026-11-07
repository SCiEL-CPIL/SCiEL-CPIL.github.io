"""Multi-objective optimizer for RE21 using hypervolume-guided search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-guided multi-objective optimizer with adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hv_2d(self, points, ref):
        """Compute exact hypervolume for 2D points against reference point."""
        if len(points) == 0:
            return 0.0
        # Filter points that are strictly better than reference in all objectives
        valid = points[(points[:, 0] < ref[0]) & (points[:, 1] < ref[1])]
        if len(valid) == 0:
            return 0.0
        
        # Sort by first objective (ascending)
        valid = valid[np.argsort(valid[:, 0])]
        
        hv = 0.0
        prev_y = ref[1]
        for i in range(len(valid)):
            x, y = valid[i]
            if y < prev_y:
                width = ref[0] - x
                height = prev_y - y
                hv += width * height
                prev_y = y
        return hv

    def _pareto_front_2d(self, F):
        """Extract non-dominated front from 2D objective matrix efficiently.
        Returns indices of non-dominated points sorted by obj1 ascending."""
        if len(F) == 0:
            return np.array([], dtype=int)
        # Sort by first objective ascending
        order = np.argsort(F[:, 0])
        F_sorted = F[order]
        # Walk through, keeping track of best (min) second objective seen so far
        # A point is non-dominated if its second objective is strictly less than 
        # all previously seen second objectives (since obj1 is already sorted ascending)
        n = len(F_sorted)
        nd_mask = np.ones(n, dtype=bool)
        best_y = np.inf
        for i in range(n):
            if F_sorted[i, 1] < best_y:
                best_y = F_sorted[i, 1]
            else:
                nd_mask[i] = False
        # Map back to original indices
        nd_indices = order[nd_mask]
        return nd_indices

    def _hv_contribution_2d_fast(self, point, front, ref):
        """Compute hypervolume contribution of a single point in 2D efficiently.
        front is assumed to be the Pareto front (non-dominated)."""
        if point[0] >= ref[0] or point[1] >= ref[1]:
            return 0.0
        if len(front) == 0:
            return (ref[0] - point[0]) * (ref[1] - point[1])
        
        # Find where this point would be inserted in the front sorted by obj1
        # The contribution is the area of the "step" that this point adds
        # For 2D, if we sort front by obj1, the contribution of a point is:
        # (ref[0] - point[0]) * (min_y_above - point[1]) where min_y_above is the 
        # minimum y-value among front points with x < point[0], or ref[1] if none
        
        # Sort front by obj1
        front_sorted = front[np.argsort(front[:, 0])]
        
        # Find the point in front with the largest x that is still < point[0]
        # and the point with the smallest y among those with x >= point[0]
        
        # Actually, simpler: the contribution is the area that becomes dominated
        # when we add this point. In 2D, this is:
        # Let's find the "neighbors" in the Pareto front
        # The contribution = (ref[0] - point[0]) * (y_left - point[1]) 
        # where y_left is the y-value of the Pareto point with the largest x < point[0]
        # (or ref[1] if no such point exists)
        
        # Find the Pareto point with largest x < point[0]
        candidates = front_sorted[front_sorted[:, 0] < point[0]]
        if len(candidates) > 0:
            y_left = candidates[-1, 1]  # largest x, so last in sorted order
        else:
            y_left = ref[1]
        
        # The contribution is the rectangle from (point[0], point[1]) to (ref[0], y_left)
        # but only if point[1] < y_left
        if point[1] < y_left:
            return (ref[0] - point[0]) * (y_left - point[1])
        return 0.0

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        ref = np.array([3000.0, 0.06])
        ranges = hi - lo
        
        # Smaller population for faster per-iteration computation
        pop_size = min(20, budget // 10)
        X = np.zeros((pop_size, dim))
        
        # Initialize with diverse points: corners + random
        corners = np.array([
            [lo[j] if (i >> j) & 1 else hi[j] for j in range(dim)]
            for i in range(2 ** dim)
        ])
        
        # Use all corners if they fit, otherwise sample
        if pop_size >= len(corners):
            X[:len(corners)] = corners
            X[len(corners):] = lo + ranges * np.random.random((pop_size - len(corners), dim))
        else:
            idx = np.random.choice(len(corners), pop_size, replace=False)
            X = corners[idx].copy()
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Track all evaluated points for final hypervolume
        all_F = F.copy()
        
        # Adaptive parameters
        mutation_scale = 0.25
        crossover_prob = 0.8
        mutation_prob = 0.9
        restart_interval = 25
        last_restart = 0
        no_improve_count = 0
        best_hv = self._hv_2d(F, ref)
        
        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break
            
            # Extract Pareto front efficiently
            nd_indices = self._pareto_front_2d(F)
            
            if len(nd_indices) == 0:
                nd_indices = np.array([np.argmin(np.sum(F, axis=1))])
            
            # Compute HV contributions for front points for parent selection
            front = F[nd_indices]
            contributions = np.array([
                self._hv_contribution_2d_fast(front[i], np.delete(front, i, axis=0), ref)
                for i in range(len(front))
            ])
            
            total_contrib = np.sum(contributions)
            if total_contrib < 1e-10:
                probs = np.ones(len(nd_indices)) / len(nd_indices)
            else:
                probs = contributions / total_contrib
            
            # Select two parents from front
            if len(nd_indices) >= 2:
                p1_idx = nd_indices[np.random.choice(len(nd_indices), p=probs)]
                p2_idx = nd_indices[np.random.choice(len(nd_indices), p=probs)]
                while p2_idx == p1_idx:
                    p2_idx = nd_indices[np.random.choice(len(nd_indices))]
                p1 = X[p1_idx]
                p2 = X[p2_idx]
            else:
                p1 = X[nd_indices[0]]
                p2 = X[np.random.randint(len(X))]
            
            # Crossover: uniform with SBX-like blending
            if np.random.random() < crossover_prob:
                child = np.zeros(dim)
                for j in range(dim):
                    if np.random.random() < 0.5:
                        child[j] = p1[j]
                    else:
                        child[j] = p2[j]
                    # Add slight blending
                    if np.random.random() < 0.3:
                        blend = np.random.random()
                        child[j] = blend * p1[j] + (1 - blend) * p2[j]
            else:
                child = p1.copy()
            
            # Adaptive mutation
            if np.random.random() < mutation_prob:
                for j in range(dim):
                    if np.random.random() < 0.7:
                        delta = np.random.exponential(1.0) * mutation_scale
                        sign = 1.0 if np.random.random() < 0.5 else -1.0
                        child[j] += sign * delta * ranges[j]
            
            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1
            
            # Update all_F
            all_F = np.vstack([all_F, [fc]])
            
            # Find worst point in population by HV contribution
            # Only compute for the Pareto front members + child
            # The worst is likely a dominated point or a front point with low contribution
            
            # Quick check: if child is non-dominated and has good contribution, replace worst
            # Find dominated points first (they have 0 contribution)
            dominated_mask = np.ones(len(X), dtype=bool)
            for i in range(len(X)):
                # Check if i is dominated by any other
                is_dom = False
                for j in range(len(X)):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        is_dom = True
                        break
                if not is_dom:
                    dominated_mask[i] = False
            
            # Find the worst point: prefer dominated, then lowest HV contribution
            if np.any(dominated_mask):
                # Among dominated, pick one with worst sum of objectives
                dom_indices = np.where(dominated_mask)[0]
                worst_idx = dom_indices[np.argmax(np.sum(F[dom_indices], axis=1))]
            else:
                # All non-dominated, find lowest HV contribution
                front_indices = np.where(~dominated_mask)[0]
                front = F[front_indices]
                contribs = np.array([
                    self._hv_contribution_2d_fast(front[i], np.delete(front, i, axis=0), ref)
                    for i in range(len(front))
                ])
                worst_local = np.argmin(contribs)
                worst_idx = front_indices[worst_local]
            
            # Compute child's contribution if it were added to the front
            # Check if child is non-dominated
            child_is_dom = False
            for i in range(len(X)):
                if np.all(F[i] <= fc) and np.any(F[i] < fc):
                    child_is_dom = True
                    break
            
            if not child_is_dom:
                # Compute contribution of child to current front
                front = F[~dominated_mask]
                child_contrib = self._hv_contribution_2d_fast(fc, front, ref)
                
                # Compute contribution of worst point
                worst_point = F[worst_idx]
                front_without_worst = np.delete(F[~dominated_mask], 
                                               np.where(~dominated_mask)[0].tolist().index(worst_idx) 
                                               if worst_idx in np.where(~dominated_mask)[0] else -1, axis=0)
                worst_contrib = self._hv_contribution_2d_fast(worst_point, front_without_worst, ref)
                
                # Replace if child contributes more
                if child_contrib > worst_contrib * 0.7:
                    X[worst_idx] = child
                    F[worst_idx] = fc
                    
                    # Track improvement
                    new_hv = self._hv_2d(F, ref)
                    if new_hv > best_hv:
                        best_hv = new_hv
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
            else:
                no_improve_count += 1
            
            # Adaptive mutation scale
            if used % 8 == 0:
                mutation_scale *= 0.92
                if mutation_scale < 0.02:
                    mutation_scale = 0.02
            
            # Periodic injection of random points to maintain diversity
            if used - last_restart >= restart_interval:
                n_inject = max(1, pop_size // 15)
                for _ in range(n_inject):
                    if used >= budget:
                        break
                    rand_point = lo + ranges * np.random.random(dim)
                    rand_F = problem(rand_point)
                    used += 1
                    all_F = np.vstack([all_F, [rand_F]])
                    
                    # Replace worst point
                    dominated_mask = np.ones(len(X), dtype=bool)
                    for i in range(len(X)):
                        is_dom = False
                        for j in range(len(X)):
                            if i == j:
                                continue
                            if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                                is_dom = True
                                break
                        if not is_dom:
                            dominated_mask[i] = False
                    
                    if np.any(dominated_mask):
                        dom_indices = np.where(dominated_mask)[0]
                        worst_idx = dom_indices[np.argmax(np.sum(F[dom_indices], axis=1))]
                    else:
                        front_indices = np.where(~dominated_mask)[0]
                        front = F[front_indices]
                        contribs = np.array([
                            self._hv_contribution_2d_fast(front[i], np.delete(front, i, axis=0), ref)
                            for i in range(len(front))
                        ])
                        worst_local = np.argmin(contribs)
                        worst_idx = front_indices[worst_local]
                    
                    X[worst_idx] = rand_point
                    F[worst_idx] = rand_F
                
                last_restart = used
        
        # Final polish: local refinement around best front points
        if time.time() < deadline - 10:
            nd_indices = self._pareto_front_2d(F)
            if len(nd_indices) > 0:
                # Try small perturbations around best points
                for idx in nd_indices:
                    if used >= budget or time.time() > deadline - 5:
                        break
                    base = X[idx].copy()
                    for attempt in range(3):
                        if used >= budget or time.time() > deadline - 5:
                            break
                        perturbation = (np.random.random(dim) - 0.5) * 0.05 * ranges
                        candidate = np.clip(base + perturbation, lo, hi)
                        fc = problem(candidate)
                        used += 1
                        all_F = np.vstack([all_F, [fc]])
                        # Keep if it improves the local front
                        if (fc[0] < F[idx, 0] and fc[1] < F[idx, 1]) or \
                           (fc[0] < F[idx, 0] and fc[1] <= F[idx, 1] * 1.01) or \
                           (fc[1] < F[idx, 1] and fc[0] <= F[idx, 0] * 1.01):
                            X[idx] = candidate
                            F[idx] = fc
        
        return X
# EVOLVE-BLOCK-END
