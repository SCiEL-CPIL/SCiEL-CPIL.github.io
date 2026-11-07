"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """NSGA-II with hypervolume-guided selection and local polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        ref = np.array([3000.0, 0.06])
        
        # Smaller population for faster generations, more generations
        pop_size = 30
        if budget < pop_size * 2:
            pop_size = max(10, budget // 4)

        # Initialize population with some structure
        X = lo + (hi - lo) * np.random.rand(pop_size, dim)
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Parameters
        eta_c = 10.0
        eta_m = 10.0
        p_c = 0.95
        p_m = 0.5  # Higher mutation for exploration in 4D

        # Track best HV for selection
        best_hv = self._hv_2d(F, ref)

        while used + pop_size <= budget:
            # Check time guard
            if time.time() > deadline - 10.0:
                break

            # Compute ranks and crowding
            ranks, crowding = self._non_dominated_sort(F)
            
            # Tournament selection
            parents_idx = np.zeros(pop_size, dtype=int)
            for i in range(pop_size):
                idx1 = np.random.randint(pop_size)
                idx2 = np.random.randint(pop_size)
                if ranks[idx1] < ranks[idx2]:
                    parents_idx[i] = idx1
                elif ranks[idx2] < ranks[idx1]:
                    parents_idx[i] = idx2
                else:
                    if crowding[idx1] >= crowding[idx2]:
                        parents_idx[i] = idx1
                    else:
                        parents_idx[i] = idx2
            
            # Generate offspring
            offspring = np.zeros((pop_size, dim))
            for i in range(0, pop_size, 2):
                if np.random.rand() < p_c:
                    p1 = X[parents_idx[i]]
                    p2 = X[parents_idx[i+1]]
                    c1, c2 = self._sbx_crossover(p1, p2, lo, hi, eta_c)
                    offspring[i] = c1
                    offspring[i+1] = c2
                else:
                    offspring[i] = X[parents_idx[i]].copy()
                    offspring[i+1] = X[parents_idx[i+1]].copy()
            
            # Mutation with higher probability
            for i in range(pop_size):
                offspring[i] = self._polynomial_mutation(offspring[i], lo, hi, eta_m, p_m)
            
            # Evaluate offspring
            F_offspring = np.array([problem(x) for x in offspring])
            used += pop_size
            
            # Combine
            X_combined = np.vstack([X, offspring])
            F_combined = np.vstack([F, F_offspring])
            
            # Hypervolume-guided selection: keep top pop_size by HV contribution
            X, F = self._hv_select(X_combined, F_combined, pop_size, ref)

        # Final local polish on non-dominated points
        if time.time() < deadline - 15.0:
            X, F = self._local_polish(problem, X, F, lo, hi, ref, deadline, used, budget)

        return X

    def _hv_select(self, X, F, n_keep, ref):
        """Select n_keep points that maximize hypervolume contribution."""
        n = len(F)
        if n <= n_keep:
            return X, F
        
        # Compute HV contribution for each point
        contributions = np.zeros(n)
        for i in range(n):
            # Remove point i and compute HV
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            F_without = F[mask]
            # Only count points better than ref
            valid = np.all(F_without < ref, axis=1)
            if np.sum(valid) == 0:
                contributions[i] = 0
            else:
                contributions[i] = self._hv_2d(F_without[valid], ref)
        
        # Keep top n_keep by contribution
        keep_idx = np.argsort(contributions)[::-1][:n_keep]
        keep_idx = np.sort(keep_idx)  # Keep consistent ordering
        return X[keep_idx], F[keep_idx]

    def _hv_2d(self, F, ref):
        """Compute hypervolume for 2D objective space with given reference point."""
        # Only consider points strictly better than ref in both objectives
        valid = np.all(F < ref, axis=1)
        if np.sum(valid) == 0:
            return 0.0
        F = F[valid]
        
        # Sort by first objective (ascending)
        order = np.argsort(F[:, 0])
        F_sorted = F[order]
        
        hv = 0.0
        max_f1 = -np.inf
        for i in range(len(F_sorted)):
            f0 = F_sorted[i, 0]
            f1 = F_sorted[i, 1]
            if f1 < max_f1:
                continue
            # This point contributes (ref[0] - f0) * (ref[1] - f1)
            # but we need to account for previous points
            # Standard 2D HV: sort by obj1, then for each point that improves obj2,
            # add (ref[0] - f0) * (ref[1] - f1)
            # Actually, simpler: the contribution of point i when sorted by obj1 is
            # (ref[0] - f0_i) * (ref[1] - f1_i) only if f1_i is the new minimum so far
            # But that's not quite right either.
            
            # Correct 2D HV: sort by obj1 ascending. For each point, if its obj2 is less than
            # all previous obj2 values, it contributes (ref[0]-f0)*(ref[1]-f1).
            # But actually the standard algorithm:
            # Sort by f1 ascending. Track max f2 seen so far. For each point, if f2 < max_f2_so_far... 
            # No wait. Let me think again.
            
            # The hypervolume is the union of boxes [f0, ref0] x [f1, ref1] for all points.
            # Sort by f0 ascending. For each point in order, if f1 is less than the minimum f1 seen so far,
            # then this point extends the front. The contribution is (ref0 - f0) * (ref1 - f1).
            # But we need to be careful: if a later point has smaller f0 but larger f1, it doesn't help.
            
            # Actually the correct simple 2D HV algorithm:
            # Sort points by f1 (second objective) ascending.
            # Then sweep from left to right in f1. For each point, if f0 is less than the minimum f0 seen so far,
            # add (ref0 - f0) * (ref1 - f1).
            pass
        
        # Let me just implement it properly
        return self._hv_2d_correct(F, ref)

    def _hv_2d_correct(self, F, ref):
        """Correct 2D hypervolume computation."""
        valid = np.all(F < ref, axis=1)
        if np.sum(valid) == 0:
            return 0.0
        F = F[valid]
        n = len(F)
        
        # Sort by first objective ascending
        order = np.argsort(F[:, 0])
        F_sorted = F[order]
        
        hv = 0.0
        # Track the minimum second objective seen so far (from the right, i.e., from worst f1)
        # Actually: sort by f0 ascending. The Pareto front in 2D: as f0 increases, f1 must decrease.
        # So after sorting by f0 ascending, we want points where f1 is decreasing.
        
        # Standard algorithm: sort by f1 ascending. Sweep from smallest f1 to largest.
        # For each point, if f0 is smaller than all previously seen f0's, it contributes.
        
        order2 = np.argsort(F[:, 1])
        F_by_f1 = F[order2]
        
        min_f0 = np.inf
        for i in range(n):
            f0 = F_by_f1[i, 0]
            f1 = F_by_f1[i, 1]
            if f0 < min_f0:
                min_f0 = f0
                hv += (ref[0] - f0) * (ref[1] - f1)
        
        return hv

    def _local_polish(self, problem, X, F, lo, hi, ref, deadline, used, budget):
        """Polish non-dominated points using local optimization."""
        # Get non-dominated points
        n = len(F)
        non_dom_idx = []
        for i in range(n):
            is_dom = False
            for j in range(n):
                if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    is_dom = True
                    break
            if not is_dom:
                non_dom_idx.append(i)
        
        if len(non_dom_idx) == 0:
            return X, F
        
        # Polish each non-dominated point with a few local steps
        new_X = list(X)
        new_F = list(F)
        
        for idx in non_dom_idx[:10]:  # Limit to 10 points for time
            if time.time() > deadline - 5.0:
                break
            
            x0 = X[idx]
            f0 = F[idx]
            
            # Try to improve using L-BFGS-B on a weighted sum
            for w in [0.3, 0.5, 0.7]:
                if time.time() > deadline - 5.0:
                    break
                
                def objective(x):
                    fx = problem(x)
                    return w * fx[0] + (1-w) * fx[1]
                
                try:
                    res = minimize(objective, x0, method='L-BFGS-B', 
                                  bounds=list(zip(lo, hi)),
                                  options={'maxiter': 20, 'ftol': 1e-8})
                    if res.success and res.fun < objective(x0):
                        new_X.append(res.x)
                        new_F.append(problem(res.x))
                        used += 1
                except:
                    pass
        
        if len(new_X) > len(X):
            X_new = np.array(new_X)
            F_new = np.array(new_F)
            # Keep only non-dominated from the expanded set
            X_new, F_new = self._prune_non_dominated(X_new, F_new, ref)
            return X_new, F_new
        
        return X, F

    def _prune_non_dominated(self, X, F, ref):
        """Keep only non-dominated points that are better than ref."""
        n = len(F)
        keep = []
        for i in range(n):
            if not np.all(F[i] < ref):
                continue
            is_dom = False
            for j in range(n):
                if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    is_dom = True
                    break
            if not is_dom:
                keep.append(i)
        
        if len(keep) == 0:
            return X, F
        return X[keep], F[keep]

    def _non_dominated_sort(self, F):
        """
        Compute non-dominated ranks and crowding distances.
        Returns:
            ranks: array of ranks (0 is best)
            crowding: array of crowding distances
        """
        n = len(F)
        ranks = np.zeros(n, dtype=int)
        crowding = np.zeros(n)
        
        # Non-dominated sorting
        S = [[] for _ in range(n)]
        n_dominates = np.zeros(n, dtype=int)
        front = []
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dominates[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_dominates[i] += 1
        
        for i in range(n):
            if n_dominates[i] == 0:
                ranks[i] = 0
                front.append(i)
        
        k = 0
        while front:
            next_front = []
            for i in front:
                for j in S[i]:
                    n_dominates[j] -= 1
                    if n_dominates[j] == 0:
                        ranks[j] = k + 1
                        next_front.append(j)
            k += 1
            front = next_front
        
        # Calculate crowding distance for each front
        max_rank = max(ranks) if n > 0 else 0
        fronts = [[] for _ in range(max_rank + 1)]
        for i in range(n):
            fronts[ranks[i]].append(i)
        
        for front_indices in fronts:
            if len(front_indices) <= 2:
                for i in front_indices:
                    crowding[i] = np.inf
                continue
            
            m = self.n_obj
            for k_obj in range(m):
                sorted_idx = sorted(front_indices, key=lambda x: F[x, k_obj])
                crowding[sorted_idx[0]] = np.inf
                crowding[sorted_idx[-1]] = np.inf
                f_min = F[sorted_idx[0], k_obj]
                f_max = F[sorted_idx[-1], k_obj]
                if f_max - f_min == 0:
                    continue
                for i in range(1, len(sorted_idx) - 1):
                    idx = sorted_idx[i]
                    idx_prev = sorted_idx[i-1]
                    idx_next = sorted_idx[i+1]
                    crowding[idx] += (F[idx_next, k_obj] - F[idx_prev, k_obj]) / (f_max - f_min)
        
        return ranks, crowding

    def _sbx_crossover(self, p1, p2, lo, hi, eta):
        c1 = p1.copy()
        c2 = p2.copy()
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                if abs(p2[i] - p1[i]) < 1e-12:
                    beta = 1.0
                else:
                    r = np.random.rand()
                    if r <= 0.5:
                        alpha = 2.0 * r
                        beta = (alpha) ** (1.0 / (eta + 1.0))
                    else:
                        alpha = 2.0 * (1.0 - r)
                        beta = (alpha) ** (1.0 / (eta + 1.0))
                
                c1[i] = 0.5 * ((1.0 + beta) * p1[i] + (1.0 - beta) * p2[i])
                c2[i] = 0.5 * ((1.0 - beta) * p1[i] + (1.0 + beta) * p2[i])
        
        c1 = np.clip(c1, lo, hi)
        c2 = np.clip(c2, lo, hi)
        return c1, c2

    def _polynomial_mutation(self, x, lo, hi, eta, p_m=0.5):
        x_mut = x.copy()
        for i in range(self.dim):
            if np.random.rand() < p_m:
                range_i = hi[i] - lo[i]
                if range_i == 0:
                    continue
                x_norm = (x[i] - lo[i]) / range_i
                
                r = np.random.rand()
                if r < 0.5:
                    xy = 2.0 * r
                    c = (xy) ** (1.0 / (eta + 1.0))
                    delta = c * 1.0
                    y_new = x_norm - delta
                else:
                    xy = 2.0 * (1.0 - r)
                    c = (xy) ** (1.0 / (eta + 1.0))
                    delta = c * 1.0
                    y_new = x_norm + delta
                
                y_new = np.clip(y_new, 0.0, 1.0)
                x_mut[i] = lo[i] + y_new * range_i
        
        return x_mut
# EVOLVE-BLOCK-END

    def _non_dominated_sort(self, F):
        """
        Compute non-dominated ranks and crowding distances.
        Returns:
            ranks: array of ranks (0 is best)
            crowding: array of crowding distances
        """
        n = len(F)
        ranks = np.zeros(n, dtype=int)
        crowding = np.zeros(n)
        
        # Non-dominated sorting
        S = [[] for _ in range(n)] # S[i] = set of individuals dominated by i
        n_dominates = np.zeros(n, dtype=int) # n_dominates[i] = number of individuals dominating i
        front = []
        
        for i in range(n):
            for j in range(i + 1, n):
                # Check if i dominates j
                # Minimization: i dominates j if F[i] <= F[j] and F[i] != F[j]
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dominates[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_dominates[i] += 1
        
        # Find first front
        for i in range(n):
            if n_dominates[i] == 0:
                ranks[i] = 0
                front.append(i)
        
        k = 0
        while front:
            next_front = []
            for i in front:
                for j in S[i]:
                    n_dominates[j] -= 1
                    if n_dominates[j] == 0:
                        ranks[j] = k + 1
                        next_front.append(j)
            k += 1
            front = next_front
        
        # Calculate crowding distance for each front
        # Group indices by rank
        max_rank = max(ranks) if n > 0 else 0
        fronts = [[] for _ in range(max_rank + 1)]
        for i in range(n):
            fronts[ranks[i]].append(i)
        
        for front_indices in fronts:
            if len(front_indices) <= 2:
                # If front has 1 or 2 individuals, crowding is infinity (or large)
                for i in front_indices:
                    crowding[i] = np.inf
                continue
            
            m = self.n_obj
            # Sort front by each objective
            for k_obj in range(m):
                # Sort indices in front_indices by F[obj]
                sorted_idx = sorted(front_indices, key=lambda x: F[x, k_obj])
                # Assign inf to first and last
                crowding[sorted_idx[0]] = np.inf
                crowding[sorted_idx[-1]] = np.inf
                # Calculate range
                f_min = F[sorted_idx[0], k_obj]
                f_max = F[sorted_idx[-1], k_obj]
                if f_max - f_min == 0:
                    continue
                for i in range(1, len(sorted_idx) - 1):
                    idx = sorted_idx[i]
                    idx_prev = sorted_idx[i-1]
                    idx_next = sorted_idx[i+1]
                    # Add to existing crowding distance
                    # Note: standard NSGA-II adds to the distance. 
                    # We initialize to 0, so we add.
                    # But wait, we set first/last to inf. 
                    # For middle points, we add (F[next] - F[prev]) / (f_max - f_min)
                    crowding[idx] += (F[idx_next, k_obj] - F[idx_prev, k_obj]) / (f_max - f_min)
        
        return ranks, crowding

    def _sbx_crossover(self, p1, p2, lo, hi, eta):
        c1 = p1.copy()
        c2 = p2.copy()
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                # Calculate beta
                if abs(p2[i] - p1[i]) < 1e-12:
                    beta = 1.0
                else:
                    r = np.random.rand()
                    if r <= 0.5:
                        alpha = 2.0 * r
                        beta = (alpha) ** (1.0 / (eta + 1.0))
                    else:
                        alpha = 2.0 * (1.0 - r)
                        beta = (alpha) ** (1.0 / (eta + 1.0))
                
                # c1 = 0.5 * ((1+beta)*p1 + (1-beta)*p2)
                # c2 = 0.5 * ((1-beta)*p1 + (1+beta)*p2)
                c1[i] = 0.5 * ((1.0 + beta) * p1[i] + (1.0 - beta) * p2[i])
                c2[i] = 0.5 * ((1.0 - beta) * p1[i] + (1.0 + beta) * p2[i])
        
        # Clip to bounds
        c1 = np.clip(c1, lo, hi)
        c2 = np.clip(c2, lo, hi)
        return c1, c2

    def _polynomial_mutation(self, x, lo, hi, eta):
        x_mut = x.copy()
        for i in range(self.dim):
            if np.random.rand() < 0.5: # Mutation probability per gene is usually 1/dim, but 0.5 is safe? 
                # Standard NSGA-II uses p_m = 1/dim. Let's use that.
                pass # We'll implement standard p_m = 1/dim below
        
        # Reset to standard implementation
        p_m = 1.0 / self.dim
        for i in range(self.dim):
            if np.random.rand() < p_m:
                # Calculate delta
                if x[i] == lo[i]:
                    y_low = 0.0
                    y_high = (hi[i] - lo[i]) / (hi[i] - lo[i]) # Normalized? 
                    # Standard formula:
                    # if x == lo, y_low = 0, y_high = (hi - lo) / (hi - lo) = 1? 
                    # Actually, the polynomial mutation formula works on the normalized space [0,1] or directly?
                    # Let's stick to the direct formula from Deb's book.
                    pass
        
        # Let's implement a simpler, robust mutation: Gaussian or Uniform within bounds?
        # Polynomial is better. Let's implement it correctly.
        
        x_mut = x.copy()
        for i in range(self.dim):
            if np.random.rand() < p_m:
                # Normalize x to [0,1]
                range_i = hi[i] - lo[i]
                if range_i == 0:
                    continue
                x_norm = (x[i] - lo[i]) / range_i
                
                y_low = 0.0
                y_high = 1.0
                delta_1 = (y_high - y_low)
                
                if x_norm == y_low:
                    y_low = 0.0
                    delta_2 = y_high - y_low
                elif x_norm == y_high:
                    y_high = 1.0
                    delta_2 = y_high - y_low
                else:
                    delta_2 = y_high - y_low
                
                r = np.random.rand()
                if r < 0.5:
                    xy = 2.0 * r
                    c = (xy) ** (1.0 / (eta + 1.0))
                    delta = c * delta_2
                    y_new = x_norm - delta
                else:
                    xy = 2.0 * (1.0 - r)
                    c = (xy) ** (1.0 / (eta + 1.0))
                    delta = c * delta_2
                    y_new = x_norm + delta
                
                # Clip y_new to [0,1]
                y_new = np.clip(y_new, 0.0, 1.0)
                x_mut[i] = lo[i] + y_new * range_i
        
        return x_mut
# EVOLVE-BLOCK-END
