"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Fast evolutionary multi-objective optimizer for WFG7."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # Use a local RNG for speed and reproducibility
        rng = np.random.default_rng(42)

        # Population size: larger population helps with diversity in early stages
        pop_size = min(100, max(20, self.budget // 2))
        
        # Initial population
        X = lo + span * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        def fast_non_dominated_sort(F):
            """Vectorized non-dominated sorting."""
            n = len(F)
            if n == 0:
                return np.array([], dtype=int)
            
            # F is (n, m). We want to find i such that no j exists with F[j] <= F[i] and F[j] < F[i].
            # Dominance matrix D[i,j] is True if i dominates j.
            # D[i,j] = all(F[i] <= F[j]) and any(F[i] < F[j])
            
            # To speed up, we can check if any point is dominated by any other.
            # A point is non-dominated if it is not dominated by any other.
            
            # Let's use a simple loop but optimized with numpy broadcasting where possible, 
            # or just a fast python loop if n is small.
            # For n=100, m=3, a python loop is acceptable if tight.
            
            nd_indices = []
            for i in range(n):
                is_dominated = False
                # Check if any j dominates i
                # F[j] <= F[i] and any(F[j] < F[i])
                # We can vectorize the comparison for a single i against all j
                # But checking 'any' is tricky in pure numpy without loops.
                
                # Simpler: check all pairs.
                # Since n is small (<=100), O(n^2) is 10,000 ops, which is fast.
                
                # Optimization: Precompute nothing, just loop.
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        is_dominated = True
                        break
                if not is_dominated:
                    nd_indices.append(i)
            
            return np.array(nd_indices, dtype=int)

        def crowding_distance(F, nd_indices):
            """Compute crowding distance for non-dominated points."""
            n_nd = len(nd_indices)
            if n_nd <= 2:
                return np.ones(n_nd)
            
            distances = np.zeros(n_nd)
            F_nd = F[nd_indices]
            
            for m in range(self.n_obj):
                min_val = F_nd[:, m].min()
                max_val = F_nd[:, m].max()
                range_val = max_val - min_val
                if range_val < 1e-15:
                    continue
                
                # Sort indices by this objective
                sorted_idx = np.argsort(F_nd[:, m])
                
                # Assign infinity to extremes
                distances[sorted_idx[0]] = np.inf
                distances[sorted_idx[-1]] = np.inf
                
                # Assign distances to middle points
                for k in range(1, n_nd - 1):
                    prev_val = F_nd[sorted_idx[k-1], m]
                    next_val = F_nd[sorted_idx[k+1], m]
                    distances[sorted_idx[k]] += (next_val - prev_val) / range_val
            
            return distances

        # Main loop
        while used < self.budget:
            # Check time
            if time.time() > deadline - 2.0:
                break
            
            # Find non-dominated set
            nd_idx = fast_non_dominated_sort(F)
            
            if len(nd_idx) == 0:
                # Should not happen, but fallback
                nd_idx = np.array([rng.integers(pop_size)])

            # Compute crowding distance for non-dominated
            cd = crowding_distance(F, nd_idx)

            # Selection: Tournament selection among non-dominated
            # Pick 2 random from non-dominated, choose the one with higher crowding distance
            if len(nd_idx) > 1:
                a, b = rng.choice(len(nd_idx), size=2, replace=False)
                if cd[a] >= cd[b]:
                    parent_idx = nd_idx[a]
                else:
                    parent_idx = nd_idx[b]
            else:
                parent_idx = nd_idx[0]
            
            parent = X[parent_idx]

            # Mutation: Adaptive step size
            # Use a mix of global and local search
            # Early on: larger steps. Later: smaller steps.
            progress = used / self.budget
            if rng.random() < 0.1:
                # Global restart
                step = 0.5
            else:
                # Local refinement, decreasing step
                step = 0.1 * (1.0 - progress) + 0.01

            mutation = step * span * rng.standard_normal(self.dim)
            child = parent + mutation
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Replacement: 
            # If child is non-dominated, add to population (if size allows, or replace worst)
            # Simple strategy: Add to population, then prune to pop_size.
            X = np.vstack([X, child])
            F = np.vstack([F, fc])
            
            # Prune if population exceeds size
            if len(X) > pop_size:
                # Find non-dominated set in the larger population
                nd_idx_new = fast_non_dominated_sort(F)
                
                if len(nd_idx_new) > pop_size:
                    # Keep pop_size points from non-dominated based on crowding distance
                    cd_new = crowding_distance(F, nd_idx_new)
                    # Sort by crowding distance descending
                    sorted_nd = nd_idx_new[np.argsort(-cd_new)]
                    keep_indices = sorted_nd[:pop_size]
                    
                    # Also keep any dominated points? No, we want to keep the best front.
                    # But if we just keep top pop_size from non-dominated, we might lose diversity?
                    # Standard NSGA-II keeps the entire non-dominated front if it fits.
                    # If it doesn't fit, we use crowding distance.
                    
                    # Let's just keep the top pop_size points by non-dominance and crowding.
                    # We need to map back to original indices or just reconstruct.
                    # It's easier to just filter X and F.
                    
                    # Create a mask of indices to keep
                    keep_mask = np.zeros(len(X), dtype=bool)
                    keep_mask[keep_indices] = True
                    X = X[keep_mask]
                    F = F[keep_mask]
                else:
                    # Non-dominated set is smaller than pop_size, so we need to add dominated points?
                    # Or just keep all non-dominated and fill with best dominated?
                    # Simpler: Keep all non-dominated, and if we need more, pick from dominated.
                    # But usually, we just want to maintain a population of size pop_size.
                    # If nd_idx_new < pop_size, we can keep all nd_idx_new and fill the rest with random or best dominated.
                    # For simplicity, if nd_idx_new < pop_size, we just keep all nd_idx_new and random others?
                    # Actually, let's just keep the top pop_size points from the whole population based on a simple score.
                    # But that's expensive.
                    
                    # Alternative: If nd_idx_new < pop_size, we keep all of them and add random points?
                    # No, let's just keep the non-dominated ones and if we have space, keep the best dominated ones.
                    # To keep it fast: Just keep the non-dominated ones. If len < pop_size, we can just let the population shrink?
                    # No, that's bad.
                    
                    # Let's do this: If nd_idx_new < pop_size, we keep all nd_idx_new. 
                    # We need to add (pop_size - len(nd_idx_new)) points. 
                    # We can pick them randomly from the remaining dominated points.
                    dominated_indices = np.setdiff1d(np.arange(len(X)), nd_idx_new)
                    if len(dominated_indices) > 0:
                        to_add = rng.choice(dominated_indices, size=pop_size - len(nd_idx_new), replace=False)
                        keep_indices = np.concatenate([nd_idx_new, to_add])
                    else:
                        keep_indices = nd_idx_new
                    
                    keep_mask = np.zeros(len(X), dtype=bool)
                    keep_mask[keep_indices] = True
                    X = X[keep_mask]
                    F = F[keep_mask]

        return X
# EVOLVE-BLOCK-END
