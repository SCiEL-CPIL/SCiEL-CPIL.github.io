"""Multi-objective optimizer for WFG4."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using a simple evolutionary strategy with local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj
        
        # Budget tracking
        evals_used = 0
        max_evals = self.budget
        
        # Population size
        pop_size = min(25, max(10, self.budget // 8))
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, n))
        F = np.array([problem(x) for x in X])
        evals_used += pop_size
        
        # Helper: check if time is up
        def time_up():
            return time.time() > deadline or evals_used >= max_evals
        
        # Helper: non-dominated set
        def get_non_dominated(F):
            nd = []
            for i in range(len(F)):
                dominated = False
                for j in range(len(F)):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated = True
                        break
                if not dominated:
                    nd.append(i)
            return nd
        
        # Helper: crowding distance
        def crowding_distance(F, indices):
            if len(indices) <= 1:
                return np.ones(len(indices))
            dist = np.zeros(len(indices))
            for k in range(m):
                obj = F[indices, k]
                min_val, max_val = np.min(obj), np.max(obj)
                if max_val - min_val < 1e-10:
                    continue
                idx_sorted = np.argsort(obj)
                dist[idx_sorted[0]] = np.inf
                dist[idx_sorted[-1]] = np.inf
                for i in range(1, len(indices) - 1):
                    dist[idx_sorted[i]] += (obj[idx_sorted[i+1]] - obj[idx_sorted[i-1]]) / (max_val - min_val)
            return dist
        
        # Main loop
        while not time_up():
            # Get non-dominated set
            nd_indices = get_non_dominated(F)
            if len(nd_indices) == 0:
                nd_indices = [np.argmin(np.sum(F, axis=1))]
            
            # Select parent from non-dominated set (or all if too few)
            if len(nd_indices) < pop_size // 2:
                candidates = list(range(pop_size))
            else:
                candidates = nd_indices
            
            # Pick parent with some randomness
            p_idx = candidates[rng.integers(len(candidates))]
            p = X[p_idx].copy()
            fp = F[p_idx].copy()
            
            # Adaptive mutation step
            # Start with larger steps, shrink over time
            progress = evals_used / max_evals
            step_size = 0.15 * (1 - 0.5 * progress)
            
            # Try several mutations and pick the best
            best_child = None
            best_child_f = None
            best_score = np.sum(fp)
            
            for _ in range(3):
                if time_up():
                    break
                child = p + step_size * (hi - lo) * rng.standard_normal(n)
                child = np.clip(child, lo, hi)
                fc = problem(child)
                evals_used += 1
                
                # Score: sum of objectives (lower is better)
                score = np.sum(fc)
                if score < best_score:
                    best_score = score
                    best_child = child
                    best_child_f = fc
            
            if best_child is None:
                break
            
            # Replacement: find the worst point in the population
            # Use sum of objectives as a simple indicator
            worst_idx = np.argmax(np.sum(F, axis=1))
            
            # Only replace if the child is better than the worst point
            if np.sum(best_child_f) < np.sum(F[worst_idx]):
                X[worst_idx] = best_child
                F[worst_idx] = best_child_f
            elif not np.all(F[worst_idx] <= best_child_f) and not np.all(F[worst_idx] >= best_child_f):
                # Non-dominated: replace if random
                if rng.random() < 0.3:
                    X[worst_idx] = best_child
                    F[worst_idx] = best_child_f
        
        # Return the final population (return value is ignored by evaluator)
        return X
# EVOLVE-BLOCK-END
