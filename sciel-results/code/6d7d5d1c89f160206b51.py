"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer with deadline guard, adaptive mutation, 
    and dominance-based replacement using vectorized operations.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        pop_size = int(min(20, self.budget // 4))
        
        # Latin Hypercube Sampling
        X = np.zeros((pop_size, self.dim))
        for d in range(self.dim):
            samples = np.linspace(0, 1, pop_size)
            rng.shuffle(samples)
            X[:, d] = lo[d] + (hi[d] - lo[d]) * samples
            
        F = np.array([problem(x) for x in X])
        used = pop_size

        def get_dominance_counts(F):
            """
            Returns an array of dominance counts for each point in F.
            count[i] = number of points j such that j dominates i.
            """
            n = len(F)
            if n == 0:
                return np.array([], dtype=int)
            
            # F shape (n, m)
            # Compare F[i] and F[j]
            # F_i = F[:, np.newaxis, :] shape (n, 1, m)
            # F_j = F[np.newaxis, :, :] shape (1, n, m)
            
            # j dominates i if F[j] <= F[i] and F[j] < F[i] (strictly in at least one)
            # Note: We want count for i. So we check if j dominates i.
            # Matrix D where D[j, i] is True if j dominates i.
            
            less_or_equal = np.all(F[np.newaxis, :, :] <= F[:, np.newaxis, :], axis=2)
            less_than = np.any(F[np.newaxis, :, :] < F[:, np.newaxis, :], axis=2)
            
            # D[j, i] = less_or_equal[j, i] and less_than[j, i]
            # Wait, the broadcasting:
            # F[np.newaxis, :, :] is (1, n, m) -> indices (0, j, k)
            # F[:, np.newaxis, :] is (n, 1, m) -> indices (i, 0, k)
            # Result is (n, n, m) -> indices (i, j, k)
            # So result[i, j, k] is F[j, k] <= F[i, k]
            # We want to know if j dominates i.
            # So we need a matrix M where M[i, j] is True if j dominates i.
            # M[i, j] = all(F[j] <= F[i]) and any(F[j] < F[i])
            
            # Let's re-orient.
            # F_i = F[:, np.newaxis, :] (n, 1, m) -> i varies first
            # F_j = F[np.newaxis, :, :] (1, n, m) -> j varies second
            # Comparison: F_j <= F_i
            # np.all(F[np.newaxis, :, :] <= F[:, np.newaxis, :], axis=2)
            # This gives (n, n) array A where A[i, j] is True if F[j] <= F[i] element-wise.
            
            # np.any(F[np.newaxis, :, :] < F[:, np.newaxis, :], axis=2)
            # This gives (n, n) array B where B[i, j] is True if F[j] < F[i] in at least one dim.
            
            # So D[i, j] = A[i, j] and B[i, j] means j dominates i.
            # We want count of j for each i. So sum over axis 1 (j).
            
            dominates = less_or_equal & less_than
            # dominates[i, j] is True if j dominates i
            counts = np.sum(dominates, axis=1)
            return counts

        while used < self.budget and time.time() < deadline:
            dom_counts = get_dominance_counts(F)
            non_dominated_idx = np.where(dom_counts == 0)[0]
            
            if len(non_dominated_idx) > 0:
                parent_idx = non_dominated_idx[rng.integers(len(non_dominated_idx))]
            else:
                parent_idx = rng.integers(len(F))
            
            parent = X[parent_idx]
            
            progress = used / self.budget
            # Decay step size from 0.1 to 0.005
            step_size = 0.1 * (1.0 - progress) + 0.005
            
            child = parent + step_size * (hi - lo) * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Add child to population
            F_new = np.vstack([F, fc])
            X_new = np.vstack([X, child])
            
            # Find worst point to replace
            dom_counts_new = get_dominance_counts(F_new)
            max_dom = np.max(dom_counts_new)
            worst_candidates = np.where(dom_counts_new == max_dom)[0]
            
            if len(worst_candidates) > 1:
                # Tie-breaker: max sum of objectives
                sums = np.sum(F_new[worst_candidates], axis=1)
                worst_in_candidates = np.argmax(sums)
                worst_idx = worst_candidates[worst_in_candidates]
            else:
                worst_idx = worst_candidates[0]
                
            F = np.delete(F_new, worst_idx, axis=0)
            X = np.delete(X_new, worst_idx, axis=0)

        return X
# EVOLVE-BLOCK-END
