"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer with LHS init, adaptive mutation, and hypervolume-based replacement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _lhs_init(self, n, dim, lo, hi, rng):
        """Latin Hypercube Sampling for better initial coverage."""
        X = np.zeros((n, dim))
        for d in range(dim):
            perm = rng.permutation(n)
            X[:, d] = (perm + rng.random(n)) / n
        return lo + (hi - lo) * X

    def _non_dominated_rank(self, F):
        """Compute non-dominated ranks (0 = best)."""
        n = len(F)
        ranks = np.full(n, -1, dtype=int)
        dominated_count = np.zeros(n, dtype=int)
        dominates = [[] for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    dominates[i].append(j)
                    dominated_count[j] += 1
        
        front = 0
        while np.any(ranks == -1):
            current_front = np.where((ranks == -1) & (dominated_count == 0))[0]
            if len(current_front) == 0:
                current_front = np.where(ranks == -1)[0]
            for i in current_front:
                ranks[i] = front
            for i in current_front:
                for j in dominates[i]:
                    if ranks[j] == -1:
                        dominated_count[j] -= 1
            front += 1
        return ranks

    def _crowding_distance(self, F):
        """Compute crowding distances."""
        n = len(F)
        if n <= 2:
            return np.full(n, np.inf)
        cd = np.zeros(n)
        for m in range(F.shape[1]):
            order = np.argsort(F[:, m])
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            span = F[order[-1], m] - F[order[0], m]
            if span > 1e-12:
                for i in range(1, n - 1):
                    cd[order[i]] += (F[order[i+1], m] - F[order[i-1], m]) / span
        return cd

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # LHS initialization with population size
        pop_size = 20
        X = self._lhs_init(pop_size, self.dim, lo, hi, rng)
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Adaptive mutation step
        base_step = 0.1
        
        while used < self.budget:
            # Check time limit
            if time.time() > deadline - 5:
                break
            
            # Get non-dominated set
            ranks = self._non_dominated_rank(F)
            nondom_mask = (ranks == 0)
            
            if np.any(nondom_mask):
                parents_idx = np.where(nondom_mask)[0]
                # Use crowding distance for parent selection
                cd = self._crowding_distance(F[parents_idx])
                # Tournament selection based on crowding distance
                i1 = rng.integers(len(parents_idx))
                i2 = rng.integers(len(parents_idx))
                if cd[i1] >= cd[i2]:
                    parent_idx = parents_idx[i1]
                else:
                    parent_idx = parents_idx[i2]
            else:
                parent_idx = rng.integers(pop_size)
            
            # Adaptive mutation
            step = base_step * (1.0 - 0.5 * used / self.budget)
            child = X[parent_idx].copy()
            child += step * (hi - lo) * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Replacement: use hypervolume contribution approximation
            # Find the individual with lowest hypervolume contribution
            # Approximate by finding worst in rank, then worst in crowding
            all_F = np.vstack([F, fc.reshape(1, -1)])
            all_ranks = self._non_dominated_rank(all_F)
            
            # If child is non-dominated, replace the worst non-dominated by crowding
            if all_ranks[-1] == 0:
                nondom_idx = np.where(all_ranks == 0)[0]
                if len(nondom_idx) > pop_size:
                    cd_all = self._crowding_distance(all_F[nondom_idx])
                    worst_local = nondom_idx[np.argmin(cd_all)]
                    if worst_local == pop_size:
                        # Child would be removed, so don't add
                        continue
                    # Remove worst, add child
                    F = np.delete(all_F, worst_local, axis=0)
                    X = np.delete(X, worst_local, axis=0)
                    F = np.vstack([F, fc.reshape(1, -1)])
                    X = np.vstack([X, child.reshape(1, -1)])
                else:
                    # Just add child
                    F = np.vstack([F, fc.reshape(1, -1)])
                    X = np.vstack([X, child.reshape(1, -1)])
                    # If too large, remove worst
                    if len(F) > pop_size:
                        ranks2 = self._non_dominated_rank(F)
                        cd2 = self._crowding_distance(F[ranks2 == 0])
                        nondom2 = np.where(ranks2 == 0)[0]
                        worst2 = nondom2[np.argmin(cd2)]
                        F = np.delete(F, worst2, axis=0)
                        X = np.delete(X, worst2, axis=0)
            else:
                # Child is dominated, replace worst dominated individual
                dominated_idx = np.where(ranks > 0)[0]
                if len(dominated_idx) > 0:
                    worst_d = dominated_idx[np.argmax(ranks[dominated_idx])]
                    F[worst_d] = fc
                    X[worst_d] = child
                else:
                    # All non-dominated, replace worst by crowding
                    cd = self._crowding_distance(F)
                    worst_c = np.argmin(cd)
                    F[worst_c] = fc
                    X[worst_c] = child
        
        return X
# EVOLVE-BLOCK-END
