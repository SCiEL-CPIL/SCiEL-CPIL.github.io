"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-estimation-based island model for WFG4 multi-objective optimization."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    @staticmethod
    def _hypervolume_contribution(point, archive):
        """Estimate hypervolume contribution of a point relative to an archive."""
        # Simple approximation: count how many archive points it dominates
        # and use that as a proxy for contribution
        n = len(archive)
        if n == 0:
            return 1.0
        dominated_count = 0
        for a in archive:
            if np.all(point <= a) and np.any(point < a):
                dominated_count += 1
        return dominated_count + 1.0

    @staticmethod
    def _is_non_dominated(point, archive):
        """Check if point is not dominated by any point in archive."""
        for a in archive:
            if np.all(a <= point) and np.any(a < point):
                return False
        return True

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310.0  # Safety margin before 330s kill
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim, n_obj = self.dim, self.n_obj
        budget = self.budget

        # Island model: partition search space into sub-regions
        # For WFG4 with 3 objectives, use 3 islands with different weight biases
        n_islands = 3
        island_pop_size = max(8, min(20, budget // (n_islands * 3)))
        total_pop = n_islands * island_pop_size
        
        # Initialize islands with different weight biases
        islands_X = []
        islands_F = []
        used = 0
        
        for island_id in range(n_islands):
            # Different mutation intensities per island
            # Island 0: aggressive exploration, Island 1: balanced, Island 2: exploitation
            X_island = lo + (hi - lo) * rng.random((island_pop_size, dim))
            F_island = np.array([problem(x) for x in X_island])
            used += island_pop_size
            islands_X.append(X_island)
            islands_F.append(F_island)

        # Reference point for hypervolume estimation
        ref_point = np.array([3.0, 5.0, 7.0])
        
        # Weight vectors for each island (bias toward different regions of Pareto front)
        weights = [
            np.array([0.6, 0.3, 0.1]),  # Island 0: emphasize obj1
            np.array([0.3, 0.4, 0.3]),  # Island 1: balanced
            np.array([0.1, 0.3, 0.6]),  # Island 2: emphasize obj3
        ]
        
        # Mutation intensities per island
        K_values = [0.8, 0.5, 0.3]  # Aggressive, balanced, exploitative
        CR_values = [0.7, 0.9, 0.95]  # Crossover rates
        
        # Global archive for tracking best points
        archive = []
        
        while used < budget:
            if time.time() > deadline:
                break
            
            # For each island, generate one offspring
            for island_id in range(n_islands):
                if used >= budget or time.time() > deadline:
                    break
                
                X_island = islands_X[island_id]
                F_island = islands_F[island_id]
                pop_size = island_pop_size
                
                # Select parents using weight-biased tournament
                # Score each individual by weighted sum
                w = weights[island_id]
                weighted_scores = F_island @ w
                # Select top 3 by weighted score
                top_indices = np.argsort(weighted_scores)[:3]
                if len(top_indices) < 3:
                    top_indices = list(range(pop_size))
                
                p1, p2, p3 = top_indices[0], top_indices[1], top_indices[2]
                
                # DE mutation with island-specific K
                K = K_values[island_id]
                v = X_island[p1] + K * (X_island[p2] - X_island[p3])
                
                # DE crossover
                CR = CR_values[island_id]
                jrand = rng.integers(dim)
                trial = np.empty(dim)
                for j in range(dim):
                    if rng.random() < CR or j == jrand:
                        trial[j] = v[j]
                    else:
                        trial[j] = X_island[p1][j]
                
                trial = np.clip(trial, lo, hi)
                
                # Evaluate
                ft = problem(trial)
                used += 1
                
                # Check if point is in bounds of interest (better than reference in all objectives)
                if np.all(ft < ref_point):
                    # Check if non-dominated in global archive
                    if self._is_non_dominated(ft, archive):
                        # Remove dominated points from archive
                        archive = [a for a in archive if not (np.all(ft <= a) and np.any(ft < a))]
                        archive.append(ft)
                
                # Island-level replacement: replace worst individual by weighted score
                # Find the individual with highest weighted score (worst)
                worst_idx = int(np.argmax(weighted_scores))
                
                # Only replace if new point is better in weighted sense or non-dominated
                new_weighted = ft @ w
                old_weighted = weighted_scores[worst_idx]
                
                # Check if new point dominates the worst individual
                dominates_worst = np.all(ft <= F_island[worst_idx]) and np.any(ft < F_island[worst_idx])
                
                if new_weighted < old_weighted or dominates_worst:
                    X_island[worst_idx] = trial
                    F_island[worst_idx] = ft
                    weighted_scores[worst_idx] = new_weighted

            # Periodic migration: every 20 evaluations, swap worst individuals between islands
            if used % 20 == 0 and n_islands > 1:
                for i in range(n_islands):
                    j = (i + 1) % n_islands
                    # Swap worst individual from island i with best from island j
                    w_i = weights[i]
                    w_j = weights[j]
                    
                    worst_in_i = int(np.argmax(islands_F[i] @ w_i))
                    best_in_j = int(np.argmin(islands_F[j] @ w_j))
                    
                    # Swap
                    tmp_X = islands_X[i][worst_in_i].copy()
                    tmp_F = islands_F[i][worst_in_i].copy()
                    islands_X[i][worst_in_i] = islands_X[j][best_in_j]
                    islands_F[i][worst_in_i] = islands_F[j][best_in_j]
                    islands_X[j][best_in_j] = tmp_X
                    islands_F[j][best_in_j] = tmp_F

        return islands_X[0]
# EVOLVE-BLOCK-END
