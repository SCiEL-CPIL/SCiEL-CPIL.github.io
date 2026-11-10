"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Island-model multi-objective optimizer with weight-specific mutation intensities."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    @staticmethod
    def _dominance_sort(F):
        """Return non-dominated sort fronts as list of lists of indices."""
        n = len(F)
        if n == 0:
            return []
        S = [[] for _ in range(n)]
        n_dominated = np.zeros(n, dtype=int)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dominated[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    n_dominated[i] += 1
        front0 = [i for i in range(n) if n_dominated[i] == 0]
        fronts = [front0]
        while front0:
            next_front = []
            for i in front0:
                for j in S[i]:
                    n_dominated[j] -= 1
                    if n_dominated[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            front0 = next_front
        return fronts

    @staticmethod
    def _crowding_distance(F, front_indices):
        """Compute crowding distance for indices in front."""
        m = len(front_indices)
        if m <= 2:
            return np.full(m, np.inf)
        dist = np.zeros(m)
        for k in range(F.shape[1]):
            vals = F[front_indices, k]
            v_min, v_max = vals.min(), vals.max()
            if v_max == v_min:
                continue
            order = np.argsort(vals)
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf
            for idx in range(1, m - 1):
                dist[order[idx]] += (vals[order[idx + 1]] - vals[order[idx - 1]]) / (v_max - v_min)
        return dist

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim, n_obj, budget = self.dim, self.n_obj, self.budget

        # Island model: partition search space into sub-regions based on objective weights
        # 3 islands for 3 objectives, each with weight-specific mutation intensity
        n_islands = 3
        island_weights = [
            np.array([1.0, 0.1, 0.1]),
            np.array([0.1, 1.0, 0.1]),
            np.array([0.1, 0.1, 1.0])
        ]
        # Mutation intensities: higher for the primary objective of each island
        mutation_scales = [0.3, 0.3, 0.3]  # Base scale, will be modulated by island
        
        # Population size per island
        total_pop = max(15, min(30, budget // 7))
        pop_per_island = max(5, total_pop // n_islands)
        
        # Initialize islands
        islands_X = []
        islands_F = []
        for i in range(n_islands):
            # Bias initialization toward region favorable to island's primary objective
            bias = 0.5 + 0.3 * (island_weights[i] / np.sum(island_weights[i]))
            bias = np.clip(bias, 0.1, 0.9)
            X_i = lo + (hi - lo) * (bias + (1 - 2 * bias) * rng.random((pop_per_island, dim)))
            X_i = np.clip(X_i, lo, hi)
            F_i = np.array([problem(x) for x in X_i])
            islands_X.append(X_i)
            islands_F.append(F_i)
        
        used = sum(len(F_i) for F_i in islands_F)
        
        # Main loop
        while used < budget:
            if time.time() > deadline:
                break
            
            for i in range(n_islands):
                if used >= budget or time.time() > deadline:
                    break
                
                X_i = islands_X[i]
                F_i = islands_F[i]
                
                # Non-dominated sort for this island
                fronts = self._dominance_sort(F_i)
                front0 = fronts[0] if fronts else list(range(len(F_i)))
                
                # Generate offspring using DE with island-specific mutation intensity
                n_offspring = pop_per_island
                for _ in range(n_offspring):
                    if used >= budget or time.time() > deadline:
                        break
                    
                    if len(front0) < 3:
                        # Fallback to random mutation
                        parent_idx = rng.integers(len(X_i))
                        scale = mutation_scales[i] * (1.0 + 0.5 * rng.random())
                        child = X_i[parent_idx] + scale * (hi - lo) * rng.standard_normal(dim)
                    else:
                        # DE/rand/1 with island-specific scale
                        idxs = rng.choice(len(front0), size=3, replace=False)
                        a, b, c = front0[idxs[0]], front0[idxs[1]], front0[idxs[2]]
                        # Scale factor modulated by island weight
                        F_de = 0.5 + 0.5 * island_weights[i].max()
                        v = X_i[a] + F_de * (X_i[b] - X_i[c])
                        # Crossover
                        CR = 0.9
                        j_rand = rng.integers(dim)
                        mask = rng.random(dim) < CR
                        mask[j_rand] = True
                        child = np.where(mask, v, X_i[a])
                    
                    child = np.clip(child, lo, hi)
                    fc = problem(child)
                    used += 1
                    
                    # Selection: keep if non-dominated or replace worst dominated
                    # Check if child dominates any existing point
                    dominates_any = False
                    dominated_indices = []
                    for j in range(len(F_i)):
                        if np.all(fc <= F_i[j]) and np.any(fc < F_i[j]):
                            dominates_any = True
                            dominated_indices.append(j)
                    
                    if dominates_any:
                        # Replace the worst dominated point (highest weighted sum)
                        weights = island_weights[i]
                        sums = np.sum(F_i[dominated_indices] * weights, axis=1)
                        worst_local = dominated_indices[np.argmax(sums)]
                        X_i[worst_local] = child
                        F_i[worst_local] = fc
                    else:
                        # Check if child is dominated
                        is_dominated = False
                        for j in range(len(F_i)):
                            if np.all(F_i[j] <= fc) and np.any(F_i[j] < fc):
                                is_dominated = True
                                break
                        
                        if not is_dominated:
                            # Child is non-dominated: replace worst point by weighted sum
                            weights = island_weights[i]
                            sums = np.sum(F_i * weights, axis=1)
                            worst_idx = np.argmax(sums)
                            X_i[worst_idx] = child
                            F_i[worst_idx] = fc
                        else:
                            # Child is dominated: only accept if it improves diversity
                            # Simple heuristic: accept with small probability
                            if rng.random() < 0.1:
                                weights = island_weights[i]
                                sums = np.sum(F_i * weights, axis=1)
                                worst_idx = np.argmax(sums)
                                X_i[worst_idx] = child
                                F_i[worst_idx] = fc
                
                islands_X[i] = X_i
                islands_F[i] = F_i
            
            # Periodic migration: exchange best points between islands
            if used % (total_pop * 2) < pop_per_island:
                for i in range(n_islands):
                    j = (i + 1) % n_islands
                    # Get best point from island j by weighted sum for island i
                    weights_i = island_weights[i]
                    sums_j = np.sum(islands_F[j] * weights_i, axis=1)
                    best_in_j = np.argmin(sums_j)
                    # Get worst point in island i
                    sums_i = np.sum(islands_F[i] * weights_i, axis=1)
                    worst_in_i = np.argmax(sums_i)
                    # Migrate
                    islands_X[i][worst_in_i] = islands_X[j][best_in_j]
                    islands_F[i][worst_in_i] = islands_F[j][best_in_j]
        
        # Combine all islands and return
        all_X = np.vstack(islands_X)
        all_F = np.vstack(islands_F)
        
        # Final non-dominated sort and return best points
        fronts = self._dominance_sort(all_F)
        result_X = []
        result_F = []
        for front in fronts:
            for idx in front:
                result_X.append(all_X[idx])
                result_F.append(all_F[idx])
            if len(result_X) >= total_pop:
                break
        
        if not result_X:
            return all_X[:total_pop]
        
        return np.array(result_X)
# EVOLVE-BLOCK-END
