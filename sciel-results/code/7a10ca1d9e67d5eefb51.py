"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE37 using refined evolutionary search with diversity."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _fast_non_dominated_sort(self, F):
        """Vectorized non-dominated sorting. Returns list of fronts (indices)."""
        n = len(F)
        domination_count = np.zeros(n, dtype=int)
        dominated_set = [[] for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    dominated_set[i].append(j)
                    domination_count[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated_set[j].append(i)
                    domination_count[i] += 1
        
        fronts = []
        current_front = np.where(domination_count == 0)[0]
        while len(current_front) > 0:
            fronts.append(current_front)
            next_front = set()
            for i in current_front:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.add(j)
            current_front = np.array(list(next_front), dtype=int)
        
        return fronts

    def _crowding_distance(self, F, indices):
        """Compute crowding distance for given indices."""
        n = len(indices)
        if n == 0:
            return np.array([])
        if n == 1:
            return np.array([np.inf])
        
        cd = np.zeros(n)
        F_sub = F[indices]
        for m in range(F_sub.shape[1]):
            order = np.argsort(F_sub[:, m])
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            for i in range(1, n - 1):
                cd[order[i]] += (F_sub[order[i+1], m] - F_sub[order[i-1], m]) / (F_sub[:, m].max() - F_sub[:, m].min() + 1e-10)
        
        return cd

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit with 20s safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)
        
        pop_size = 30
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Track all evaluated points for hypervolume
        all_X = [X.copy()]
        all_F = [F.copy()]
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Find non-dominated front
            fronts = self._fast_non_dominated_sort(F)
            front0 = fronts[0] if len(fronts) > 0 else np.arange(len(F))
            
            # Compute crowding distance on first front
            cd = self._crowding_distance(F, front0)
            
            # Select parent from first front using tournament with crowding
            if len(front0) > 1:
                # Tournament selection
                candidates = rng.choice(len(front0), size=min(3, len(front0)), replace=False)
                best_idx = candidates[np.argmax(cd[candidates])]
                parent_idx = front0[best_idx]
            else:
                parent_idx = front0[0]
            
            # Adaptive mutation: larger steps early, smaller steps late
            gen = used - pop_size
            total_gen = max(1, self.budget - pop_size)
            progress = gen / total_gen
            
            # Base mutation scale decreases with progress
            base_scale = 0.2 * (1.0 - 0.5 * progress)
            
            # Occasionally add large exploration jumps (20% chance)
            if rng.random() < 0.2:
                mutation_scale = 0.4 * (1.0 - 0.3 * progress)
            else:
                mutation_scale = base_scale
            
            child = X[parent_idx] + mutation_scale * (hi - lo) * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Replace worst point based on sum of objectives
            worst = int(np.argmax(np.sum(F, axis=1)))
            X[worst], F[worst] = child, fc
            
            # Every 15 evaluations, add a random point to maintain diversity
            if used % 15 == 0 and used < self.budget:
                if time.time() > deadline:
                    break
                rand_x = lo + (hi - lo) * rng.random(self.dim)
                rand_f = problem(rand_x)
                used += 1
                worst = int(np.argmax(np.sum(F, axis=1)))
                X[worst], F[worst] = rand_x, rand_f
        
        return X
# EVOLVE-BLOCK-END
