# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 exploiting known front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _dominance_mask(self, F):
        """Return a boolean array where True means the point is dominated by at least one other."""
        n = len(F)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated[i] = True
                    break
        return dominated

    def _non_dominated_sort(self, F):
        """Return list of fronts (indices), sorted by dominance level."""
        n = len(F)
        S = [[] for _ in range(n)]  # S[i] = set of points dominated by i
        n_count = np.zeros(n, dtype=int)  # number of points that dominate i
        fronts = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    n_count[i] += 1
        front0 = [i for i in range(n) if n_count[i] == 0]
        fronts.append(front0)
        k = 0
        while fronts[k]:
            next_front = []
            for i in fronts[k]:
                for j in S[i]:
                    n_count[j] -= 1
                    if n_count[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
                k += 1
        return fronts

    def _crowding_distance(self, F, indices):
        """Compute crowding distance for a set of indices in F."""
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)
        cd = np.zeros(n)
        F_sub = F[indices]
        for m in range(self.n_obj):
            min_val = F_sub[:, m].min()
            max_val = F_sub[:, m].max()
            if max_val - min_val < 1e-15:
                continue
            order = np.argsort(F_sub[:, m])
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            for i in range(1, n - 1):
                cd[order[i]] += (F_sub[order[i + 1], m] - F_sub[order[i - 1], m]) / (max_val - min_val)
        return cd

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        # Initial population: mix of random and front-biased points
        pop_size = min(40, self.budget // 5)
        X = np.zeros((pop_size, self.dim))
        
        # Half random
        n_rand = pop_size // 2
        X[:n_rand] = lo + (hi - lo) * rng.random((n_rand, self.dim))
        
        # Half: bias toward DTLZ4 Pareto front structure
        # For DTLZ4: x_1..x_{n_obj-1} are the front variables, x_{n_obj} should be ~1
        # f_i = x_i for i=1,2; f_3 = 1 - x_3 (when x_3 close to 1)
        n_front = pop_size - n_rand
        for i in range(n_front):
            x = np.zeros(self.dim)
            # Sample on simplex for first n_obj-1 objectives
            if self.n_obj >= 2:
                # Dirichlet-like sampling
                u = rng.random(self.n_obj - 1)
                u = u / u.sum()
                x[0:self.n_obj - 1] = u
                # x_{n_obj-1} (0-indexed: index n_obj-1) should be close to 1
                x[self.n_obj - 1] = 1.0 - 0.05 * rng.random()
                # Remaining dimensions random
                if self.dim > self.n_obj:
                    x[self.n_obj:] = 0.5 + 0.5 * rng.random(self.dim - self.n_obj)
            else:
                x = lo + (hi - lo) * rng.random(self.dim)
            X[n_rand + i] = np.clip(x, lo, hi)

        F = np.array([problem(x) for x in X])
        used = pop_size

        # Adaptive mutation scale
        def mutation_scale(used_frac):
            # Start large, decrease
            return 0.3 * (1.0 - 0.7 * used_frac)

        # Main loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Non-dominated sort
            fronts = self._non_dominated_sort(F)
            
            # Select parents from first few fronts
            parent_pool = []
            for front in fronts[:2]:
                parent_pool.extend(front)
            if not parent_pool:
                parent_pool = list(range(len(F)))
            
            # Select two parents
            p1 = parent_pool[rng.integers(len(parent_pool))]
            p2 = parent_pool[rng.integers(len(parent_pool))]

            # Crossover: blend + mutation
            alpha = rng.random()
            child = alpha * X[p1] + (1 - alpha) * X[p2]
            
            # Mutation
            scale = mutation_scale(used / self.budget)
            child += scale * (hi - lo) * rng.standard_normal(self.dim)
            
            # Occasionally do a larger jump
            if rng.random() < 0.1:
                child += 0.5 * (hi - lo) * rng.standard_normal(self.dim)
            
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1

            # Replace worst point in the population
            # Use a combined criterion: dominated points first, then worst crowding
            dominated = self._dominance_mask(F)
            if np.any(dominated):
                # Among dominated, pick the one with worst sum
                dom_indices = np.where(dominated)[0]
                worst = dom_indices[np.argmax(np.sum(F[dom_indices], axis=1))]
            else:
                # All non-dominated: use crowding distance to find least crowded
                cd = self._crowding_distance(F, list(range(len(F))))
                worst = np.argmin(cd)

            X[worst] = child
            F[worst] = fc

        return X
# EVOLVE-BLOCK-END
