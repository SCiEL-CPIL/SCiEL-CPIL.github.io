"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Multi-objective optimizer using NSGA-II-inspired approach with:
    - Diverse random initialization
    - Tournament selection from non-dominated fronts
    - Adaptive mutation
    - Crowding distance for diversity maintenance
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _non_dominated_sort(self, F):
        """Perform non-dominated sorting on F (n x m array)."""
        n = len(F)
        m = self.n_obj
        domination_count = np.zeros(n, dtype=int)
        dominated_list = [[] for _ in range(n)]
        fronts = []
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    dominated_list[i].append(j)
                    domination_count[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated_list[j].append(i)
                    domination_count[i] += 1
        
        current_front = np.where(domination_count == 0)[0]
        while len(current_front) > 0:
            fronts.append(current_front.copy())
            next_front = set()
            for i in current_front:
                for j in dominated_list[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.add(j)
            current_front = np.array(sorted(next_front)) if next_front else np.array([], dtype=int)
        
        # Assign front indices
        front_idx = np.full(n, -1, dtype=int)
        for fi, front in enumerate(fronts):
            for i in front:
                front_idx[i] = fi
        return front_idx

    def _crowding_distance(self, F, front):
        """Compute crowding distance for points in a front."""
        n = len(front)
        if n <= 2:
            return np.full(n, np.inf)
        
        dist = np.zeros(n)
        for k in range(self.n_obj):
            obj_vals = F[front, k]
            order = np.argsort(obj_vals)
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf
            obj_range = obj_vals[order[-1]] - obj_vals[order[0]]
            if obj_range > 1e-15:
                for i in range(1, n - 1):
                    dist[order[i]] += (obj_vals[order[i + 1]] - obj_vals[order[i - 1]]) / obj_range
        return dist

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj
        
        # Population size
        pop_size = min(60, max(15, self.budget // 3))
        
        # Initialize with diverse random points
        X = lo + (hi - lo) * rng.random((pop_size, n))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Main evolutionary loop
        while used < self.budget:
            # Non-dominated sort
            front_idx = self._non_dominated_sort(F)
            
            # Compute crowding distance for each front
            crowd = np.zeros(pop_size)
            unique_fronts = np.unique(front_idx)
            for fi in unique_fronts:
                mask = front_idx == fi
                front_indices = np.where(mask)[0]
                if len(front_indices) > 0:
                    cd = self._crowding_distance(F, front_indices)
                    crowd[front_indices] = cd
            
            # Select parents using tournament selection
            parent1_idx = self._tournament_select(front_idx, crowd, rng)
            parent2_idx = self._tournament_select(front_idx, crowd, rng)
            
            p1 = X[parent1_idx]
            p2 = X[parent2_idx]
            
            # SBX crossover
            if rng.random() < 0.9:
                child1, child2 = self._sbx_crossover(p1, p2, lo, hi, rng)
            else:
                child1 = p1.copy()
                child2 = p2.copy()
            
            # Mutation
            child1 = self._polynomial_mutation(child1, lo, hi, rng)
            child2 = self._polynomial_mutation(child2, lo, hi, rng)
            
            # Evaluate
            f1 = problem(child1)
            used += 1
            if used >= self.budget:
                # Replace worst point
                worst_idx = int(np.argmax(np.sum(F, axis=1)))
                X[worst_idx] = child1
                F[worst_idx] = f1
                break
            f2 = problem(child2)
            used += 1
            
            # Replace worst two points
            worst_indices = np.argsort(np.sum(F, axis=1))[-2:]
            X[worst_indices[0]] = child1
            F[worst_indices[0]] = f1
            X[worst_indices[1]] = child2
            F[worst_indices[1]] = f2
        
        return X

    def _tournament_select(self, front_idx, crowd, rng, k=3):
        """Tournament selection based on front and crowding distance."""
        n = len(front_idx)
        candidates = rng.integers(n, size=k)
        best = candidates[0]
        for i in range(1, k):
            c = candidates[i]
            if front_idx[c] < front_idx[best]:
                best = c
            elif front_idx[c] == front_idx[best] and crowd[c] > crowd[best]:
                best = c
        return best

    def _sbx_crossover(self, p1, p2, lo, hi, rng, eta=20):
        """Simulated Binary Crossover."""
        n = len(p1)
        c1 = np.zeros(n)
        c2 = np.zeros(n)
        for i in range(n):
            if rng.random() < 0.5:
                if abs(p2[i] - p1[i]) > 1e-14:
                    if rng.random() < 0.5:
                        beta = 1 + 2 * (p1[i] - lo[i]) / (p2[i] - p1[i])
                        alpha = 2 - beta ** (-eta - 1)
                        if rng.random() <= 1 / alpha:
                            beta_q = beta ** (-1 / eta)
                        else:
                            beta_q = (1 / (2 - 2 * beta)) ** (1 / (eta + 1))
                        c1[i] = 0.5 * ((p1[i] + p2[i]) - beta_q * (p2[i] - p1[i]))
                    else:
                        beta = 1 + 2 * (hi[i] - p2[i]) / (p2[i] - p1[i])
                        alpha = 2 - beta ** (-eta - 1)
                        if rng.random() <= 1 / alpha:
                            beta_q = beta ** (-1 / eta)
                        else:
                            beta_q = (1 / (2 - 2 * beta)) ** (1 / (eta + 1))
                        c1[i] = 0.5 * ((p1[i] + p2[i]) + beta_q * (p2[i] - p1[i]))
                else:
                    c1[i] = p1[i]
                c2[i] = p2[i] if c1[i] != p1[i] else p1[i]
            else:
                c1[i] = p1[i]
                c2[i] = p2[i]
        
        # Clip to bounds
        c1 = np.clip(c1, lo, hi)
        c2 = np.clip(c2, lo, hi)
        return c1, c2

    def _polynomial_mutation(self, x, lo, hi, rng, eta=20):
        """Polynomial mutation."""
        n = len(x)
        xm = x.copy()
        for i in range(n):
            if rng.random() < 0.5:
                if hi[i] - lo[i] < 1e-14:
                    continue
                delta1 = (xm[i] - lo[i]) / (hi[i] - lo[i])
                delta2 = (hi[i] - xm[i]) / (hi[i] - lo[i])
                mut_pow = 1 / (eta + 1)
                u = rng.random()
                if u < 0.5:
                    xy = 1 - delta1
                    val = 2 * u + (1 - 2 * u) * (xy) ** (eta + 1)
                    deltaq = val ** mut_pow - 1
                else:
                    xy = 1 - delta2
                    val = 2 * (1 - u) + 2 * (u - 0.5) * (xy) ** (eta + 1)
                    deltaq = 1 - val ** mut_pow
                xm[i] = xm[i] + deltaq * (hi[i] - lo[i])
                xm[i] = np.clip(xm[i], lo[i], hi[i])
        return xm

# EVOLVE-BLOCK-END
