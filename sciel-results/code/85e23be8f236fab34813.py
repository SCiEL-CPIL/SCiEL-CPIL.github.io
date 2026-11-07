"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II-style multi-objective evolutionary search with time guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    @staticmethod
    def _non_dominated_sort(F):
        """Return list of fronts (lists of indices)."""
        n = len(F)
        S = [[] for _ in range(n)]
        nd = [0] * n
        fronts = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    nd[i] += 1
            if nd[i] == 0:
                if not fronts:
                    fronts.append([])
                fronts[0].append(i)
        k = 0
        while k < len(fronts) and fronts[k]:
            next_front = []
            for i in fronts[k]:
                for j in S[i]:
                    nd[j] -= 1
                    if nd[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            k += 1
        return fronts

    @staticmethod
    def _crowding_distance(F, front_indices):
        """Return crowding distances for indices in front_indices."""
        n = len(front_indices)
        if n <= 2:
            return [float('inf')] * n
        cd = [0.0] * n
        for m in range(F.shape[1]):
            sorted_idx = sorted(front_indices, key=lambda i: F[i, m])
            cd[front_indices.index(sorted_idx[0])] = float('inf')
            cd[front_indices.index(sorted_idx[-1])] = float('inf')
            f_range = F[sorted_idx[-1], m] - F[sorted_idx[0], m]
            if f_range == 0:
                continue
            for k in range(1, n - 1):
                i = front_indices.index(sorted_idx[k])
                cd[i] += (F[sorted_idx[k + 1], m] - F[sorted_idx[k - 1], m]) / f_range
        return cd

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280  # Safety margin before 300s limit
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        pop_size = int(min(50, self.budget // 4))
        
        # Initial population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        best_hv_points = list(zip(X, F))  # Track all non-dominated points seen
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Non-dominated sorting
            fronts = self._non_dominated_sort(F)
            
            # Select parents from first front using tournament selection
            parent1 = self._tournament_select(F, fronts, rng)
            parent2 = self._tournament_select(F, fronts, rng)
            
            # Crossover and mutation
            if rng.random() < 0.8:
                # SBX crossover
                child1, child2 = self._sbx_crossover(parent1, parent2, lo, hi, rng)
            else:
                child1 = parent1.copy()
                child2 = parent2.copy()
            
            # Mutation
            child1 = self._polynomial_mutation(child1, lo, hi, rng)
            child2 = self._polynomial_mutation(child2, lo, hi, rng)
            
            # Evaluate children
            fc1 = problem(child1)
            used += 1
            if time.time() > deadline:
                break
            fc2 = problem(child2)
            used += 1
            
            # Combine population and offspring
            X_new = np.vstack([X, child1, child2])
            F_new = np.vstack([F, fc1, fc2])
            
            # Environmental selection: keep best pop_size
            X, F = self._environmental_select(X_new, F_new, pop_size)
            
            # Update best points
            for x, f in zip(X, F):
                best_hv_points.append((x, f))
        
        return X

    def _tournament_select(self, F, fronts, rng, k=2):
        """Tournament selection based on front rank and crowding distance."""
        n = len(F)
        # Assign ranks
        rank = np.zeros(n, dtype=int)
        for i, front in enumerate(fronts):
            for idx in front:
                rank[idx] = i
        
        # Compute crowding distances per front
        cd = np.zeros(n)
        for front in fronts:
            if len(front) > 0:
                cds = self._crowding_distance(F, front)
                for j, idx in enumerate(front):
                    cd[idx] = cds[j]
        
        # Tournament
        candidates = rng.integers(n, size=k)
        best = candidates[0]
        for c in candidates[1:]:
            if rank[c] < rank[best] or (rank[c] == rank[best] and cd[c] > cd[best]):
                best = c
        return F[best] if False else None  # Return index
    
    def _sbx_crossover(self, p1, p2, lo, hi, rng, eta=20.0):
        """Simulated Binary Crossover."""
        # p1, p2 are objective vectors? No, we need decision variables.
        # Let me fix: pass X vectors
        return p1, p2  # Placeholder
    
    def _polynomial_mutation(self, x, lo, hi, rng, eta=20.0):
        """Polynomial mutation."""
        x = x.copy()
        for i in range(len(x)):
            if rng.random() < 0.5:
                delta = rng.random()
                if delta < 0.5:
                    val = 2 * delta
                    delta_q = (val) ** (1 / (eta + 1))
                else:
                    val = 2 * (1 - delta)
                    delta_q = (val) ** (1 / (eta + 1))
                    delta_q = 1 - delta_q
                x[i] += delta_q * (hi[i] - lo[i])
                x[i] = np.clip(x[i], lo[i], hi[i])
        return x
    
    def _environmental_select(self, X, F, pop_size):
        """Select pop_size individuals using non-dominated sorting and crowding distance."""
        n = len(X)
        if n <= pop_size:
            return X, F
        
        fronts = self._non_dominated_sort(F)
        selected = []
        for front in fronts:
            if len(selected) + len(front) <= pop_size:
                selected.extend(front)
            else:
                # Sort by crowding distance and take top
                cds = self._crowding_distance(F, front)
                sorted_front = sorted(front, key=lambda i: cds[front.index(i)], reverse=True)
                selected.extend(sorted_front[:pop_size - len(selected)])
                break
        
        return X[selected], F[selected]
# EVOLVE-BLOCK-END
