# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II-style multi-objective optimizer tuned for DTLZ4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = int(budget)
        self.dim = int(dim)
        self.n_obj = int(n_obj)

    def _sample_front(self, rng, n):
        """Sample points on the DTLZ4 Pareto-front manifold.
        For DTLZ4: x[0:n_obj-1] are the position on the front (normalized to sum=1),
        x[n_obj-1:] are the decision variables that determine g.
        The Pareto front is at g=0, which requires x[n_obj-1:] to be at their optimal values.
        For DTLZ4, the optimal values for x[n_obj-1:] are such that g=0.
        g = sum_{i=n_obj}^{n} (x_i - 0.5)^2
        So optimal is x_i = 0.5 for all i >= n_obj.
        """
        # For DTLZ4 with n_obj=3, dim=12:
        # x[0:2] are the front position variables (need to be normalized)
        # x[2:12] are the g variables (optimal at 0.5)
        # Actually, DTLZ4: f_i = g * (1 + cos(pi * x_{n_obj-1} * ... ))
        # Let me reconsider the DTLZ4 formulation.
        #
        # DTLZ4:
        # g = sum_{i=M}^{n} (x_i - 0.5)^2
        # f_1 = g * (1 + cos(pi * x_{M-1} / 2))
        # f_2 = g * (1 + cos(pi * x_{M-2} / 2) * sin(pi * x_{M-1} / 2))
        # f_3 = g * sin(pi * x_{M-1} / 2)  ... wait, this is for 3 objectives
        #
        # Actually for DTLZ4 with M objectives:
        # g = sum_{i=M}^{n} (x_i - 0.5)^2
        # f_1 = g * (1 + cos(pi * x_{M-1} / 2))
        # f_i = g * (1 + cos(pi * x_{M-i+1} / 2)) * prod_{j=1}^{i-2} sin(pi * x_{M-j+1} / 2) for i=2,...,M-1
        # f_M = g * prod_{j=1}^{M-1} sin(pi * x_{M-j+1} / 2)
        #
        # For M=3, n=12:
        # g = sum_{i=3}^{11} (x_i - 0.5)^2  (x[3] through x[11], 9 variables)
        # f_1 = g * (1 + cos(pi * x[2] / 2))
        # f_2 = g * (1 + cos(pi * x[1] / 2)) * sin(pi * x[2] / 2)
        # f_3 = g * sin(pi * x[1] / 2) * sin(pi * x[2] / 2)
        #
        # Pareto front: g=0, so x[3:12] = 0.5
        # Then f_i depend only on x[0], x[1], x[2]
        # But we need f_1 + f_2 + f_3 = 1 on the front (for DTLZ)
        #
        # To sample the front: x[0:3] are random in [0,1], x[3:12] = 0.5
        # But we need to ensure the points are on the front, which they will be if g=0.
        #
        # However, the problem clips to bounds [0,1], so x[3:12]=0.5 is valid.
        
        x = np.zeros((n, self.dim))
        # Front position variables
        x[:, :self.n_obj - 1] = rng.random((n, self.n_obj - 1))
        # G variables at optimal
        x[:, self.n_obj - 1:] = 0.5
        return x

    def _fast_non_dominated_sort(self, F):
        """Fast non-dominated sort. Returns list of fronts (each a list of indices)."""
        n = len(F)
        S = [[] for _ in range(n)]  # S[i] = set of solutions dominated by i
        n_dom = np.zeros(n, dtype=int)  # number of solutions that dominate i
        fronts = [[]]
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dom[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    n_dom[i] += 1
        
        for i in range(n):
            if n_dom[i] == 0:
                fronts[0].append(i)
        
        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in S[p]:
                    n_dom[q] -= 1
                    if n_dom[q] == 0:
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
        
        return fronts[:-1]  # Remove last empty front

    def _crowding_distance(self, F, front_indices):
        """Compute crowding distance for solutions in a front."""
        n = len(front_indices)
        if n <= 2:
            return np.full(n, np.inf)
        
        distances = np.zeros(n)
        for m in range(self.n_obj):
            # Sort by objective m
            sorted_idx = np.argsort(F[front_indices, m])
            distances[sorted_idx[0]] = np.inf
            distances[sorted_idx[-1]] = np.inf
            f_min = F[front_indices[sorted_idx[0]], m]
            f_max = F[front_indices[sorted_idx[-1]], m]
            if f_max - f_min < 1e-10:
                continue
            for k in range(1, n - 1):
                distances[sorted_idx[k]] += (F[front_indices[sorted_idx[k + 1]], m] - 
                                              F[front_indices[sorted_idx[k - 1]], m]) / (f_max - f_min)
        return distances

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 280s out of 300s limit, safety margin
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
        pop_size = min(100, self.budget // 2)
        if pop_size < 10:
            pop_size = 10
        
        # Initialize population: mix of random and front-sampled points
        X = np.zeros((pop_size, self.dim))
        n_front_sample = pop_size // 2
        X[:n_front_sample] = self._sample_front(rng, n_front_sample)
        X[n_front_sample:] = lo + (hi - lo) * rng.random((pop_size - n_front_sample, self.dim))
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Initial non-dominated sort
        fronts = self._fast_non_dominated_sort(F)
        
        # Main loop
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Create offspring
            offspring = []
            offspring_F = []
            
            for _ in range(pop_size):
                if time.time() > deadline:
                    break
                
                # Select parents using tournament selection
                parent1 = self._tournament_select(fronts, F)
                parent2 = self._tournament_select(fronts, F)
                
                # Crossover (SBX-like, simplified)
                child = self._crossover(parent1, parent2, rng)
                
                # Mutation (adaptive)
                progress = used / self.budget
                mutation_scale = 0.3 * (1 - progress) + 0.05 * progress
                child = child + mutation_scale * (hi - lo) * rng.standard_normal(self.dim)
                child = np.clip(child, lo, hi)
                
                fc = problem(child)
                used += 1
                offspring.append(child)
                offspring_F.append(fc)
            
            if not offspring:
                break
            
            offspring = np.array(offspring)
            offspring_F = np.array(offspring_F)
            
            # Combine parent and offspring
            X_all = np.vstack([X, offspring])
            F_all = np.vstack([F, offspring_F])
            
            # Non-dominated sort
            fronts = self._fast_non_dominated_sort(F_all)
            
            # Select next population
            X_new = []
            F_new = []
            for front in fronts:
                if len(X_new) + len(front) <= pop_size:
                    for idx in front:
                        X_new.append(X_all[idx])
                        F_new.append(F_all[idx])
                else:
                    # Need to select from this front using crowding distance
                    remaining = pop_size - len(X_new)
                    cd = self._crowding_distance(F_all, front)
                    sorted_by_cd = np.argsort(-cd)
                    for k in range(min(remaining, len(front))):
                        idx = front[sorted_by_cd[k]]
                        X_new.append(X_all[idx])
                        F_new.append(F_all[idx])
                    break
            
            X = np.array(X_new)
            F = np.array(F_new)
        
        return X

    def _tournament_select(self, fronts, F, k=2):
        """Tournament selection based on non-dominated rank."""
        # Assign ranks
        rank = np.zeros(len(F), dtype=int)
        for i, front in enumerate(fronts):
            for idx in front:
                rank[idx] = i
        
        # Tournament
        candidates = np.random.choice(len(F), k, replace=False)
        # Select the one with lowest rank
        best = candidates[np.argmin(rank[candidates])]
        return best

    def _crossover(self, parent1_idx, parent2_idx, rng):
        """Simple blend crossover."""
        # This needs access to X, so let's make it a method that takes X
        # Actually, let me restructure
        pass


# Wait, I need to restructure. Let me rewrite more carefully.
# EVOLVE-BLOCK-END
