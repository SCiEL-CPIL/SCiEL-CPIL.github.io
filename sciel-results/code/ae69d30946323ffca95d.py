"""Multi-objective optimizer on ZDT2 using NSGA-II."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """NSGA-II based multi-objective optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
        pop_size = min(20, self.budget)
        if pop_size < 2:
            pop_size = 2
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # NSGA-II parameters
        mutation_rate = 0.1
        crossover_rate = 0.8
        
        while used < self.budget:
            # Fast non-dominated sorting
            fronts = self._fast_non_dominated_sort(F)
            
            # Crowding distance
            crowd = self._crowding_distance(F)
            
            # Select parents using tournament selection
            parents = self._tournament_select(fronts, crowd, F, pop_size, rng)
            
            # Generate offspring
            offspring_X = []
            offspring_F = []
            
            n_offspring = min(pop_size, self.budget - used)
            
            for _ in range(n_offspring):
                if len(parents) < 2:
                    # Not enough parents, just mutate
                    p1 = parents[0] if len(parents) > 0 else X[0]
                    child = p1 + mutation_rate * (hi - lo) * rng.standard_normal(self.dim)
                    child = np.clip(child, lo, hi)
                    fc = problem(child)
                    used += 1
                    offspring_X.append(child)
                    offspring_F.append(fc)
                else:
                    # Select two parents via tournament
                    p1_idx = self._tournament(fronts, crowd, F, rng)
                    p2_idx = self._tournament(fronts, crowd, F, rng)
                    p1 = X[p1_idx]
                    p2 = X[p2_idx]
                    
                    # Simulated binary crossover
                    if rng.random() < crossover_rate:
                        child1, child2 = self._sbx_crossover(p1, p2, lo, hi, rng)
                    else:
                        child1 = p1.copy()
                        child2 = p2.copy()
                    
                    # Mutation
                    child1 = self._gaussian_mutation(child1, lo, hi, rng, mutation_rate)
                    child2 = self._gaussian_mutation(child2, lo, hi, rng, mutation_rate)
                    
                    fc1 = problem(child1)
                    used += 1
                    fc2 = problem(child2)
                    used += 1
                    
                    offspring_X.append(child1)
                    offspring_F.append(fc2)
                    
                    # If we've used all budget, break
                    if used >= self.budget:
                        break
            
            if not offspring_X:
                break
            
            offspring_F = np.array(offspring_F)
            
            # Combine parent and offspring populations
            combined_X = np.vstack([X, np.array(offspring_X)])
            combined_F = np.vstack([F, offspring_F])
            
            # Fast non-dominated sorting on combined
            combined_fronts = self._fast_non_dominated_sort(combined_F)
            combined_crowd = self._crowding_distance(combined_F)
            
            # Select next generation
            next_X = []
            next_F = []
            for front in combined_fronts:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                else:
                    # Need to select some from this front based on crowding
                    remaining = pop_size - len(next_X)
                    if remaining > 0:
                        # Sort by crowding distance in descending order
                        front_indices = list(front)
                        front_crowd = [combined_crowd[i] for i in front_indices]
                        sorted_indices = sorted(range(len(front_indices)), key=lambda i: -front_crowd[i])
                        for i in sorted_indices[:remaining]:
                            idx = front_indices[i]
                            next_X.append(combined_X[idx])
                            next_F.append(combined_F[idx])
                    break
            
            if len(next_X) == 0:
                break
                
            X = np.array(next_X)
            F = np.array(next_F)
        
        return X

    def _fast_non_dominated_sort(self, F):
        """Fast non-dominated sorting. Returns list of fronts, each a list of indices."""
        n = len(F)
        fronts = [[] for _ in range(n)]
        domination_count = np.zeros(n, dtype=int)
        dominated_by = [[] for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    dominated_by[i].append(j)
                    domination_count[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated_by[j].append(i)
                    domination_count[i] += 1
        
        # First front
        for i in range(n):
            if domination_count[i] == 0:
                fronts[0].append(i)
        
        k = 0
        while fronts[k]:
            next_front = []
            for i in fronts[k]:
                for j in dominated_by[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            k += 1
            fronts[k] = next_front
        
        return [f for f in fronts if f]

    def _crowding_distance(self, F):
        """Compute crowding distance for each point."""
        n = len(F)
        if n == 0:
            return np.array([])
        if n == 1:
            return np.array([np.inf])
        
        crowd = np.zeros(n)
        for m in range(F.shape[1]):
            # Sort by objective m
            sorted_idx = np.argsort(F[:, m])
            crowd[sorted_idx[0]] = np.inf
            crowd[sorted_idx[-1]] = np.inf
            f_min = F[sorted_idx[0], m]
            f_max = F[sorted_idx[-1], m]
            if f_max - f_min > 1e-10:
                for i in range(1, n - 1):
                    crowd[sorted_idx[i]] += (F[sorted_idx[i + 1], m] - F[sorted_idx[i - 1], m]) / (f_max - f_min)
        
        return crowd

    def _tournament_select(self, fronts, crowd, F, k, rng):
        """Select k parents via tournament selection."""
        selected = []
        n = len(F)
        for _ in range(k):
            idx = self._tournament(fronts, crowd, F, rng)
            selected.append(idx)
        return selected

    def _tournament(self, fronts, crowd, F, rng):
        """Tournament selection: pick 2 random, return the better one."""
        n = len(F)
        i = rng.integers(n)
        j = rng.integers(n)
        
        # Find front rank for i and j
        rank_i = self._get_rank(i, fronts)
        rank_j = self._get_rank(j, fronts)
        
        if rank_i < rank_j:
            return i
        elif rank_j < rank_i:
            return j
        else:
            # Same front, use crowding distance
            if crowd[i] > crowd[j]:
                return i
            else:
                return j

    def _get_rank(self, idx, fronts):
        """Get the front rank (0-based) of a given index."""
        for rank, front in enumerate(fronts):
            if idx in front:
                return rank
        return len(fronts)  # Fallback

    def _sbx_crossover(self, p1, p2, lo, hi, rng, eta=20.0):
        """Simulated binary crossover."""
        child1 = p1.copy()
        child2 = p2.copy()
        
        for i in range(self.dim):
            if rng.random() < 0.5:
                if abs(p1[i] - p2[i]) > 1e-10:
                    if rng.random() < 0.5:
                        beta = 1 + 2 * (p1[i] - lo[i]) / (p2[i] - p1[i])
                    else:
                        beta = 1 + 2 * (hi[i] - p2[i]) / (p2[i] - p1[i])
                    alpha = 2 - beta ** (-(eta + 1))
                    
                    if rng.random() < 1.0 / alpha:
                        beta_q = (0.5 * beta) ** (1.0 / (eta + 1))
                    else:
                        beta_q = (0.5 / (alpha - beta)) ** (1.0 / (eta + 1))
                    
                    child1[i] = 0.5 * ((1 + beta_q) * p1[i] + (1 - beta_q) * p2[i])
                    child2[i] = 0.5 * ((1 - beta_q) * p1[i] + (1 + beta_q) * p2[i])
        
        child1 = np.clip(child1, lo, hi)
        child2 = np.clip(child2, lo, hi)
        
        return child1, child2

    def _gaussian_mutation(self, x, lo, hi, rng, rate):
        """Gaussian mutation."""
        x = x.copy()
        for i in range(self.dim):
            if rng.random() < 0.5:
                sigma = (hi[i] - lo[i]) * rate
                x[i] += rng.standard_normal() * sigma
        return np.clip(x, lo, hi)
# EVOLVE-BLOCK-END
