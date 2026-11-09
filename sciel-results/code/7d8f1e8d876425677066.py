"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective evolutionary optimizer using NSGA-II-like selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Population size
        pop_size = min(20, self.budget // 2)
        if pop_size < 2:
            pop_size = 2
            
        # Initialize population
        X = lo + range_ * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Helper functions
        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] contains indices dominated by i
            n_dom = np.zeros(n, dtype=int)  # number of solutions dominating i
            rank = np.zeros(n, dtype=int)
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dom[i] += 1
            
            current_front = [i for i in range(n) if n_dom[i] == 0]
            current_rank = 0
            while current_front:
                for i in current_front:
                    rank[i] = current_rank
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                current_front = next_front
                current_rank += 1
            return rank

        def crowding_distance(F, rank):
            n = len(F)
            dist = np.zeros(n)
            for r in set(rank):
                front = [i for i in range(n) if rank[i] == r]
                if len(front) <= 2:
                    for i in front:
                        dist[i] = float('inf')
                    continue
                for m in range(self.n_obj):
                    sorted_front = sorted(front, key=lambda i: F[i][m])
                    dist[sorted_front[0]] = float('inf')
                    dist[sorted_front[-1]] = float('inf')
                    f_min = F[sorted_front[0]][m]
                    f_max = F[sorted_front[-1]][m]
                    if f_max - f_min == 0:
                        continue
                    for k in range(1, len(sorted_front) - 1):
                        i = sorted_front[k]
                        dist[i] += (F[sorted_front[k+1]][m] - F[sorted_front[k-1]][m]) / (f_max - f_min)
            return dist

        def tournament_select(rank, dist, k=2):
            idx1 = rng.integers(len(rank))
            idx2 = rng.integers(len(rank))
            if rank[idx1] < rank[idx2]:
                return idx1
            elif rank[idx2] < rank[idx1]:
                return idx2
            else:
                if dist[idx1] > dist[idx2]:
                    return idx1
                else:
                    return idx2

        def sbx_crossover(x1, x2, eta=20.0):
            x1 = np.asarray(x1)
            x2 = np.asarray(x2)
            u = rng.random(self.dim)
            c1 = np.zeros(self.dim)
            c2 = np.zeros(self.dim)
            for i in range(self.dim):
                if u[i] <= 0.5:
                    beta = (2 * u[i]) ** (1.0 / (eta + 1.0))
                else:
                    beta = (1.0 / (2 * (1.0 - u[i]))) ** (1.0 / (eta + 1.0))
                c1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                c2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
            return c1, c2

        while used < self.budget:
            if time.time() > deadline:
                break
            
            rank = fast_non_dominated_sort(F)
            dist = crowding_distance(F, rank)
            
            # Generate next generation
            new_X = []
            new_F = []
            
            # We generate pairs of children to fill the population
            # To keep it simple and within budget, we generate one child per iteration 
            # and replace the worst in the current population based on rank/dist?
            # Or better: keep a population of size pop_size.
            # Let's stick to a steady-state approach to ensure we don't blow budget or time.
            # Select 2 parents
            p1 = tournament_select(rank, dist)
            p2 = tournament_select(rank, dist)
            
            # Crossover
            if rng.random() < 0.9:
                c1, c2 = sbx_crossover(X[p1], X[p2])
            else:
                c1 = X[p1].copy()
                c2 = X[p2].copy()
            
            # Mutation
            # Adaptive mutation: start large, shrink
            # Simple: Gaussian
            sigma = 0.1 * (1.0 - used / self.budget) + 0.01
            c1 = c1 + sigma * range_ * rng.standard_normal(self.dim)
            c2 = c2 + sigma * range_ * rng.standard_normal(self.dim)
            
            c1 = np.clip(c1, lo, hi)
            c2 = np.clip(c2, lo, hi)
            
            # Evaluate
            f1 = problem(c1)
            used += 1
            if used > self.budget: break
            
            f2 = problem(c2)
            used += 1
            if used > self.budget: break
            
            # Update population
            # We need to decide which individuals to replace.
            # Simple strategy: Replace the individual with the worst rank (and lowest dist if tie) in the current population.
            # To do this efficiently, we can just append to a temporary list and sort, but that's O(N log N).
            # Given N=20, it's fine.
            
            # Create a combined list of current pop and new children
            # Actually, let's just replace the two worst individuals in the current population.
            
            # Identify worst 2
            # Sort indices by (rank, -dist)
            indices = np.argsort(rank) # lower rank is better
            # We want to remove the ones with highest rank (worst)
            # If rank is same, remove one with lowest dist (worst diversity)
            
            # Let's just find the worst single individual to replace one child, 
            # but we have 2 children. Let's replace 2 worst.
            
            # Create a list of (rank, -dist, index)
            keys = [(rank[i], -dist[i], i) for i in range(len(F))]
            keys.sort(reverse=True) # Worst first
            worst_idx_1 = keys[0][2]
            worst_idx_2 = keys[1][2]
            
            # Replace
            X[worst_idx_1] = c1
            F[worst_idx_1] = f1
            X[worst_idx_2] = c2
            F[worst_idx_2] = f2

        return X
# EVOLVE-BLOCK-END
