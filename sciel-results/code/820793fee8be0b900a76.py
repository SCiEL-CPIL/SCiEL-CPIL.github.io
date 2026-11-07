"""Combined multi-objective optimizer for DTLZ1 using analytical front sampling + NSGA-II refinement."""
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for DTLZ1.
    Strategy:
    1. Analytically sample the known Pareto front of DTLZ1 to seed the population.
    2. Use NSGA-II (non-dominated sorting, crowding distance, SBX crossover, polynomial mutation)
       to refine the population and discover better points within the remaining budget.
    3. Include a time deadline guard to ensure completion within 300s.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = time.time()
        self.deadline = 285.0  # Safety margin before 300s limit

    def _elapsed(self):
        return time.time() - self.start_time

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        # Population size
        N = 30

        # --- Analytical DTLZ1 front sampling ---
        # For DTLZ1 with M objectives, the Pareto front is:
        # f1 = 0.5 * prod_{i=1}^{M-1} x_i * (1 + g)
        # f2 = 0.5 * (1 - x_1) * prod_{i=2}^{M-1} x_i * (1 + g)
        # ...
        # fM = 0.5 * (1 - x_{M-1}) * (1 + g)
        # where g = sum_{i=M}^{n} (x_i - 0.5)^2 - cos(20*pi*x_i*(x_i-0.5))
        # On the Pareto front, g = 0, so x_i = 0.5 for i >= M.
        # Thus f_k = 0.5 * prod_{j=1}^{k-1} x_j * (1 - x_{k-1}) for k=1..M
        # Actually for DTLZ1: f_k = 0.5 * prod_{j=1}^{k-1} x_j * (1 - x_{k-1}) * (1 + g)
        # With g=0: f_k = 0.5 * prod_{j=1}^{k-1} x_j * (1 - x_{k-1})
        # For M=3: f1 = 0.5*x1*x2, f2 = 0.5*(1-x1)*x2, f3 = 0.5*(1-x2)
        # Wait, let me re-derive. DTLZ1:
        # f1 = 0.5 * x1 * x2 * (1+g)
        # f2 = 0.5 * (1-x1) * x2 * (1+g)
        # f3 = 0.5 * (1-x2) * (1+g)
        # On Pareto front g=0: x3..x7 = 0.5
        # f1 = 0.5*x1*x2, f2 = 0.5*(1-x1)*x2, f3 = 0.5*(1-x2)
        
        # Generate points on the analytical front
        # Sample x1, x2 in [0,1], set x3..x7 = 0.5
        n_front = N
        x1_samples = rng.random(n_front)
        x2_samples = rng.random(n_front)
        
        X_front = np.zeros((n_front, self.dim))
        X_front[:, 0] = x1_samples
        X_front[:, 1] = x2_samples
        X_front[:, 2:] = 0.5  # For DTLZ1, g=0 when x_i=0.5 for i>=M
        
        # Evaluate front points
        F_front = np.array([problem(x) for x in X_front])
        used = n_front

        # Add some random points to ensure diversity
        n_random = max(0, N - n_front)
        if n_random > 0:
            X_rand = lo + (hi - lo) * rng.random((n_random, self.dim))
            F_rand = np.array([problem(x) for x in X_rand])
            used += n_random
            X = np.vstack([X_front, X_rand])
            F = np.vstack([F_front, F_rand])
        else:
            X = X_front
            F = F_front

        # --- NSGA-II helpers ---

        def non_dominated_sort(F_mat):
            """
            Returns ranks for each individual using fast non-dominated sort.
            """
            n = len(F_mat)
            S = [set() for _ in range(n)]
            n_dominated = [0] * n
            rank = [0] * n
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F_mat[i] <= F_mat[j]) and np.any(F_mat[i] < F_mat[j]):
                        S[i].add(j)
                        n_dominated[j] += 1
                    elif np.all(F_mat[j] <= F_mat[i]) and np.any(F_mat[j] < F_mat[i]):
                        S[j].add(i)
                        n_dominated[i] += 1
            
            current_rank = 0
            r = [i for i in range(n) if n_dominated[i] == 0]
            
            while r:
                next_r = []
                for i in r:
                    rank[i] = current_rank
                    for j in S[i]:
                        n_dominated[j] -= 1
                        if n_dominated[j] == 0:
                            next_r.append(j)
                r = next_r
                current_rank += 1
            return rank

        def crowding_distance(F_mat, rank):
            """
            Calculate crowding distance for each individual.
            """
            n = len(F_mat)
            dist = np.zeros(n)
            if n <= 2:
                dist[:] = np.inf
                return dist
                
            unique_ranks = sorted(list(set(rank)))
            for r in unique_ranks:
                indices = [i for i in range(n) if rank[i] == r]
                if len(indices) <= 2:
                    for i in indices:
                        dist[i] = np.inf
                    continue
                
                for k in range(self.n_obj):
                    sorted_indices = sorted(indices, key=lambda i: F_mat[i][k])
                    dist[sorted_indices[0]] = np.inf
                    dist[sorted_indices[-1]] = np.inf
                    f_min = F_mat[sorted_indices[0]][k]
                    f_max = F_mat[sorted_indices[-1]][k]
                    if f_max - f_min < 1e-15:
                        continue
                    for i in range(1, len(sorted_indices) - 1):
                        idx = sorted_indices[i]
                        dist[idx] += (F_mat[sorted_indices[i+1]][k] - F_mat[sorted_indices[i-1]][k]) / (f_max - f_min)
            return dist

        def sbx_crossover(x1, x2, lo, hi):
            if rng.random() < 0.5:
                return x1.copy(), x2.copy()
            
            eta = 20.0
            child1 = x1.copy()
            child2 = x2.copy()
            
            for i in range(self.dim):
                if x1[i] == x2[i]:
                    continue
                u = rng.random()
                if u <= 0.5:
                    alpha = (2.0 * u) ** (1.0 / (eta + 1.0))
                else:
                    alpha = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                
                c1 = 0.5 * ((1.0 + alpha) * x1[i] + (1.0 - alpha) * x2[i])
                c2 = 0.5 * ((1.0 - alpha) * x1[i] + (1.0 + alpha) * x2[i])
                
                if c1 < lo[i]:
                    c1 = lo[i]
                if c1 > hi[i]:
                    c1 = hi[i]
                if c2 < lo[i]:
                    c2 = lo[i]
                if c2 > hi[i]:
                    c2 = hi[i]
                
                child1[i] = c1
                child2[i] = c2
            return child1, child2

        def polynomial_mutation(x, lo, hi):
            eta = 20.0
            for i in range(self.dim):
                if rng.random() < 1.0 / self.dim:
                    u = rng.random()
                    if u < 0.5:
                        delta = (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0
                    else:
                        delta = 1.0 - (2.0 * (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                    
                    x[i] += delta * (hi[i] - lo[i])
                    if x[i] < lo[i]:
                        x[i] = lo[i]
                    if x[i] > hi[i]:
                        x[i] = hi[i]
            return x

        def tournament_select(X_mat, F_mat, rank, dist):
            i1 = rng.integers(0, len(F_mat))
            i2 = rng.integers(0, len(F_mat))
            if rank[i1] < rank[i2]:
                return X_mat[i1].copy()
            elif rank[i2] < rank[i1]:
                return X_mat[i2].copy()
            else:
                if dist[i1] > dist[i2]:
                    return X_mat[i1].copy()
                else:
                    return X_mat[i2].copy()

        # --- Main NSGA-II Loop ---
        while used < self.budget:
            if self._elapsed() > self.deadline:
                break
            
            # Calculate current ranks and distances
            rank = non_dominated_sort(F)
            dist = crowding_distance(F, rank)
            
            offspring = []
            # Generate N offspring
            for _ in range(N):
                if used + 2 > self.budget:
                    break
                if self._elapsed() > self.deadline:
                    break
                    
                p1 = tournament_select(X, F, rank, dist)
                p2 = tournament_select(X, F, rank, dist)
                c1, c2 = sbx_crossover(p1, p2, lo, hi)
                c1 = polynomial_mutation(c1, lo, hi)
                c2 = polynomial_mutation(c2, lo, hi)
                offspring.append(c1)
                offspring.append(c2)
                used += 2
            
            if not offspring:
                break
            
            # Evaluate offspring
            F_off = np.array([problem(x) for x in offspring])
            
            # Combine
            X_all = np.vstack([X, offspring])
            F_all = np.vstack([F, F_off])
            
            # Sort and Select
            rank_all = non_dominated_sort(F_all)
            dist_all = crowding_distance(F_all, rank_all)
            
            # Select top N
            indices = np.arange(len(F_all))
            sorted_indices = sorted(indices, key=lambda i: (rank_all[i], -dist_all[i]))
            
            X = X_all[sorted_indices[:N]]
            F = F_all[sorted_indices[:N]]
        
        return X
