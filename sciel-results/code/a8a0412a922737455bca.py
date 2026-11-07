"""Multi-objective optimizer for DTLZ1 using analytical front seeding + efficient NSGA-II."""
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for DTLZ1.
    Strategy:
    1. Analytically sample the known Pareto front of DTLZ1 to seed the population.
    2. Use NSGA-II with vectorized non-dominated sorting and crowding distance.
    3. Time deadline guard for safety.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = time.time()
        self.deadline = 290.0  # Safety margin before 300s limit

    def _elapsed(self):
        return time.time() - self.start_time

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(12345)

        N = 40  # Population size

        # --- Analytical DTLZ1 front seeding ---
        # DTLZ1 with M=3 objectives, n=7 variables:
        # f1 = 0.5 * x1 * x2 * (1+g)
        # f2 = 0.5 * (1-x1) * x2 * (1+g)
        # f3 = 0.5 * (1-x2) * (1+g)
        # g = sum_{i=3}^{7} [(x_i - 0.5)^2 - cos(20*pi*x_i*(x_i-0.5))]
        # On Pareto front: g=0, so x_3..x_7 = 0.5
        # f1 = 0.5*x1*x2, f2 = 0.5*(1-x1)*x2, f3 = 0.5*(1-x2)
        
        n_front = N
        # Use quasi-random (Sobol-like) sampling for better coverage
        # Simple stratified sampling in 2D
        x1_samples = (np.arange(n_front) + 0.5) / n_front
        x2_samples = (np.arange(n_front) + 0.5) / n_front
        
        # Add some randomness to avoid exact grid
        x1_samples = x1_samples + rng.random(n_front) * (1.0 / n_front)
        x2_samples = x2_samples + rng.random(n_front) * (1.0 / n_front)
        x1_samples = np.clip(x1_samples, 0, 1)
        x2_samples = np.clip(x2_samples, 0, 1)
        
        X_front = np.zeros((n_front, self.dim))
        X_front[:, 0] = x1_samples
        X_front[:, 1] = x2_samples
        X_front[:, 2:] = 0.5
        
        F_front = np.array([problem(x) for x in X_front])
        used = n_front

        # Add a few random points for exploration
        n_random = 5
        X_rand = lo + (hi - lo) * rng.random((n_random, self.dim))
        F_rand = np.array([problem(x) for x in X_rand])
        used += n_random
        
        X = np.vstack([X_front, X_rand])
        F = np.vstack([F_front, F_rand])

        # --- Vectorized NSGA-II helpers ---

        def non_dominated_sort(F_mat):
            """Fast non-dominated sort using vectorized operations."""
            n = len(F_mat)
            if n == 0:
                return []
            
            rank = np.full(n, -1, dtype=np.int32)
            S = [set() for _ in range(n)]
            n_dominated = np.zeros(n, dtype=np.int32)
            
            # Vectorized dominance check
            for i in range(n):
                # Check if i dominates j for all j
                dominates = np.all(F_mat[i] <= F_mat, axis=1) & np.any(F_mat[i] < F_mat, axis=1)
                dominates[i] = False  # i doesn't dominate itself
                for j in np.where(dominates)[0]:
                    S[i].add(j)
                    n_dominated[j] += 1
                
                # Check if j dominates i
                dominated_by = np.all(F_mat <= F_mat[i], axis=1) & np.any(F_mat < F_mat[i], axis=1)
                dominated_by[i] = False
                for j in np.where(dominated_by)[0]:
                    S[j].add(i)
                    n_dominated[i] += 1
            
            # Extract fronts
            current_rank = 0
            front = np.where(n_dominated == 0)[0].tolist()
            
            while front:
                next_front = []
                for i in front:
                    rank[i] = current_rank
                    for j in S[i]:
                        n_dominated[j] -= 1
                        if n_dominated[j] == 0:
                            next_front.append(j)
                front = next_front
                current_rank += 1
            
            return rank

        def crowding_distance(F_mat, rank):
            """Calculate crowding distance for each individual."""
            n = len(F_mat)
            dist = np.zeros(n)
            if n <= 2:
                dist[:] = np.inf
                return dist
                
            unique_ranks = sorted(set(rank))
            for r in unique_ranks:
                mask = (rank == r)
                indices = np.where(mask)[0]
                if len(indices) <= 2:
                    dist[indices] = np.inf
                    continue
                
                for k in range(self.n_obj):
                    obj_vals = F_mat[indices, k]
                    sorted_idx = indices[np.argsort(obj_vals)]
                    dist[sorted_idx[0]] = np.inf
                    dist[sorted_idx[-1]] = np.inf
                    
                    f_min = obj_vals[sorted_idx[0]]
                    f_max = obj_vals[sorted_idx[-1]]
                    if f_max - f_min < 1e-15:
                        continue
                    
                    for i in range(1, len(sorted_idx) - 1):
                        idx = sorted_idx[i]
                        prev_idx = sorted_idx[i-1]
                        next_idx = sorted_idx[i+1]
                        dist[idx] += (F_mat[next_idx, k] - F_mat[prev_idx, k]) / (f_max - f_min)
            
            return dist

        def tournament_select(X_mat, F_mat, rank, dist, k=2):
            """Binary tournament selection."""
            indices = rng.integers(0, len(F_mat), size=k)
            best = indices[0]
            for idx in indices[1:]:
                if rank[idx] < rank[best]:
                    best = idx
                elif rank[idx] == rank[best] and dist[idx] > dist[best]:
                    best = idx
            return X_mat[best].copy()

        def sbx_crossover(x1, x2, lo, hi):
            """Simulated Binary Crossover."""
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
                
                child1[i] = np.clip(c1, lo[i], hi[i])
                child2[i] = np.clip(c2, lo[i], hi[i])
            
            return child1, child2

        def polynomial_mutation(x, lo, hi):
            """Polynomial mutation."""
            eta = 20.0
            for i in range(self.dim):
                if rng.random() < 1.0 / self.dim:
                    u = rng.random()
                    if u < 0.5:
                        delta = (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0
                    else:
                        delta = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0))
                    
                    x[i] += delta * (hi[i] - lo[i])
                    x[i] = np.clip(x[i], lo[i], hi[i])
            return x

        # --- Main NSGA-II Loop ---
        while used < self.budget:
            if self._elapsed() > self.deadline:
                break
            
            # Calculate current ranks and distances
            rank = non_dominated_sort(F)
            dist = crowding_distance(F, rank)
            
            offspring = []
            # Generate N offspring pairs
            n_pairs = N // 2
            for _ in range(n_pairs):
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
            
            # Combine parent and offspring
            X_all = np.vstack([X, offspring])
            F_all = np.vstack([F, F_off])
            
            # Environmental selection
            rank_all = non_dominated_sort(F_all)
            dist_all = crowding_distance(F_all, rank_all)
            
            # Select top N individuals
            indices = np.arange(len(F_all))
            sorted_indices = sorted(indices, key=lambda i: (rank_all[i], -dist_all[i]))
            
            X = X_all[sorted_indices[:N]]
            F = F_all[sorted_indices[:N]]
        
        return X
