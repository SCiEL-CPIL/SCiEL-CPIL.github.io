import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 295.0  # Safety margin before 330s limit
        
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        
        # Population size: 20 allows 10 generations with 200 budget
        pop_size = 20
        
        # Initialize with WFG7-aware sampling:
        # WFG7's Pareto front is near specific regions. 
        # For WFG7, the first variable (position) should be uniform,
        # while shape variables tend toward mid-range values for optimal front.
        X = np.empty((pop_size, self.dim))
        for i in range(pop_size):
            for d in range(self.dim):
                if d == 0:
                    # Position variable: uniform
                    X[i, d] = lo[d] + (hi[d] - lo[d]) * rng.random()
                else:
                    # Shape variables: bias toward center with some spread
                    center = (lo[d] + hi[d]) / 2.0
                    spread = (hi[d] - lo[d]) / 2.0
                    # Sample from a distribution centered at midpoint
                    u = rng.random()
                    # Use a triangular-like distribution peaking at center
                    if u < 0.5:
                        X[i, d] = lo[d] + spread * (2 * u) ** 0.7
                    else:
                        X[i, d] = hi[d] - spread * (2 * (1 - u)) ** 0.7
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Decomposition weights for MOEA/D-style guidance
        # Generate weight vectors for 3 objectives
        n_weights = pop_size
        weights = self._generate_weights(n_weights, self.n_obj, rng)
        
        # Parameters
        eta_c = 15.0
        pc = 0.9
        base_sigma = 0.15  # Initial mutation sigma
        
        gen = 0
        total_gens = (self.budget - used) // pop_size
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Adaptive mutation strength: decrease over generations
            progress = gen / max(total_gens, 1)
            sigma_scale = base_sigma * (1.0 - 0.6 * progress)  # Reduce by 60% over run
            sigma = sigma_scale * (hi - lo)
            
            # Generate offspring
            offspring = []
            for i in range(pop_size):
                # Weight-based selection: pick parent closest to weight vector i
                w = weights[i]
                # Normalize F for weight computation
                F_min = F.min(axis=0)
                F_max = F.max(axis=0)
                F_range = F_max - F_min
                F_range[F_range == 0] = 1.0
                F_norm = (F - F_min) / F_range
                
                # Compute Tchebycheff distance to weight
                dists = np.max(w * F_norm, axis=1)
                parent_idx = np.argmin(dists)
                p1 = X[parent_idx].copy()
                
                # Second parent: tournament selection
                p2 = self._tournament_select(X, F, rng)
                
                # Crossover (SBX)
                if rng.random() < pc:
                    c1, c2 = self._sbx_crossover(p1, p2, lo, hi, eta_c, rng)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                
                # Adaptive Gaussian mutation
                c1 = c1 + sigma * rng.standard_normal(self.dim)
                c2 = c2 + sigma * rng.standard_normal(self.dim)
                
                c1 = np.clip(c1, lo, hi)
                c2 = np.clip(c2, lo, hi)
                
                # Use both offspring if budget allows, otherwise just c1
                if used + len(offspring) + 2 <= self.budget:
                    offspring.append(c1)
                    offspring.append(c2)
                else:
                    offspring.append(c1)
                
                if used + len(offspring) >= self.budget:
                    break
            
            if not offspring:
                break
            
            Q = np.array(offspring)
            F_q = np.array([problem(x) for x in Q])
            used += len(Q)
            
            # Combine parent and offspring
            P = np.vstack([X, Q])
            F_p = np.vstack([F, F_q])
            
            # Select next generation using NSGA-II
            X, F = self._nsga2_select(P, F_p, pop_size, rng)
            
            gen += 1
            
        return X

    def _generate_weights(self, n, n_obj, rng):
        """Generate weight vectors for decomposition."""
        if n_obj == 1:
            return np.ones((n, 1))
        
        weights = np.empty((n, n_obj))
        # Use simplex lattice points
        if n_obj == 3:
            # Generate points on the simplex
            for i in range(n):
                # Use Dirichlet distribution for diverse weights
                w = rng.dirichlet([1.0, 1.0, 1.0])
                weights[i] = w
        else:
            for i in range(n):
                w = rng.dirichlet(np.ones(n_obj))
                weights[i] = w
        return weights

    def _tournament_select(self, X, F, rng, k=3):
        idxs = rng.integers(0, len(F), size=k)
        best_idx = idxs[0]
        for idx in idxs[1:]:
            if self._dominates(F[idx], F[best_idx]):
                best_idx = idx
        return X[best_idx].copy()

    def _dominates(self, a, b):
        return np.all(a <= b) and np.any(a < b)

    def _nsga2_select(self, P, F_p, target_size, rng):
        ranks, fronts = self._non_dominated_sort(F_p)
        
        next_X = []
        next_F = []
        
        for front in fronts:
            if len(next_X) + len(front) <= target_size:
                next_X.extend(P[front])
                next_F.extend(F_p[front])
            else:
                front_indices = np.array(front)
                F_front = F_p[front_indices]
                cd = self._crowding_distance(F_front)
                sorted_indices = np.argsort(-cd)
                
                needed = target_size - len(next_X)
                for idx in sorted_indices[:needed]:
                    global_idx = front_indices[idx]
                    next_X.append(P[global_idx])
                    next_F.append(F_p[global_idx])
                break
        
        return np.array(next_X), np.array(next_F)

    def _non_dominated_sort(self, F):
        n = len(F)
        S = [[] for _ in range(n)]
        n_dominating = [0] * n
        fronts = []
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._dominates(F[i], F[j]):
                    S[i].append(j)
                elif self._dominates(F[j], F[i]):
                    n_dominating[i] += 1
        
        current_front = [i for i in range(n) if n_dominating[i] == 0]
        
        k = 0
        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in S[i]:
                    n_dominating[j] -= 1
                    if n_dominating[j] == 0:
                        next_front.append(j)
            current_front = next_front
            k += 1
        
        return k, fronts

    def _crowding_distance(self, F):
        n = len(F)
        if n <= 2:
            return np.full(n, np.inf)
            
        cd = np.zeros(n)
        m = F.shape[1]
        
        for k in range(m):
            # Sort by k-th objective
            sorted_idx = np.argsort(F[:, k])
            cd[sorted_idx[0]] = np.inf
            cd[sorted_idx[-1]] = np.inf
            
            f_range = F[sorted_idx[-1], k] - F[sorted_idx[0], k]
            if f_range == 0:
                continue
                
            for i in range(1, n - 1):
                cd[sorted_idx[i]] += (F[sorted_idx[i+1], k] - F[sorted_idx[i-1], k]) / f_range
                
        return cd

    def _sbx_crossover(self, p1, p2, lo, hi, eta, rng):
        c1 = np.empty_like(p1)
        c2 = np.empty_like(p2)
        
        for i in range(len(p1)):
            if rng.random() < 0.5:
                if abs(p1[i] - p2[i]) > 1e-14:
                    if 2 * rng.random() <= (1 / (eta + 1)):
                        beta = (2 * rng.random()) ** (1 / (eta + 1))
                    else:
                        beta = (1 / (2 * (1 - rng.random()))) ** (1 / (eta + 1))
                    
                    c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                    c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
                else:
                    c1[i] = p1[i]
                    c2[i] = p2[i]
            else:
                c1[i] = p2[i]
                c2[i] = p1[i]
                
        # Clip to bounds
        c1 = np.clip(c1, lo, hi)
        c2 = np.clip(c2, lo, hi)
        
        return c1, c2
