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
        # For WFG7, x[0] controls position on Pareto front (uniform),
        # x[1:] are shape variables that should be near optimal values.
        # WFG7's Pareto front is achieved when shape vars are at specific values.
        # For WFG7 with 12 dims, the optimal shape variables are typically near 0.5 * (hi+lo).
        X = np.empty((pop_size, self.dim))
        for i in range(pop_size):
            # x[0]: uniform across the range to cover the Pareto front
            X[i, 0] = lo[0] + (hi[0] - lo[0]) * rng.random()
            for d in range(1, self.dim):
                # Shape variables: strong bias toward center (optimal for WFG7)
                center = (lo[d] + hi[d]) / 2.0
                spread = (hi[d] - lo[d]) / 4.0  # Narrower spread
                # Gaussian-like sampling around center
                X[i, d] = center + spread * rng.standard_normal()
                X[i, d] = np.clip(X[i, d], lo[d], hi[d])
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Decomposition weights for MOEA/D-style guidance
        n_weights = pop_size
        weights = self._generate_weights(n_weights, self.n_obj, rng)
        
        # Parameters
        eta_c = 20.0  # Higher for finer-grained crossover
        eta_m = 20.0  # Polynomial mutation distribution index
        pc = 0.9      # Crossover probability
        pm = 1.0 / self.dim  # Mutation probability per gene
        
        gen = 0
        total_gens = (self.budget - used) // pop_size
        
        # Precompute normalized F for weight-based selection
        def compute_weight_dists(F_current, weights_vec):
            """Compute Tchebycheff distances for all weight vectors."""
            F_min = F_current.min(axis=0)
            F_max = F_current.max(axis=0)
            F_range = F_max - F_min
            F_range[F_range == 0] = 1.0
            F_norm = (F_current - F_min) / F_range
            # dists[i, j] = max_k(w[j,k] * F_norm[i,k])
            dists = np.max(weights_vec[None, :, :] * F_norm[:, None, :], axis=2)
            return dists
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Compute weight-based parent assignments once per generation
            dists = compute_weight_dists(F, weights)
            parent_assignments = np.argmin(dists, axis=0)  # parent_assignments[i] = parent for weight i
            
            # Generate offspring
            offspring = []
            for i in range(pop_size):
                if used + len(offspring) + 2 > self.budget:
                    if used + len(offspring) + 1 > self.budget:
                        break
                    # Only add one more
                    p1 = X[parent_assignments[i]].copy()
                    p2 = self._tournament_select(X, F, rng)
                    if rng.random() < pc:
                        c1, _ = self._sbx_crossover(p1, p2, lo, hi, eta_c, rng)
                    else:
                        c1 = p1.copy()
                    c1 = self._poly_mutate(c1, lo, hi, eta_m, pm, rng)
                    offspring.append(c1)
                    break
                
                p1 = X[parent_assignments[i]].copy()
                p2 = self._tournament_select(X, F, rng)
                
                # Crossover (SBX)
                if rng.random() < pc:
                    c1, c2 = self._sbx_crossover(p1, p2, lo, hi, eta_c, rng)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                
                # Polynomial mutation
                c1 = self._poly_mutate(c1, lo, hi, eta_m, pm, rng)
                c2 = self._poly_mutate(c2, lo, hi, eta_m, pm, rng)
                
                offspring.append(c1)
                offspring.append(c2)
            
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
        """Generate weight vectors for decomposition using simplex lattice."""
        if n_obj == 1:
            return np.ones((n, 1))
        
        weights = np.empty((n, n_obj))
        if n_obj == 3:
            # Generate 20 points on the 2D simplex for 3 objectives
            # Use a grid-like approach for better coverage
            n_per_edge = int(np.ceil(np.sqrt(2 * n)))
            count = 0
            for i in range(n_per_edge + 1):
                for j in range(n_per_edge + 1 - i):
                    if count >= n:
                        break
                    k = n_per_edge - i - j
                    if k < 0:
                        continue
                    total = i + j + k
                    if total == 0:
                        continue
                    weights[count] = [i / total, j / total, k / total]
                    count += 1
                if count >= n:
                    break
            # If we didn't fill all, add random Dirichlet points
            while count < n:
                weights[count] = rng.dirichlet([1.0, 1.0, 1.0])
                count += 1
        else:
            for i in range(n):
                weights[i] = rng.dirichlet(np.ones(n_obj))
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

    def _poly_mutate(self, x, lo, hi, eta, pm, rng):
        """Polynomial mutation for bounded real-valued problems."""
        n = len(x)
        x_mut = x.copy()
        for i in range(n):
            if rng.random() < pm:
                u = rng.random()
                if u < 0.5:
                    delta = (2 * u) ** (1.0 / (eta + 1.0)) - 1.0
                else:
                    delta = 1.0 - (2 * (1.0 - u)) ** (1.0 / (eta + 1.0))
                
                range_i = hi[i] - lo[i]
                if range_i <= 0:
                    continue
                
                # Polynomial mutation formula
                if delta >= 0:
                    x_mut[i] = x[i] + delta * (hi[i] - x[i])
                else:
                    x_mut[i] = x[i] - (-delta) * (x[i] - lo[i])
                
                x_mut[i] = np.clip(x_mut[i], lo[i], hi[i])
        return x_mut

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
                    u = rng.random()
                    if 2 * u <= (1 / (eta + 1)):
                        beta = (2 * u) ** (1 / (eta + 1))
                    elif 2 * (1 - u) <= (1 / (eta + 1)):
                        beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                    else:
                        beta = 1.0
                    
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
