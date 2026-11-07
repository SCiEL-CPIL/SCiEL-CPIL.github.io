import numpy as np


class Optimizer:
    """NSGA-II multi-objective optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        pop_size = 50
        used = 0
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used += pop_size

        if used >= self.budget:
            return F

        # Helper functions for NSGA-II
        def non_dominated_sort(F):
            """
            Perform non-dominated sorting.
            Returns:
                ranks: list of lists of indices, ranks[i] contains indices of the i-th front
            """
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] = set of indices dominated by i
            n_doms = [0] * n           # n_doms[i] = number of individuals dominating i
            rank = [0] * n             # rank[i] = front index of i
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_doms[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_doms[i] += 1

            current_front = [i for i in range(n) if n_doms[i] == 0]
            for i in current_front:
                rank[i] = 0

            k = 0
            while current_front:
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_doms[j] -= 1
                        if n_doms[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                k += 1
                current_front = next_front

            # Group indices by rank
            max_rank = max(rank) if rank else 0
            fronts = [[] for _ in range(max_rank + 1)]
            for i in range(n):
                fronts[rank[i]].append(i)
            
            return fronts

        def crowding_distance(F, front_indices):
            """
            Calculate crowding distance for individuals in a front.
            Returns dict {index: distance}
            """
            if len(front_indices) <= 2:
                return {i: float('inf') for i in front_indices}
            
            m = self.n_obj
            n = len(front_indices)
            dist = {i: 0.0 for i in front_indices}
            
            F_front = F[front_indices]
            
            for k in range(m):
                sorted_indices = np.argsort(F_front[:, k])
                dist[front_indices[sorted_indices[0]]] = float('inf')
                dist[front_indices[sorted_indices[-1]]] = float('inf')
                
                f_min = F_front[sorted_indices[0], k]
                f_max = F_front[sorted_indices[-1], k]
                f_range = f_max - f_min
                
                if f_range > 1e-10:
                    for i in range(1, n - 1):
                        idx = front_indices[sorted_indices[i]]
                        dist[idx] += (F_front[sorted_indices[i+1], k] - F_front[sorted_indices[i-1], k]) / f_range
            
            return dist

        def sbx_crossover(x1, x2, eta=20.0):
            """Simulated Binary Crossover"""
            if rng.random() > 0.5:
                x1, x2 = x2, x1
            
            x1_new = np.zeros_like(x1)
            x2_new = np.zeros_like(x2)
            
            for i in range(self.dim):
                if rng.random() < 0.5:
                    if abs(x1[i] - x2[i]) < 1e-14:
                        x1_new[i] = x1[i]
                        x2_new[i] = x2[i]
                    else:
                        u = rng.random()
                        if u <= 0.5:
                            beta = (2 * u) ** (1 / (eta + 1))
                        else:
                            beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                        
                        x1_new[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                        x2_new[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
                else:
                    x1_new[i] = x1[i]
                    x2_new[i] = x2[i]
            
            return x1_new, x2_new

        def polynomial_mutation(x, eta=20.0):
            """Polynomial Mutation"""
            x_new = x.copy()
            for i in range(self.dim):
                if rng.random() < 1.0 / self.dim:
                    u = rng.random()
                    if u < 0.5:
                        delta = (2 * u) ** (1 / (eta + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                    
                    x_new[i] += delta * (hi[i] - lo[i])
                    x_new[i] = np.clip(x_new[i], lo[i], hi[i])
            return x_new

        # Main loop
        while used < self.budget:
            # Create offspring
            offspring_X = np.zeros((pop_size, self.dim))
            
            for i in range(0, pop_size, 2):
                if used >= self.budget:
                    break
                
                # Tournament selection
                # Pick 2 parents from current population
                idx1 = rng.integers(0, pop_size)
                idx2 = rng.integers(0, pop_size)
                
                # We need a selection mechanism. 
                # Since we update X and F in place, we use the current population.
                # However, proper NSGA-II selects from the combined parent+offspring pool for the *next* generation.
                # Here, we generate offspring from the current population.
                
                x1 = X[idx1]
                x2 = X[idx2]
                
                # Crossover
                c1, c2 = sbx_crossover(x1, x2)
                
                # Mutation
                c1 = polynomial_mutation(c1)
                c2 = polynomial_mutation(c2)
                
                offspring_X[i] = c1
                if i + 1 < pop_size:
                    offspring_X[i + 1] = c2
            
            # Evaluate offspring
            offspring_F = np.array([problem(x) for x in offspring_X])
            used += pop_size
            
            # Combine parents and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            
            # Non-dominated sorting
            fronts = non_dominated_sort(combined_F)
            
            # Select next population
            next_X = np.zeros((pop_size, self.dim))
            next_F = np.zeros((pop_size, self.n_obj))
            
            count = 0
            for front in fronts:
                if count + len(front) <= pop_size:
                    # Take all from this front
                    for idx in front:
                        next_X[count] = combined_X[idx]
                        next_F[count] = combined_F[idx]
                        count += 1
                else:
                    # Need to fill the rest with this front, sorted by crowding distance
                    cds = crowding_distance(combined_F, front)
                    sorted_front = sorted(front, key=lambda i: cds[i], reverse=True)
                    remaining = pop_size - count
                    for idx in sorted_front[:remaining]:
                        next_X[count] = combined_X[idx]
                        next_F[count] = combined_F[idx]
                        count += 1
                    break
            
            X = next_X
            F = next_F

        return F
