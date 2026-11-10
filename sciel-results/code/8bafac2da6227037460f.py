import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        N = 20
        if self.budget < N:
            X = lo + (hi - lo) * np.random.random((self.budget, self.dim))
            F = np.array([problem(x) for x in X])
            return X

        G = (self.budget - N) // N
        
        # Initialize population
        X = lo + (hi - lo) * np.random.random((N, self.dim))
        F = np.array([problem(x) for x in X])
        used = N
        
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dominates = [0] * n
            rank = [0] * n
            fronts = []
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                    elif dominates(F[j], F[i]):
                        n_dominates[i] += 1
            
            current_front = []
            for i in range(n):
                if n_dominates[i] == 0:
                    rank[i] = 0
                    current_front.append(i)
            
            k = 0
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dominates[j] -= 1
                        if n_dominates[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                current_front = next_front
                k += 1
            
            return fronts, rank

        def crowding_distance(F, front_indices):
            if len(front_indices) <= 2:
                return [float('inf')] * len(front_indices)
            
            distances = [0.0] * len(front_indices)
            F_front = F[front_indices]
            
            for m in range(self.n_obj):
                sorted_indices = np.argsort(F_front[:, m])
                distances[sorted_indices[0]] = float('inf')
                distances[sorted_indices[-1]] = float('inf')
                
                f_min = F_front[sorted_indices[0], m]
                f_max = F_front[sorted_indices[-1], m]
                range_f = f_max - f_min
                if range_f == 0:
                    continue
                    
                for k in range(1, len(front_indices) - 1):
                    i = sorted_indices[k]
                    i_prev = sorted_indices[k-1]
                    i_next = sorted_indices[k+1]
                    distances[i] += (F_front[i_next, m] - F_front[i_prev, m]) / range_f
            
            return distances

        def binary_tournament(indices, ranks, distances):
            i1 = np.random.randint(len(indices))
            i2 = np.random.randint(len(indices))
            idx1 = indices[i1]
            idx2 = indices[i2]
            
            if ranks[idx1] < ranks[idx2]:
                return idx1
            elif ranks[idx2] < ranks[idx1]:
                return idx2
            else:
                if distances[idx1] > distances[idx2]:
                    return idx1
                else:
                    return idx2

        def sbx_crossover(x1, x2, eta=15.0):
            child1 = x1.copy()
            child2 = x2.copy()
            
            for i in range(self.dim):
                if np.random.rand() < 0.5:
                    if abs(x2[i] - x1[i]) > 1e-14:
                        if np.random.rand() <= 0.5:
                            beta = 1.0 + (2.0 * np.random.rand() * (x1[i] - x2[i]) / (hi[i] - lo[i]))
                            alpha = (beta + 1.0) ** (eta + 1.0)
                            beta1 = (2.0 * beta ** (eta + 1.0) + (2.0 - beta) ** (eta + 1.0)) ** (-1.0 / (eta + 1.0))
                            child1[i] = 0.5 * ((1.0 + beta1) * x1[i] + (1.0 - beta1) * x2[i])
                            beta = 1.0 + (2.0 * np.random.rand() * (x2[i] - x1[i]) / (hi[i] - lo[i]))
                            alpha = (beta + 1.0) ** (eta + 1.0)
                            beta2 = (2.0 * beta ** (eta + 1.0) + (2.0 - beta) ** (eta + 1.0)) ** (-1.0 / (eta + 1.0))
                            child2[i] = 0.5 * ((1.0 + beta2) * x2[i] + (1.0 - beta2) * x1[i])
                        else:
                            beta = 1.0 + (2.0 * np.random.rand() * (x1[i] - x2[i]) / (hi[i] - lo[i]))
                            alpha = (beta + 1.0) ** (eta + 1.0)
                            beta1 = (2.0 * beta ** (eta + 1.0) + (2.0 - beta) ** (eta + 1.0)) ** (-1.0 / (eta + 1.0))
                            child1[i] = 0.5 * ((1.0 + beta1) * x2[i] + (1.0 - beta1) * x1[i])
                            beta = 1.0 + (2.0 * np.random.rand() * (x2[i] - x1[i]) / (hi[i] - lo[i]))
                            alpha = (beta + 1.0) ** (eta + 1.0)
                            beta2 = (2.0 * beta ** (eta + 1.0) + (2.0 - beta) ** (eta + 1.0)) ** (-1.0 / (eta + 1.0))
                            child2[i] = 0.5 * ((1.0 + beta2) * x1[i] + (1.0 - beta2) * x2[i])
            return child1, child2

        def polynomial_mutation(x, eta=20.0):
            x_mut = x.copy()
            for i in range(self.dim):
                if np.random.rand() < (1.0 / self.dim):
                    delta1 = (1.0 + 2.0 * (x_mut[i] - lo[i]) / (hi[i] - lo[i])) ** (-(eta + 1.0))
                    delta2 = (1.0 + 2.0 * (hi[i] - x_mut[i]) / (hi[i] - lo[i])) ** (-(eta + 1.0))
                    if np.random.rand() <= 0.5:
                        delta = delta1 - 1.0
                    else:
                        delta = 1.0 - delta2
                    x_mut[i] += delta * (hi[i] - lo[i])
                    x_mut[i] = np.clip(x_mut[i], lo[i], hi[i])
            return x_mut

        # Main Loop
        while used + N <= self.budget:
            # Generate Offspring
            X_off = np.zeros((N, self.dim))
            F_off = np.zeros((N, self.n_obj))
            
            for k in range(N):
                # Selection
                indices = list(range(N))
                # We need ranks and distances for the current population X, F
                fronts, ranks = non_dominated_sort(F)
                # Calculate crowding distance for each front
                distances = [0.0] * N
                for front in fronts:
                    d = crowding_distance(F, front)
                    for idx, dist in zip(front, d):
                        distances[idx] = dist
                
                p1 = binary_tournament(indices, ranks, distances)
                p2 = binary_tournament(indices, ranks, distances)
                
                # Crossover
                if np.random.rand() < 0.9:
                    child1, child2 = sbx_crossover(X[p1], X[p2])
                else:
                    child1 = X[p1].copy()
                    child2 = X[p2].copy()
                
                # Mutation
                child1 = polynomial_mutation(child1)
                child2 = polynomial_mutation(child2)
                
                # Clip
                child1 = np.clip(child1, lo, hi)
                child2 = np.clip(child2, lo, hi)
                
                # Evaluate
                F_off[k*2] = problem(child1)
                F_off[k*2+1] = problem(child2)
                X_off[k*2] = child1
                X_off[k*2+1] = child2
                
                used += 2
                if used >= self.budget:
                    break
            
            # Note: The loop above generates 2 offspring per iteration, so N/2 iterations.
            # But we need exactly N offspring per generation to maintain population size N.
            # So we should generate N offspring.
            # Let's restructure the offspring generation to generate exactly N individuals.
            
            # Restarting offspring generation logic to be cleaner
            X_off = np.zeros((N, self.dim))
            F_off = np.zeros((N, self.n_obj))
            
            # Re-calculate ranks/distances once per generation
            fronts, ranks = non_dominated_sort(F)
            distances = [0.0] * N
            for front in fronts:
                d = crowding_distance(F, front)
                for idx, dist in zip(front, d):
                    distances[idx] = dist

            for k in range(N):
                indices = list(range(N))
                p1 = binary_tournament(indices, ranks, distances)
                p2 = binary_tournament(indices, ranks, distances)
                
                if np.random.rand() < 0.9:
                    child1, _ = sbx_crossover(X[p1], X[p2])
                else:
                    child1 = X[p1].copy()
                
                child1 = polynomial_mutation(child1)
                child1 = np.clip(child1, lo, hi)
                
                F_off[k] = problem(child1)
                X_off[k] = child1
                used += 1
                
                if used >= self.budget:
                    break
            
            if used >= self.budget:
                break

            # Combine
            X_comb = np.vstack((X, X_off))
            F_comb = np.vstack((F, F_off))
            
            # Sort and Select
            fronts_comb, ranks_comb = non_dominated_sort(F_comb)
            
            X_new = np.zeros((N, self.dim))
            F_new = np.zeros((N, self.n_obj))
            
            count = 0
            for front in fronts_comb:
                if count + len(front) <= N:
                    for idx in front:
                        X_new[count] = X_comb[idx]
                        F_new[count] = F_comb[idx]
                        count += 1
                else:
                    # Need crowding distance for this front
                    d = crowding_distance(F_comb, front)
                    # Sort by distance descending
                    sorted_front = sorted(front, key=lambda i: d[i], reverse=True)
                    for idx in sorted_front:
                        if count < N:
                            X_new[count] = X_comb[idx]
                            F_new[count] = F_comb[idx]
                            count += 1
                        else:
                            break
                
                if count >= N:
                    break
            
            X = X_new
            F = F_new

        return X
