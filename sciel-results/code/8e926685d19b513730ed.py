import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _fast_non_dominated_sort(self, F):
        """
        Perform fast non-dominated sorting.
        Returns a list of lists of indices, where each inner list is a non-dominated front.
        """
        n = len(F)
        if n == 0:
            return []
        
        S = [[] for _ in range(n)]  # Dominance sets
        n_dominating = np.zeros(n, dtype=int)
        fronts = []
        
        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                    S[p].append(q)
                elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                    n_dominating[p] += 1
            if n_dominating[p] == 0:
                if not fronts:
                    fronts.append([])
                fronts[0].append(p)
        
        i = 0
        while True:
            if i >= len(fronts) or len(fronts[i]) == 0:
                break
            Q = []
            for p in fronts[i]:
                for q in S[p]:
                    n_dominating[q] -= 1
                    if n_dominating[q] == 0:
                        Q.append(q)
            i += 1
            if Q:
                fronts.append(Q)
            else:
                break
        return fronts

    def _crowding_distance(self, F, indices):
        """
        Calculate crowding distance for points in a front.
        """
        if len(indices) <= 2:
            return np.full(len(indices), float('inf'))
        
        distances = np.zeros(len(indices))
        m = self.n_obj
        for k in range(m):
            Fk = F[indices, k]
            min_k = np.min(Fk)
            max_k = np.max(Fk)
            if max_k == min_k:
                continue
            # Sort indices by objective k
            sorted_indices = indices[np.argsort(Fk)]
            # Set boundary points to inf
            distances[np.where(indices == sorted_indices[0])[0][0]] = float('inf')
            distances[np.where(indices == sorted_indices[-1])[0][0]] = float('inf')
            
            for i in range(1, len(sorted_indices) - 1):
                # Find the position of this point in the original indices array
                idx_in_front = np.where(indices == sorted_indices[i])[0][0]
                prev_val = F[sorted_indices[i-1], k]
                next_val = F[sorted_indices[i+1], k]
                distances[idx_in_front] += (next_val - prev_val) / (max_k - min_k)
        return distances

    def _sbx_crossover(self, x1, x2, lo, hi):
        """Simulated Binary Crossover"""
        if np.random.random() < 0.9: # Crossover probability
            xc = np.zeros_like(x1)
            for i in range(len(x1)):
                if np.random.random() < 0.5:
                    beta = 1 + 2 * (x1[i] - lo[i]) / (hi[i] - lo[i]) if x1[i] > x2[i] else 1 + 2 * (hi[i] - x2[i]) / (hi[i] - lo[i])
                    alpha = 2 - beta ** (-1)
                    if np.random.random() <= 1 / (alpha + 1):
                        betaq = beta ** (-1 / (20 + 1)) # eta_c = 20
                    else:
                        betaq = (1 / (2 - beta * (2 * np.random.random() - 1))) ** (1 / (20 + 1))
                    xc[i] = 0.5 * ((x1[i] + x2[i]) - betaq * (x1[i] - x2[i]))
                else:
                    beta = 1 + 2 * (x2[i] - lo[i]) / (hi[i] - lo[i]) if x2[i] > x1[i] else 1 + 2 * (hi[i] - x1[i]) / (hi[i] - lo[i])
                    alpha = 2 - beta ** (-1)
                    if np.random.random() <= 1 / (alpha + 1):
                        betaq = beta ** (-1 / (20 + 1))
                    else:
                        betaq = (1 / (2 - beta * (2 * np.random.random() - 1))) ** (1 / (20 + 1))
                    xc[i] = 0.5 * ((x1[i] + x2[i]) + betaq * (x1[i] - x2[i]))
            return np.clip(xc, lo, hi)
        return x1

    def _polynomial_mutation(self, x, lo, hi):
        """Polynomial Mutation"""
        xm = x.copy()
        for i in range(len(xm)):
            if np.random.random() < 1.0 / self.dim: # p_m = 1/dim
                delta1 = (xm[i] - lo[i]) / (hi[i] - lo[i])
                delta2 = (hi[i] - xm[i]) / (hi[i] - lo[i])
                rand = np.random.random()
                if rand < 0.5:
                    xq = 2 * rand + (1 - 2 * rand) * (1 + delta1) ** (20 + 1) # eta_m = 20
                    deltaq = (xq ** (1 / (20 + 1)) - 1)
                else:
                    xq = 2 * (1 - rand) + (2 * rand - 1) * (1 + delta2) ** (20 + 1)
                    deltaq = 1 - (xq ** (1 / (20 + 1)))
                xm[i] = xm[i] + deltaq * (hi[i] - lo[i])
        return np.clip(xm, lo, hi)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0 # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        # Population size
        pop_size = 20
        
        # Phase 1: Initialize population
        X = np.random.uniform(lo, hi, size=(pop_size, dim))
        F = np.array([problem(x) for x in X])
        evals = pop_size
        
        # Phase 2: Evolutionary Loop
        while evals < budget:
            if time.time() > deadline:
                break
            
            # Determine how many offspring to generate
            n_offspring = min(pop_size, budget - evals)
            
            # Generate offspring
            X_off = np.zeros((n_offspring, dim))
            for i in range(n_offspring):
                # Tournament Selection
                p1 = self._tournament_selection(F, X)
                p2 = self._tournament_selection(F, X)
                
                # Crossover
                child = self._sbx_crossover(p1, p2, lo, hi)
                
                # Mutation
                child = self._polynomial_mutation(child, lo, hi)
                
                X_off[i] = child
            
            # Evaluate offspring
            F_off = np.array([problem(x) for x in X_off])
            evals += n_offspring
            
            # Combine parents and offspring
            X_combined = np.vstack([X, X_off])
            F_combined = np.vstack([F, F_off])
            
            # Selection: Keep best pop_size individuals
            # Fast Non-Dominated Sort
            fronts = self._fast_non_dominated_sort(F_combined)
            
            selected_indices = []
            for front in fronts:
                if len(selected_indices) + len(front) <= pop_size:
                    selected_indices.extend(front)
                else:
                    # Need to select from this front based on crowding distance
                    remaining = pop_size - len(selected_indices)
                    distances = self._crowding_distance(F_combined, front)
                    # Sort by distance descending
                    sorted_front = front[np.argsort(distances)[::-1]]
                    selected_indices.extend(sorted_front[:remaining])
                    break
            
            # Update population
            X = X_combined[selected_indices]
            F = F_combined[selected_indices]
            
        return X

    def _tournament_selection(self, F, X):
        """Tournament selection with size 2"""
        n = len(F)
        i1 = np.random.randint(0, n)
        i2 = np.random.randint(0, n)
        
        # Check dominance
        if np.all(F[i1] <= F[i2]) and np.any(F[i1] < F[i2]):
            return X[i1]
        elif np.all(F[i2] <= F[i1]) and np.any(F[i2] < F[i1]):
            return X[i2]
        else:
            # If non-dominated, pick randomly (or use crowding, but simple random is fine for small pop)
            return X[np.random.choice([i1, i2])]

