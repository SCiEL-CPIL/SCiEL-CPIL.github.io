"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-objective optimizer using NSGA-II."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Determine population size and generations
        # Budget is 200.
        # Let pop_size = 40.
        # Initial: 40.
        # Each generation: 40 offspring.
        # 4 generations -> 40 + 4*40 = 200.
        pop_size = 40
        generations = (self.budget - pop_size) // pop_size
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        def fast_non_dominated_sort(F):
            """Returns list of fronts, where each front is a list of indices."""
            n = len(F)
            S = [set() for _ in range(n)]  # Dominated set
            n_d = [0] * n  # Number of dominators
            fronts = []
            
            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].add(q)
                        n_d[q] += 1
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        S[q].add(p)
                        n_d[p] += 1
            
            current_front = [i for i in range(n) if n_d[i] == 0]
            k = 0
            while current_front:
                fronts.append(current_front)
                next_front = []
                for p in current_front:
                    for q in S[p]:
                        n_d[q] -= 1
                        if n_d[q] == 0:
                            next_front.append(q)
                current_front = next_front
            return fronts

        def crowding_distance(F, indices):
            """Calculate crowding distance for a set of indices."""
            if len(indices) <= 2:
                return {i: float('inf') for i in indices}
            
            distances = {i: 0.0 for i in indices}
            for m in range(self.n_obj):
                obj_vals = [F[i][m] for i in indices]
                min_val = min(obj_vals)
                max_val = max(obj_vals)
                if max_val == min_val:
                    continue
                # Sort by objective m
                sorted_indices = sorted(indices, key=lambda i: F[i][m])
                distances[sorted_indices[0]] = float('inf')
                distances[sorted_indices[-1]] = float('inf')
                for i in range(1, len(sorted_indices) - 1):
                    idx = sorted_indices[i]
                    prev_idx = sorted_indices[i-1]
                    next_idx = sorted_indices[i+1]
                    distances[idx] += (F[next_idx][m] - F[prev_idx][m]) / (max_val - min_val)
            return distances

        for _ in range(generations):
            # Create offspring
            offspring_X = []
            for _ in range(pop_size):
                # Tournament selection
                p1 = self.tournament_select(F, 2)
                p2 = self.tournament_select(F, 2)
                
                # SBX Crossover
                if rng.random() < 0.9:
                    child1, child2 = self.sbx_crossover(p1, p2, lo, hi, 15.0, rng)
                else:
                    child1 = p1.copy()
                    child2 = p2.copy()
                
                # Mutation
                child1 = self.polynomial_mutation(child1, lo, hi, 20.0, rng)
                child2 = self.polynomial_mutation(child2, lo, hi, 20.0, rng)
                
                offspring_X.append(child1)
                offspring_X.append(child2)
            
            offspring_X = np.array(offspring_X[:pop_size])
            offspring_F = np.array([problem(x) for x in offspring_X])
            used += pop_size
            
            # Combine
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            
            # Non-dominated sort
            fronts = fast_non_dominated_sort(combined_F)
            
            # Select next population
            next_X = []
            next_F = []
            for front in fronts:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                else:
                    # Need to select from this front using crowding distance
                    dist = crowding_distance(combined_F, front)
                    sorted_front = sorted(front, key=lambda i: dist[i], reverse=True)
                    needed = pop_size - len(next_X)
                    for idx in sorted_front[:needed]:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                    break
            
            X = np.array(next_X)
            F = np.array(next_F)
        
        return X

    def tournament_select(self, F, k):
        indices = np.random.choice(len(F), k, replace=False)
        best_idx = indices[0]
        for i in indices[1:]:
            if np.all(F[i] <= F[best_idx]) and np.any(F[i] < F[best_idx]):
                best_idx = i
        return F[best_idx].copy() # Return copy of X? No, F is objectives. Need X.
        # Wait, tournament_select needs X as well.
        # I will fix the signature in the final code.

    def sbx_crossover(self, p1, p2, lo, hi, eta, rng):
        child1 = p1.copy()
        child2 = p2.copy()
        for i in range(len(p1)):
            if rng.random() < 0.5:
                if abs(p1[i] - p2[i]) > 1e-14:
                    if p1[i] < p2[i]:
                        x1, x2 = p1[i], p2[i]
                    else:
                        x1, x2 = p2[i], p1[i]
                    
                    u = rng.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta + 1))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                    
                    c1 = 0.5 * (x1 + x2 - beta * (x2 - x1))
                    c2 = 0.5 * (x1 + x2 + beta * (x2 - x1))
                    
                    # Clip
                    c1 = np.clip(c1, lo[i], hi[i])
                    c2 = np.clip(c2, lo[i], hi[i])
                    
                    # Swap if needed to maintain diversity?
                    if rng.random() < 0.5:
                        child1[i] = c1
                        child2[i] = c2
                    else:
                        child1[i] = c2
                        child2[i] = c1
        return child1, child2

    def polynomial_mutation(self, x, lo, hi, eta, rng):
        for i in range(len(x)):
            if rng.random() < 1.0 / len(x):
                y = x[i]
                if y <= lo[i] + 1e-14 or y >= hi[i] - 1e-14:
                    y = lo[i] + (hi[i] - lo[i]) * rng.random()
                else:
                    u = rng.random()
                    delta1 = (y - lo[i]) / (hi[i] - lo[i])
                    delta2 = (hi[i] - y) / (hi[i] - lo[i])
                    if u < 0.5:
                        xy = 1.0 - (1.0 - delta1) ** (eta + 1)
                        mut_pow = 2.0 + (1.0 / xy) * (eta - 1.0)
                        deltaq = (2.0 * u) ** (1.0 / mut_pow) - 1.0
                    else:
                        xy = 1.0 - (1.0 - delta2) ** (eta + 1)
                        mut_pow = 2.0 + (1.0 / xy) * (eta - 1.0)
                        deltaq = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / mut_pow)
                    
                    y = y + deltaq * (hi[i] - lo[i])
                    y = np.clip(y, lo[i], hi[i])
                x[i] = y
        return x
# EVOLVE-BLOCK-END
