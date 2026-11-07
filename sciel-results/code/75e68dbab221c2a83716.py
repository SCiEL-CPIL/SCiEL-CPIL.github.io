"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-objective optimizer using NSGA-II style approach for ZDT4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_vars = self.dim
        n_obj = self.n_obj

        # Population size
        pop_size = min(100, max(20, self.budget // 4))
        remaining_budget = self.budget

        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, n_vars))
        F = np.array([problem(x) for x in X])
        remaining_budget -= pop_size

        def fast_non_dominated_sort(F):
            """Fast non-dominated sort."""
            n = len(F)
            S = [[] for _ in range(n)]
            n_count = [0] * n
            front = [[]]
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_count[i] += 1
                
                if n_count[i] == 0:
                    front[0].append(i)
            
            k = 0
            while front[k]:
                next_front = []
                for i in front[k]:
                    for j in S[i]:
                        n_count[j] -= 1
                        if n_count[j] == 0:
                            next_front.append(j)
                k += 1
                if next_front:
                    front.append(next_front)
            
            return front

        def crowding_distance(F, front_indices):
            """Calculate crowding distance for individuals in a front."""
            n = len(front_indices)
            if n <= 2:
                return [float('inf')] * n
            
            dist = [0.0] * n
            for m in range(n_obj):
                obj_vals = np.array([F[i][m] for i in front_indices])
                sorted_idx = np.argsort(obj_vals)
                dist[sorted_idx[0]] = float('inf')
                dist[sorted_idx[-1]] = float('inf')
                
                obj_range = obj_vals[sorted_idx[-1]] - obj_vals[sorted_idx[0]]
                if obj_range > 0:
                    for k in range(1, n - 1):
                        idx = sorted_idx[k]
                        dist[idx] += (obj_vals[sorted_idx[k+1]] - obj_vals[sorted_idx[k-1]]) / obj_range
            
            return dist

        def select_parents(F, n_select):
            """Select parents using non-dominated sorting and crowding distance."""
            fronts = fast_non_dominated_sort(F)
            selected = []
            
            for front in fronts:
                if len(selected) + len(front) <= n_select:
                    selected.extend(front)
                else:
                    need = n_select - len(selected)
                    dist = crowding_distance(F, front)
                    sorted_by_dist = sorted(range(len(front)), key=lambda i: dist[i], reverse=True)
                    selected.extend(front[i] for i in sorted_by_dist[:need])
                    break
            
            return selected

        def tournament_select(F, k=2):
            """Tournament selection based on non-domination rank and crowding distance."""
            fronts = fast_non_dominated_sort(F)
            ranks = np.zeros(len(F), dtype=int)
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank
            
            # Calculate crowding distances for all fronts
            crowding = np.zeros(len(F))
            for front in fronts:
                if len(front) > 0:
                    dist = crowding_distance(F, front)
                    for i, idx in enumerate(front):
                        crowding[idx] = dist[i]
            
            # Tournament selection
            candidates = rng.integers(0, len(F), size=k)
            best = candidates[0]
            for c in candidates[1:]:
                if ranks[c] < ranks[best] or (ranks[c] == ranks[best] and crowding[c] > crowding[best]):
                    best = c
            return best

        def crossover(parent1, parent2):
            """Simulated binary crossover (SBX)."""
            eta = 20.0
            child1 = np.copy(parent1)
            child2 = np.copy(parent2)
            
            for i in range(n_vars):
                if rng.random() < 0.5:
                    if parent1[i] == parent2[i]:
                        continue
                    if parent1[i] < parent2[i]:
                        x1, x2 = parent1[i], parent2[i]
                    else:
                        x1, x2 = parent2[i], parent1[i]
                    
                    if rng.random() <= 1.0 / (eta + 1):
                        beta = (2 * rng.random()) ** (1.0 / (eta + 1))
                    else:
                        beta = (1.0 / (2 * (1.0 - rng.random()))) ** (1.0 / (eta + 1))
                    
                    c1 = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
                    c2 = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)
                    
                    if parent1[i] < parent2[i]:
                        child1[i], child2[i] = c1, c2
                    else:
                        child1[i], child2[i] = c2, c1
            
            return child1, child2

        def mutate(individual):
            """Polynomial mutation."""
            eta = 20.0
            child = np.copy(individual)
            for i in range(n_vars):
                if rng.random() < 1.0 / n_vars:
                    if individual[i] == lo[i] or individual[i] == hi[i]:
                        child[i] = lo[i] + (hi[i] - lo[i]) * rng.random()
                    else:
                        delta1 = (individual[i] - lo[i]) / (hi[i] - lo[i])
                        delta2 = (hi[i] - individual[i]) / (hi[i] - lo[i])
                        
                        if rng.random() <= delta2 / (delta1 + delta2):
                            xk = delta1
                            xy = (2 * rng.random() + (1 - 2 * xk)) ** (1.0 / (eta + 1))
                        else:
                            xk = delta2
                            xy = (2 * (1.0 - rng.random()) + (2 * xk - 1)) ** (1.0 / (eta + 1))
                        
                        child[i] = individual[i] + (xy - 1.0) * (hi[i] - lo[i])
                        child[i] = np.clip(child[i], lo[i], hi[i])
            
            return child

        # Main optimization loop
        generation = 0
        while remaining_budget >= 2 * pop_size:
            # Create offspring population
            offspring_X = []
            offspring_F = []
            
            for _ in range(pop_size):
                if remaining_budget < 2:
                    break
                    
                # Select two parents
                p1_idx = tournament_select(F)
                p2_idx = tournament_select(F)
                
                # Crossover
                child1, child2 = crossover(X[p1_idx], X[p2_idx])
                
                # Mutation
                child1 = mutate(child1)
                child2 = mutate(child2)
                
                # Evaluate
                f1 = problem(child1)
                f2 = problem(child2)
                
                offspring_X.append(child1)
                offspring_X.append(child2)
                offspring_F.append(f1)
                offspring_F.append(f2)
                
                remaining_budget -= 2
            
            if len(offspring_X) == 0:
                break
            
            # Combine parent and offspring populations
            combined_X = np.vstack([X, np.array(offspring_X)])
            combined_F = np.vstack([F, np.array(offspring_F)])
            
            # Environmental selection: select pop_size best individuals
            fronts = fast_non_dominated_sort(combined_F)
            selected_indices = []
            
            for front in fronts:
                if len(selected_indices) + len(front) <= pop_size:
                    selected_indices.extend(front)
                else:
                    need = pop_size - len(selected_indices)
                    dist = crowding_distance(combined_F, front)
                    sorted_by_dist = sorted(range(len(front)), key=lambda i: dist[i], reverse=True)
                    selected_indices.extend(front[i] for i in sorted_by_dist[:need])
                    break
            
            X = combined_X[selected_indices]
            F = combined_F[selected_indices]
            
            generation += 1
        
        # Use remaining budget for local improvements
        while remaining_budget > 0:
            # Find the best point (lowest sum of objectives)
            best_idx = int(np.argmin(np.sum(F, axis=1)))
            
            # Try small perturbations around the best point
            improved = False
            for _ in range(min(remaining_budget, 10)):
                if remaining_budget <= 0:
                    break
                
                perturbation = 0.01 * (hi - lo) * rng.standard_normal(n_vars)
                new_x = np.clip(X[best_idx] + perturbation, lo, hi)
                new_f = problem(new_x)
                remaining_budget -= 1
                
                # Check if new point is better or non-dominated
                if np.any(np.all(new_f < F, axis=1)):
                    X[best_idx] = new_x
                    F[best_idx] = new_f
                    improved = True
                elif not np.any(np.all(F <= new_f, axis=1) & np.any(F < new_f, axis=1)):
                    # Replace worst point
                    worst_idx = int(np.argmax(np.sum(F, axis=1)))
                    X[worst_idx] = new_x
                    F[worst_idx] = new_f
                    improved = True
            
            if not improved:
                break
        
        return X
# EVOLVE-BLOCK-END
