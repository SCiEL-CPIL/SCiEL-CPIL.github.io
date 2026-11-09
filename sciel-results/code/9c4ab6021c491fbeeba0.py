"""Rewritten program using NSGA-II for multi-objective optimization."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-objective optimizer using NSGA-II."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # NSGA-II Parameters
        pop_size = 20
        
        # Ensure we don't exceed budget
        if budget < pop_size:
            pop_size = budget
            
        # Initialize Population
        population_x = lo + rng.random((pop_size, dim)) * (hi - lo)
        population_f = np.array([problem(x) for x in population_x])
        used = pop_size

        # Helper functions for NSGA-II

        def fast_non_dominated_sort(F):
            """
            Perform fast non-dominated sorting.
            Returns list of fronts, each containing indices.
            """
            n = len(F)
            S = [[] for _ in range(n)]  # Dominated sets
            n_dominates = np.zeros(n, dtype=int)  # Number of solutions dominating i
            rank = np.zeros(n, dtype=int)
            
            fronts = []
            
            for p in range(n):
                for q in range(n):
                    if p == q: continue
                    # Check if p dominates q
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                    # Check if q dominates p
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n_dominates[p] += 1
                        
            # Front 1
            front = [i for i in range(n) if n_dominates[i] == 0]
            if front:
                fronts.append(front)
                for i in front:
                    rank[i] = 1

            k = 1
            while front:
                next_front = []
                for p in front:
                    for q in S[p]:
                        n_dominates[q] -= 1
                        if n_dominates[q] == 0:
                            rank[q] = k + 1
                            next_front.append(q)
                if next_front:
                    fronts.append(next_front)
                    k += 1
                    front = next_front
                else:
                    break
            
            return fronts

        def crowding_distance(F, indices):
            """
            Calculate crowding distance for a set of indices.
            """
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)
            
            crowd = np.zeros(n)
            n_obj = F.shape[1]
            
            for m in range(n_obj):
                # Sort indices by objective m
                sorted_indices = sorted(range(n), key=lambda i: F[indices[i], m])
                
                # Assign infinity to boundary points
                crowd[sorted_indices[0]] = np.inf
                crowd[sorted_indices[-1]] = np.inf
                
                # Normalize objective range
                f_min = F[indices[sorted_indices[0]], m]
                f_max = F[indices[sorted_indices[-1]], m]
                f_range = f_max - f_min
                
                if f_range == 0:
                    continue
                    
                for i in range(1, n - 1):
                    idx_prev = sorted_indices[i - 1]
                    idx_next = sorted_indices[i + 1]
                    dist = (F[indices[idx_next], m] - F[indices[idx_prev], m]) / f_range
                    crowd[sorted_indices[i]] += dist

            return crowd

        def select_parents(pop_x, pop_f, k=2):
            """
            Select k parents using binary tournament selection based on rank and crowding distance.
            """
            fronts = fast_non_dominated_sort(pop_f)
            ranks = np.zeros(len(pop_f), dtype=int)
            for i, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = i + 1
            
            # Calculate crowding distance for each front
            crowd_distances = np.zeros(len(pop_f))
            for front in fronts:
                cd = crowding_distance(pop_f, front)
                for i, idx in enumerate(front):
                    crowd_distances[idx] = cd[i]
            
            def dominates_in_selection(i, j):
                """
                Return True if i is preferred to j.
                """
                if ranks[i] < ranks[j]:
                    return True
                if ranks[i] > ranks[j]:
                    return False
                # Same rank, compare crowding distance
                return crowd_distances[i] > crowd_distances[j]

            parents = []
            for _ in range(k):
                i = rng.integers(0, len(pop_f))
                j = rng.integers(0, len(pop_f))
                if dominates_in_selection(i, j):
                    parents.append(i)
                else:
                    parents.append(j)
            
            return parents

        def sbx_crossover(x1, x2, lo, hi, prob=0.9, eta=20.0):
            """
            Simulated Binary Crossover.
            """
            if rng.random() > prob:
                return x1.copy(), x2.copy()
            
            x1_child = x1.copy()
            x2_child = x2.copy()
            
            for i in range(dim):
                if rng.random() < 0.5:
                    # Do not crossover this gene
                    continue
                
                if x1[i] == x2[i]:
                    continue
                    
                if x1[i] > x2[i]:
                    x1[i], x2[i] = x2[i], x1[i]
                    
                u = rng.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1 / (eta + 1))
                else:
                    beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                
                c1 = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                c2 = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
                
                # Clip to bounds
                c1 = np.clip(c1, lo[i], hi[i])
                c2 = np.clip(c2, lo[i], hi[i])
                
                x1_child[i] = c1
                x2_child[i] = c2
            
            return x1_child, x2_child

        def polynomial_mutation(x, lo, hi, prob=0.1, eta=20.0):
            """
            Polynomial Mutation.
            """
            x_mut = x.copy()
            for i in range(dim):
                if rng.random() < prob:
                    u = rng.random()
                    if u <= 0.5:
                        delta = (2 * u) ** (1 / (eta + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                    
                    x_mut[i] += delta * (hi[i] - lo[i])
                    x_mut[i] = np.clip(x_mut[i], lo[i], hi[i])
            
            return x_mut

        # Main NSGA-II Loop
        while used + 2 * pop_size <= budget:
            # Create Offspring
            offspring_x = []
            offspring_f = []
            
            while len(offspring_x) < pop_size:
                # Select Parents
                p1_idx, p2_idx = select_parents(population_x, population_f, 2)
                p1 = population_x[p1_idx]
                p2 = population_x[p2_idx]
                
                # Crossover
                c1, c2 = sbx_crossover(p1, p2, lo, hi)
                
                # Mutation
                c1 = polynomial_mutation(c1, lo, hi)
                c2 = polynomial_mutation(c2, lo, hi)
                
                # Evaluate
                f1 = problem(c1)
                f2 = problem(c2)
                used += 2
                
                offspring_x.append(c1)
                offspring_f.append(f1)
                offspring_x.append(c2)
                offspring_f.append(f2)
                
                if used >= budget:
                    break
            
            # Trim offspring to pop_size if we overshot
            offspring_x = np.array(offspring_x[:pop_size])
            offspring_f = np.array(offspring_f[:pop_size])

            # Combine Parents and Offspring
            combined_x = np.vstack([population_x, offspring_x])
            combined_f = np.vstack([population_f, offspring_f])
            
            # Sort and Select
            fronts = fast_non_dominated_sort(combined_f)
            
            new_population_indices = []
            for front in fronts:
                if len(new_population_indices) + len(front) <= pop_size:
                    new_population_indices.extend(front)
                else:
                    # Need to select from this front
                    remaining = pop_size - len(new_population_indices)
                    cd = crowding_distance(combined_f, front)
                    # Sort by crowding distance descending
                    sorted_front = sorted(range(len(front)), key=lambda i: cd[i], reverse=True)
                    new_population_indices.extend([front[i] for i in sorted_front[:remaining]])
                    break
            
            # Update Population
            population_x = combined_x[new_population_indices]
            population_f = combined_f[new_population_indices]

        # Use remaining budget for random exploration or local search
        while used < budget:
            # Pick a random point from population and mutate it
            idx = rng.integers(0, pop_size)
            x_mut = polynomial_mutation(population_x[idx], lo, hi, prob=0.5, eta=5.0)
            f_mut = problem(x_mut)
            used += 1
            
            # If it's non-dominated, we could add it, but we don't need to for return value
            # Just ensuring we use the budget.
            # We can also update the population if it's better?
            # For simplicity, just evaluate.

        return population_x

# EVOLVE-BLOCK-END
