"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random


class Optimizer:
    """NSGA-II style multi-objective optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Setup parameters
        pop_size = 20
        eta_c = 20.0  # Distribution index for crossover
        eta_m = 20.0  # Distribution index for mutation
        p_c = 0.9     # Crossover probability
        p_m = 1.0 / self.dim  # Mutation probability per gene

        # Initialize population
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Use random uniform initialization for simplicity, or LHS.
        # Given the budget is small, random is fine, but LHS is better for diversity.
        # Let's stick to random for standard EMO approach, or maybe stratified.
        # Actually, let's just use np.random.uniform.
        
        X = np.random.uniform(lo, hi, size=(pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        
        evals_used = pop_size
        remaining_budget = self.budget - evals_used

        # Helper function for non-dominated sorting
        def non_dominated_sort(F):
            n = len(F)
            S = [set() for _ in range(n)]
            n_dom = [0] * n
            fronts = [[]]
            
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].add(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1
                        
                if n_dom[i] == 0:
                    fronts[0].append(i)
            
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
                else:
                    break
                    
            return fronts

        # Helper function for crowding distance
        def crowding_distance(F, front_indices):
            if len(front_indices) <= 2:
                return [float('inf')] * len(front_indices)
            
            distances = [0.0] * len(front_indices)
            for m in range(self.n_obj):
                obj_vals = [F[idx][m] for idx in front_indices]
                min_val = min(obj_vals)
                max_val = max(obj_vals)
                if max_val == min_val:
                    continue
                sorted_indices = sorted(front_indices, key=lambda i: F[i][m])
                distances[front_indices.index(sorted_indices[0])] = float('inf')
                distances[front_indices.index(sorted_indices[-1])] = float('inf')
                for i in range(1, len(sorted_indices) - 1):
                    idx = sorted_indices[i]
                    distances[front_indices.index(idx)] += (F[sorted_indices[i+1]][m] - F[sorted_indices[i-1]][m]) / (max_val - min_val)
            return distances

        # Helper function for SBX Crossover
        def sbx_crossover(x1, x2):
            if random.random() < p_c:
                child1 = x1.copy()
                child2 = x2.copy()
                for i in range(self.dim):
                    if random.random() < 0.5:
                        if abs(x1[i] - x2[i]) > 1e-14:
                            beta1 = 1 + 2 * (min(x1[i], x2[i]) - lo[i]) / (x1[i] - x2[i])
                            alpha1 = 2 - beta1 ** (-(eta_c + 1))
                            beta2 = 1 + 2 * (hi[i] - max(x1[i], x2[i])) / (x1[i] - x2[i])
                            alpha2 = 2 - beta2 ** (-(eta_c + 1))
                            rand = random.random()
                            if rand <= 1/alpha1:
                                betaq = (rand * alpha1) ** (1/(eta_c+1))
                            else:
                                betaq = (1/(2 - rand * alpha2)) ** (1/(eta_c+1))
                            child1[i] = 0.5 * ((x1[i] + x2[i]) - betaq * (x1[i] - x2[i]))
                            child2[i] = 0.5 * ((x1[i] + x2[i]) + betaq * (x1[i] - x2[i]))
                        else:
                            child1[i] = x1[i]
                            child2[i] = x2[i]
            else:
                child1 = x1.copy()
                child2 = x2.copy()
            return child1, child2

        # Helper function for Polynomial Mutation
        def polynomial_mutation(x):
            for i in range(self.dim):
                if random.random() < p_m:
                    delta1 = (x[i] - lo[i]) / (hi[i] - lo[i])
                    delta2 = (hi[i] - x[i]) / (hi[i] - lo[i])
                    rand = random.random()
                    if rand < 0.5:
                        xy = 1 - delta1
                        deltaq = (2 * rand + (1 - 2 * rand) * (xy ** (eta_m + 1))) ** (1/(eta_m+1)) - 1
                    else:
                        xy = 1 - delta2
                        deltaq = 1 - (2 * (1 - rand) + (2 * rand - 1) * (xy ** (eta_m + 1))) ** (1/(eta_m+1))
                    x[i] += deltaq * (hi[i] - lo[i])
                    x[i] = np.clip(x[i], lo[i], hi[i])
            return x

        # Main Loop
        while evals_used < self.budget:
            # Create offspring
            offspring_X = []
            offspring_F = []
            
            while len(offspring_X) < pop_size and evals_used < self.budget:
                # Select parents using tournament selection
                i = random.randint(0, pop_size - 1)
                j = random.randint(0, pop_size - 1)
                
                # Simple tournament: pick 2, choose best based on rank then crowding
                # We need ranks and crowding distances for current population
                # For efficiency in this small loop, we can recompute or store.
                # Let's store them before the loop iteration.
                
                # Actually, standard NSGA-II creates an offspring population of size pop_size,
                # then combines parents and offspring, sorts, and selects next pop.
                # This requires 2 * pop_size evaluations per generation if we re-evaluate?
                # No, we only evaluate new offspring.
                
                # Let's refine the loop structure.
                # 1. Generate offspring (pop_size evaluations)
                # 2. Combine parents + offspring (2 * pop_size)
                # 3. Non-dominated sort
                # 4. Select best pop_size for next generation
                
                # To fit within budget, we must be careful.
                # Budget 200. Pop size 20.
                # Gen 0: Init (20 evals). Remaining 180.
                # Gen 1: Offspring (20 evals). Combine (40). Select (20).
                # This consumes 20 evals per generation.
                # 180 / 20 = 9 generations. Total 10 generations (incl init).
                # This is fine.
                
                x1 = X[i]
                x2 = X[j]
                c1, c2 = sbx_crossover(x1, x2)
                c1 = polynomial_mutation(c1)
                c2 = polynomial_mutation(c2)
                
                offspring_X.append(c1)
                evals_used += 1
                if evals_used < self.budget:
                    offspring_X.append(c2)
                    evals_used += 1

            # Evaluate offspring
            for x in offspring_X:
                offspring_F.append(problem(x))
            offspring_F = np.array(offspring_F)
            
            # Combine
            X_combined = np.vstack([X, np.array(offspring_X)])
            F_combined = np.vstack([F, offspring_F])
            
            # Non-dominated sort
            fronts = non_dominated_sort(F_combined)
            
            # Select next population
            new_X = []
            new_F = []
            n_remaining = pop_size
            
            for front in fronts:
                if len(front) <= n_remaining:
                    for idx in front:
                        new_X.append(X_combined[idx])
                        new_F.append(F_combined[idx])
                    n_remaining -= len(front)
                else:
                    # Need crowding distance
                    dists = crowding_distance(F_combined, front)
                    # Sort front by crowding distance descending
                    sorted_front = sorted(front, key=lambda i: dists[front.index(i)], reverse=True)
                    for idx in sorted_front[:n_remaining]:
                        new_X.append(X_combined[idx])
                        new_F.append(F_combined[idx])
                    n_remaining = 0
                    break
                    
            X = np.array(new_X)
            F = np.array(new_F)

        # Return the last population (or best points found so far)
        # The task says "Every returned objective vector is recorded".
        # Usually, we return the final population.
        return X

# EVOLVE-BLOCK-END
