import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Population size and generations
        # We need to ensure total evaluations <= budget
        pop_size = 20
        # Initial population + (generations * pop_size) <= budget
        # 20 + G * 20 <= 200 => G <= 9
        n_gen = (budget - pop_size) // pop_size
        
        # 1. Initialization
        population = np.random.rand(pop_size, dim) * (hi - lo) + lo
        # Evaluate initial population
        objectives = np.array([problem(x) for x in population])
        
        # State tracking
        all_evaluated = 0
        for x in population:
            all_evaluated += 1
            
        # Helper functions for NSGA-II
        def non_dominated_sort(objs):
            """
            Returns ranks for each individual.
            """
            n = len(objs)
            ranks = np.zeros(n, dtype=int)
            S = [set() for _ in range(n)]
            n_dominates = [0] * n
            front = []
            
            for p in range(n):
                for q in range(n):
                    if p == q: continue
                    if np.all(objs[p] <= objs[q]) and np.any(objs[p] < objs[q]):
                        S[p].add(q)
                    elif np.all(objs[q] <= objs[p]) and np.any(objs[q] < objs[p]):
                        n_dominates[p] += 1
            
            current_front = [p for p in range(n) if n_dominates[p] == 0]
            rank = 0
            while current_front:
                for p in current_front:
                    ranks[p] = rank
                next_front = []
                for p in current_front:
                    for q in S[p]:
                        n_dominates[q] -= 1
                        if n_dominates[q] == 0:
                            next_front.append(q)
                current_front = next_front
                rank += 1
            return ranks

        def crowding_distance(objs, ranks):
            """
            Returns crowding distance for each individual.
            """
            n = len(objs)
            distances = np.zeros(n)
            # Group by rank
            unique_ranks = np.unique(ranks)
            for r in unique_ranks:
                indices = np.where(ranks == r)[0]
                if len(indices) <= 2:
                    distances[indices] = np.inf
                    continue
                for m in range(n_obj):
                    # Sort by objective m
                    sorted_indices = indices[np.argsort(objs[indices, m])]
                    min_val = objs[sorted_indices[0], m]
                    max_val = objs[sorted_indices[-1], m]
                    if max_val == min_val:
                        distances[sorted_indices] = np.inf
                        continue
                    # End points
                    distances[sorted_indices[0]] = np.inf
                    distances[sorted_indices[-1]] = np.inf
                    for i in range(1, len(sorted_indices) - 1):
                        prev_idx = sorted_indices[i-1]
                        next_idx = sorted_indices[i+1]
                        dist = (objs[next_idx, m] - objs[prev_idx, m]) / (max_val - min_val)
                        distances[sorted_indices[i]] += dist
            return distances

        def tournament_select(pop, obj, ranks, dists, k=2):
            """
            Select one individual from pop using tournament selection.
            """
            idxs = np.random.choice(len(pop), k, replace=False)
            best_idx = -1
            for i in idxs:
                if best_idx == -1:
                    best_idx = i
                else:
                    # Compare ranks first, then crowding distance
                    if ranks[i] < ranks[best_idx]:
                        best_idx = i
                    elif ranks[i] == ranks[best_idx]:
                        if dists[i] > dists[best_idx]:
                            best_idx = i
            return pop[best_idx].copy()

        def sbx_crossover(x1, x2, lo, hi, eta=20):
            """
            Simulated Binary Crossover.
            Returns two children.
            """
            c1 = x1.copy()
            c2 = x2.copy()
            for i in range(len(x1)):
                if np.random.rand() < 0.5:
                    x1i, x2i = x1[i], x2[i]
                    if x1i == x2i:
                        continue
                    beta = 1.0
                    if np.random.rand() < 0.5:
                        if x1i > x2i:
                            beta = (x1i - x2i) / (hi[i] - lo[i])
                        else:
                            beta = (x2i - x1i) / (hi[i] - lo[i])
                    else:
                        if x1i < x2i:
                            beta = (x2i - x1i) / (hi[i] - lo[i])
                        else:
                            beta = (x1i - x2i) / (hi[i] - lo[i])
                    
                    if beta <= 0:
                        beta = 1e-10
                    if beta >= 1:
                        beta = 1 - 1e-10
                        
                    alpha = 1 / (beta * eta + 1)
                    beta_q1 = (0.5 * (1 + beta)) ** alpha
                    beta_q2 = (0.5 * (1 + 1/beta)) ** alpha
                    
                    c1[i] = 0.5 * ((x1i + x2i) - beta_q1 * (x1i - x2i))
                    c2[i] = 0.5 * ((x1i + x2i) + beta_q2 * (x1i - x2i))
                    c1[i] = np.clip(c1[i], lo[i], hi[i])
                    c2[i] = np.clip(c2[i], lo[i], hi[i])
            return c1, c2

        def polynomial_mutation(x, lo, hi, eta=20, prob_mut=None):
            """
            Polynomial Mutation.
            """
            if prob_mut is None:
                prob_mut = 1.0 / len(x)
            x_mut = x.copy()
            for i in range(len(x)):
                if np.random.rand() < prob_mut:
                    x_i = x_mut[i]
                    delta1 = (x_i - lo[i]) / (hi[i] - lo[i])
                    delta2 = (hi[i] - x_i) / (hi[i] - lo[i])
                    rand = np.random.rand()
                    if rand < 0.5:
                        xy = 1 - delta1
                        mut_pow = (2 * xy + 1) ** (eta + 1)
                        delta_q = (2 * (mut_pow - xy ** (eta + 1))) ** (1 / (eta + 1))
                        x_mut[i] = x_i - delta_q * (hi[i] - lo[i])
                    else:
                        xy = 1 - delta2
                        mut_pow = (2 * xy + 1) ** (eta + 1)
                        delta_q = (2 * (mut_pow - xy ** (eta + 1))) ** (1 / (eta + 1))
                        x_mut[i] = x_i + delta_q * (hi[i] - lo[i])
                    x_mut[i] = np.clip(x_mut[i], lo[i], hi[i])
            return x_mut

        # Main Loop
        for gen in range(n_gen):
            # Create offspring
            offspring = np.zeros((pop_size, dim))
            offspring_objs = np.zeros((pop_size, n_obj))
            
            for i in range(pop_size):
                # Selection
                p1 = tournament_select(population, objectives, ranks, dists)
                p2 = tournament_select(population, objectives, ranks, dists)
                
                # Crossover
                c1, c2 = sbx_crossover(p1, p2, lo, hi)
                
                # Mutation
                c1 = polynomial_mutation(c1, lo, hi)
                c2 = polynomial_mutation(c2, lo, hi)
                
                offspring[i] = c1
                # We need to evaluate c2 as well? 
                # Standard NSGA-II creates 2 offspring from 1 pair, but fills the offspring population.
                # Here we just take one offspring per iteration to keep it simple and fit budget?
                # No, usually we generate pop_size offspring.
                # To fit budget exactly, we generate pop_size offspring per generation.
                # But we only have budget for pop_size evaluations per gen.
                # So we evaluate c1. What about c2?
                # We can store c2 for the next generation or just discard?
                # Better: Generate 2 parents, produce 2 children, but only evaluate 1? No, that's wasteful.
                # Let's adjust: Generate pop_size/2 pairs, produce pop_size children, evaluate all.
                # This fits the budget: pop_size evals per gen.
                
                # Wait, the loop above iterates pop_size times.
                # If I evaluate c1 here, I use 1 eval.
                # I need to evaluate c2 somewhere.
                # Let's restructure the offspring generation.
                
                # Actually, simpler:
                # In each generation, we create pop_size offspring.
                # We do this by pop_size/2 crossover events.
                # Each event produces 2 children.
                # We evaluate all pop_size children.
                
                # Let's restart the offspring logic inside the gen loop.
                # The code above is a bit messy. I will rewrite the offspring creation cleanly.
                pass

            # Re-implement offspring creation correctly
            offspring = np.zeros((pop_size, dim))
            for i in range(0, pop_size, 2):
                p1 = tournament_select(population, objectives, ranks, dists)
                p2 = tournament_select(population, objectives, ranks, dists)
                c1, c2 = sbx_crossover(p1, p2, lo, hi)
                c1 = polynomial_mutation(c1, lo, hi)
                c2 = polynomial_mutation(c2, lo, hi)
                offspring[i] = c1
                offspring[i+1] = c2
            
            # Evaluate offspring
            offspring_objs = np.array([problem(x) for x in offspring])
            all_evaluated += pop_size
            
            # Combine
            combined_pop = np.vstack([population, offspring])
            combined_objs = np.vstack([objectives, offspring_objs])
            
            # Sort combined
            ranks_combined = non_dominated_sort(combined_objs)
            dists_combined = crowding_distance(combined_objs, ranks_combined)
            
            # Select next population
            next_pop = np.zeros((pop_size, dim))
            next_objs = np.zeros((pop_size, n_obj))
            
            selected_indices = []
            # Greedy selection based on rank and crowding
            # We need to pick pop_size individuals from combined (2*pop_size)
            
            # Sort combined by rank, then crowding distance
            sort_order = np.lexsort((dists_combined, ranks_combined))
            # Note: lexsort sorts by the last key first. So ranks first, then dists.
            # We want low rank (good) and high dist (good).
            # So we should sort ranks ascending, dists descending.
            # np.lexsort(( -dists_combined, ranks_combined ))?
            # Let's just do a stable sort on ranks, then stable sort on -dists within ranks?
            # Easier: create a list of indices, sort by key (rank, -dist)
            
            indices_sorted = sorted(range(len(combined_pop)), key=lambda i: (ranks_combined[i], -dists_combined[i]))
            
            for idx in indices_sorted:
                if len(selected_indices) >= pop_size:
                    break
                selected_indices.append(idx)
                
            next_pop = combined_pop[selected_indices]
            next_objs = combined_objs[selected_indices]
            
            # Update population
            population = next_pop
            objectives = next_objs
            ranks = non_dominated_sort(objectives)
            dists = crowding_distance(objectives, ranks)
            
        return None
