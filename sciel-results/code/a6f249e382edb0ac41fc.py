"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-Island Differential Evolution with Error Repair.
    
    Uses 5 independent DE populations (islands) that compete and migrate.
    Each island uses a 'Best/1' mutation operator.
    A key structural feature is 'error_repair': when a mutated candidate
    falls outside the [-5, 5] bounds, it is clipped (repaired) back into
    the domain rather than being rejected or penalized. This is tested
    against the hypothesis that repair maintains population diversity
    and prevents stagnation in constrained multimodal landscapes.
    """

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 310.0  # safety margin vs 330 s wall-clock kill

        rng = np.random.default_rng(42)

        # ---- Global bookkeeping -------------------------------------
        x_best_global = None
        f_best_global = np.inf
        n_calls = 0

        def record(x, f):
            nonlocal x_best_global, f_best_global, n_calls
            if f < f_best_global:
                f_best_global = f
                x_best_global = x.copy()
            n_calls += 1

        def check_deadline():
            return n_calls >= budget or time.time() > deadline

        # ---- Multi-Island DE Setup ----------------------------------
        n_islands = 5
        pop_size = 10  # Small populations to allow many generations within budget
        total_pop = n_islands * pop_size
        
        # Initialize islands with random points
        islands = []
        for _ in range(n_islands):
            pop = rng.uniform(-5.0, 5.0, (pop_size, dim))
            fvals = np.array([func(x) for x in pop])
            for x, f in zip(pop, fvals):
                record(x, f)
            
            # Sort population by fitness
            order = np.argsort(fvals)
            pop = pop[order]
            fvals = fvals[order]
            islands.append({'pop': pop, 'fvals': fvals})
            
            if check_deadline():
                return x_best_global, f_best_global

        # DE Parameters
        F = 0.8  # Mutation scale factor
        CR = 0.9  # Crossover rate
        migration_interval = 20  # Migrate every 20 generations
        gen_count = 0

        # Error Repair Function: Clips to bounds
        def repair(x):
            return np.clip(x, -5.0, 5.0)

        while not check_deadline():
            gen_count += 1
            
            # Migration: Every migration_interval generations, swap worst individuals
            if gen_count % migration_interval == 0:
                # Find global best and worst across islands
                best_island_idx = 0
                best_f = np.inf
                worst_island_idx = 0
                worst_f = -np.inf
                
                for i, isl in enumerate(islands):
                    if isl['fvals'][0] < best_f:
                        best_f = isl['fvals'][0]
                        best_island_idx = i
                    if isl['fvals'][-1] > worst_f:
                        worst_f = isl['fvals'][-1]
                        worst_island_idx = i
                
                # Migrate best individual from best island to worst island
                if best_island_idx != worst_island_idx:
                    best_indiv = islands[best_island_idx]['pop'][0].copy()
                    # Replace the worst individual in the worst island
                    islands[worst_island_idx]['pop'][-1] = best_indiv
                    f_new = func(best_indiv)
                    record(best_indiv, f_new)
                    islands[worst_island_idx]['fvals'][-1] = f_new
                    # Re-sort the worst island
                    order = np.argsort(islands[worst_island_idx]['fvals'])
                    islands[worst_island_idx]['pop'] = islands[worst_island_idx]['pop'][order]
                    islands[worst_island_idx]['fvals'] = islands[worst_island_idx]['fvals'][order]

            if check_deadline():
                break

            # Update each island
            for i, isl in enumerate(islands):
                pop = isl['pop']
                fvals = isl['fvals']
                new_pop = pop.copy()
                new_fvals = fvals.copy()
                
                for j in range(pop_size):
                    if check_deadline():
                        break
                    
                    # Differential Evolution Mutation: Best/1
                    # x_r1, x_r2 distinct from x_j and each other
                    r1, r2 = j
                    while r1 == j:
                        r1 = rng.integers(0, pop_size)
                    while r2 == j or r2 == r1:
                        r2 = rng.integers(0, pop_size)
                    
                    # Mutation
                    v = pop[0] + F * (pop[r1] - pop[r2]) # Using best (index 0) as base
                    
                    # Crossover
                    trial = pop[j].copy()
                    cross_point = rng.integers(0, dim)
                    for k in range(dim):
                        if rng.random() < CR or k == cross_point:
                            trial[k] = v[k]
                    
                    # Error Repair: Clip to bounds
                    trial = repair(trial)
                    
                    # Evaluation
                    f_trial = func(trial)
                    record(trial, f_trial)
                    
                    if check_deadline():
                        break
                    
                    # Selection
                    if f_trial < fvals[j]:
                        new_pop[j] = trial
                        new_fvals[j] = f_trial

                # Re-sort island
                order = np.argsort(new_fvals)
                isl['pop'] = new_pop[order]
                isl['fvals'] = new_fvals[order]

        return x_best_global, f_best_global
# EVOLVE-BLOCK-END
