import itertools
import random
import numpy as np
import time

def construct(v, k, t):
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Helper to check if a block covers a t-subset
    def block_covers_tsub(block, ts):
        return all(p in block for p in ts)

    # Helper to get uncovered t-subset indices for a list of blocks
    def get_uncovered(blocks):
        covered = set()
        for b in blocks:
            for ts in itertools.combinations(b, t):
                covered.add(t_subset_to_idx[ts])
        return set(range(num_t_subsets)) - covered

    # Helper: Greedy reduction of a set of blocks. 
    # Removes redundant blocks.
    def greedy_reduce(blocks):
        blocks = list(blocks)
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                # Check if block i is redundant
                is_redundant = True
                for ts in itertools.combinations(blocks[i], t):
                    idx = t_subset_to_idx[ts]
                    # Check if covered by others
                    found = False
                    for j in range(len(blocks)):
                        if i == j: continue
                        if all(p in blocks[j] for p in ts):
                            found = True
                            break
                    if not found:
                        is_redundant = False
                        break
                if is_redundant:
                    blocks.pop(i)
                    changed = True
                    break
        return blocks

    # Helper: Greedy construction of a valid covering (for initialization)
    def greedy_construct():
        uncovered = set(range(num_t_subsets))
        blocks = []
        while uncovered:
            # Pick a random uncovered t-subset
            target_idx = random.choice(list(uncovered))
            target_ts = all_t_subsets[target_idx]
            
            # Find best block covering target_ts
            best_block = None
            best_gain = -1
            remaining_points = [p for p in range(v) if p not in target_ts]
            need = k - t
            
            # Sample a few candidates to speed up, or all if small
            # C(11, 3) = 165 candidates. This is fast.
            candidates = list(itertools.combinations(remaining_points, need))
            random.shuffle(candidates)
            
            for combo in candidates:
                block = tuple(sorted(target_ts + tuple(combo)))
                gain = 0
                for ts in itertools.combinations(block, t):
                    if t_subset_to_idx[ts] in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = list(block)
            
            blocks.append(best_block)
            for ts in itertools.combinations(best_block, t):
                uncovered.discard(t_subset_to_idx[ts])
        
        # Final reduction
        return greedy_reduce(blocks)

    # GA Parameters
    pop_size = 50
    generations = 100
    start_time = time.time()
    time_limit = 110  # Leave buffer for 120s limit
    
    # Initialize Population
    population = []
    for _ in range(pop_size):
        population.append(greedy_construct())
    
    # Evaluate initial fitness
    # Fitness: (num_blocks, num_uncovered) -> minimize both
    def get_fitness(ind):
        uncovered = get_uncovered(ind)
        return (len(ind), len(uncovered))
    
    best_individual = min(population, key=get_fitness)
    best_fitness = get_fitness(best_individual)
    
    # GA Loop
    for gen in range(generations):
        if time.time() - start_time > time_limit:
            break
        
        new_population = []
        
        # Elitism: Keep the best individual
        new_population.append(best_individual)
        
        while len(new_population) < pop_size:
            # Selection: Tournament
            p1 = random.choice(population)
            p2 = random.choice(population)
            if random.random() < 0.5:
                p1, p2 = p2, p1
            
            # Crossover: Union + Reduce
            if random.random() < 0.8: # Crossover rate
                combined = list(set(tuple(b) for b in p1 + p2))
                # Convert back to lists
                combined = [list(b) for b in combined]
                child = greedy_reduce(combined)
            else:
                child = list(p1) # Clone
            
            # Mutation
            if random.random() < 0.7: # Mutation rate
                # Mutation 1: Try to remove a random block
                if len(child) > 1 and random.random() < 0.5:
                    idx = random.randint(0, len(child) - 1)
                    new_child = child[:idx] + child[idx+1:]
                    # Check if valid
                    unc = get_uncovered(new_child)
                    if len(unc) == 0:
                        child = new_child
                # Mutation 2: Swap a block
                else:
                    if child:
                        idx = random.randint(0, len(child) - 1)
                        # Generate a new random block
                        # To be smart, pick a block covering a random uncovered subset if any exist
                        unc = get_uncovered(child)
                        if unc:
                            target_idx = random.choice(list(unc))
                            target_ts = all_t_subsets[target_idx]
                            remaining_points = [p for p in range(v) if p not in target_ts]
                            need = k - t
                            combo = random.sample(remaining_points, need)
                            new_block = list(sorted(target_ts + tuple(combo)))
                            child[idx] = new_block
                        else:
                            # Random block
                            new_block = list(random.sample(range(v), k))
                            child[idx] = new_block
            
            new_population.append(child)
        
        # Update population
        population = new_population
        
        # Update best
        current_best = min(population, key=get_fitness)
        current_fitness = get_fitness(current_best)
        
        if current_fitness < best_fitness:
            best_individual = current_best
            best_fitness = current_fitness
    
    return best_individual
