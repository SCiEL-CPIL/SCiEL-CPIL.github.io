import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0
    
    t_subsets = list(itertools.combinations(range(v), t))
    total_t = len(t_subsets)
    full_mask = (1 << total_t) - 1
    t_subset_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    
    block_masks = {}
    all_blocks = []
    for block in itertools.combinations(range(v), k):
        mask = 0
        for ts in itertools.combinations(block, t):
            mask |= 1 << t_subset_to_idx[ts]
        block_masks[block] = mask
        all_blocks.append(block)
        
    def greedy_construct():
        uncovered = full_mask
        blocks = []
        while uncovered:
            best_block = None
            best_gain = -1
            for b in all_blocks:
                gain = (uncovered & block_masks[b]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block = b
            if best_block is None or best_gain == 0:
                break
            blocks.append(list(best_block))
            uncovered &= ~block_masks[best_block]
        return blocks

    initial_solution = greedy_construct()
    
    pop_size = 200
    population = [initial_solution]
    
    # Generate initial population by mutating initial solution
    for _ in range(pop_size - 1):
        mutated = [b[:] for b in initial_solution]
        # Random swaps
        for _ in range(random.randint(1, 3)):
            if mutated:
                idx = random.randint(0, len(mutated) - 1)
                mutated[idx] = list(random.choice(all_blocks))
        # Repair
        # Check validity
        union = 0
        for b in mutated:
            union |= block_masks[tuple(b)]
        if union != full_mask:
            uncovered = full_mask & ~union
            while uncovered:
                best_block = None
                best_gain = -1
                for b in all_blocks:
                    gain = (uncovered & block_masks[b]).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_block = b
                if best_block is None or best_gain == 0:
                    break
                mutated.append(list(best_block))
                uncovered &= ~block_masks[best_block]
        population.append(mutated)

    best_solution = initial_solution
    best_score = len(initial_solution)
    
    def get_score(blocks):
        # Assumes valid, but check just in case
        union = 0
        for b in blocks:
            union |= block_masks[tuple(b)]
        if union != full_mask:
            return 100000
        return len(blocks)

    # Evaluate initial
    scores = [get_score(ind) for ind in population]
    min_idx = scores.index(min(scores))
    if scores[min_idx] < best_score:
        best_solution = population[min_idx]
        best_score = scores[min_idx]

    while time.time() < deadline:
        new_population = []
        
        # Elitism
        sorted_pop = sorted(zip(scores, population), key=lambda x: x[0])
        elite_count = 10
        for i in range(min(elite_count, len(sorted_pop))):
            new_population.append(sorted_pop[i][1])
            
        while len(new_population) < pop_size:
            # Selection
            p1 = random.choice(population)
            p2 = random.choice(population)
            s1 = get_score(p1)
            s2 = get_score(p2)
            parent1 = p1 if s1 < s2 else p2
            parent2 = p2 if s1 < s2 else p1
            
            # Crossover: Uniform
            child = []
            p1_blocks = parent1[:]
            p2_blocks = parent2[:]
            random.shuffle(p1_blocks)
            random.shuffle(p2_blocks)
            
            for i in range(max(len(p1_blocks), len(p2_blocks))):
                if i < len(p1_blocks) and random.random() < 0.5:
                    child.append(p1_blocks[i])
                elif i < len(p2_blocks):
                    child.append(p2_blocks[i])
            
            # Deduplicate
            unique_blocks = []
            seen = set()
            for b in child:
                tb = tuple(b)
                if tb not in seen:
                    seen.add(tb)
                    unique_blocks.append(b)
            child = unique_blocks
            
            # Mutation
            if child and random.random() < 0.5:
                # Replace a block
                idx = random.randint(0, len(child) - 1)
                child[idx] = list(random.choice(all_blocks))
            if random.random() < 0.2:
                # Add a random block
                child.append(list(random.choice(all_blocks)))
            if child and random.random() < 0.2:
                # Remove a block
                idx = random.randint(0, len(child) - 1)
                child.pop(idx)
                
            # Repair if invalid
            union = 0
            for b in child:
                union |= block_masks[tuple(b)]
            if union != full_mask:
                uncovered = full_mask & ~union
                while uncovered:
                    best_block = None
                    best_gain = -1
                    for b in all_blocks:
                        gain = (uncovered & block_masks[b]).bit_count()
                        if gain > best_gain:
                            best_gain = gain
                            best_block = b
                    if best_block is None or best_gain == 0:
                        break
                    child.append(list(best_block))
                    uncovered &= ~block_masks[best_block]
                    
            new_population.append(child)
            
        population = new_population
        scores = [get_score(ind) for ind in population]
        
        min_idx = scores.index(min(scores))
        if scores[min_idx] < best_score:
            best_score = scores[min_idx]
            best_solution = population[min_idx]

    # Final check
    union = 0
    for b in best_solution:
        union |= block_masks[tuple(b)]
    if union != full_mask:
        return initial_solution
        
    return best_solution
