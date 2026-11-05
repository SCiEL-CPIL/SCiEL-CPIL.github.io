import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 118  # Safety margin for 120s limit
    
    # 1. Precomputation
    points = range(v)
    all_triples = list(itertools.combinations(points, t))
    num_triples = len(all_triples)
    triple_to_idx = {tri: i for i, tri in enumerate(all_triples)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute coverage: list of triple indices for each block
    # Also precompute bitsets for fast intersection/coverage checks
    block_triple_indices = [[] for _ in range(num_blocks)]
    block_bitsets = [0] * num_blocks
    
    for i, b in enumerate(all_blocks):
        tris_in_block = []
        bs = 0
        for tri in itertools.combinations(b, t):
            idx = triple_to_idx[tri]
            tris_in_block.append(idx)
            bs |= (1 << idx)
        block_triple_indices[i] = tris_in_block
        block_bitsets[i] = bs

    # Helper: Greedy construction with randomization
    def greedy_construct(seed_offset=0):
        random.seed(42 + seed_offset)
        uncovered_bitset = (1 << num_triples) - 1
        selected = []
        selected_set = set()
        
        # We want to be fast. 
        # Optimization: Instead of scanning all 346k blocks every step, 
        # we can sample or use a heuristic. But for initial population, 
        # a few good greedy runs are worth it.
        # To speed up, we can pre-shuffle blocks? No, gain depends on state.
        
        # Let's limit the number of blocks we consider per step if time is tight?
        # No, let's just do full scans but break early if time is low.
        
        while uncovered_bitset:
            if time.time() > deadline - 20:
                break
            best_gain = -1
            best_idx = -1
            
            # To speed this up, we can randomly sample a subset of blocks 
            # if the full scan is too slow. 
            # 346k blocks * bit_count is fast in C (Python int bit ops).
            # Let's try full scan first.
            
            for i in range(num_blocks):
                if i in selected_set:
                    continue
                gain = (block_bitsets[i] & uncovered_bitset).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                elif gain == best_gain and gain > 0:
                    # Random tie breaking to ensure diversity in population
                    if random.random() < 0.05:
                        best_gain = gain
                        best_idx = i
            
            if best_gain <= 0:
                break
                
            selected_set.add(best_idx)
            selected.append(best_idx)
            uncovered_bitset &= ~block_bitsets[best_idx]
            
        return selected

    # Helper: Compaction (Local Search)
    # Removes any block that is redundant.
    def compact_solution(block_indices):
        if not block_indices:
            return []
        
        sol_set = set(block_indices)
        # Count coverage
        counts = [0] * num_triples
        for b_idx in sol_set:
            for tri_idx in block_triple_indices[b_idx]:
                counts[tri_idx] += 1
        
        # Check if valid
        if any(c == 0 for c in counts):
            # If invalid, we can't just remove blocks. 
            # We should return the original or try to repair?
            # For GA fitness, invalid solutions should be penalized heavily.
            # But for compaction, if it's invalid, removing a block makes it worse.
            # So we only compact valid solutions.
            return list(sol_set)
            
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Iterate over a copy because we modify the set
            for b_idx in list(sol_set):
                # Check if b_idx is redundant
                redundant = True
                for tri_idx in block_triple_indices[b_idx]:
                    if counts[tri_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    sol_set.remove(b_idx)
                    for tri_idx in block_triple_indices[b_idx]:
                        counts[tri_idx] -= 1
                    improved = True
        
        return list(sol_set)

    # Helper: Evaluate fitness
    # Lower is better. Invalid solutions get a huge penalty.
    def evaluate(block_indices):
        if not block_indices:
            return float('inf')
        sol_set = set(block_indices)
        counts = [0] * num_triples
        for b_idx in sol_set:
            for tri_idx in block_triple_indices[b_idx]:
                counts[tri_idx] += 1
        
        uncovered = sum(1 for c in counts if c == 0)
        if uncovered > 0:
            return len(sol_set) + uncovered * 1000
        return len(sol_set)

    # 1. Initialize Population
    pop_size = 20
    population = []
    
    # Create diverse initial individuals
    for i in range(pop_size):
        if time.time() > deadline - 30:
            break
        ind = greedy_construct(seed_offset=i)
        # Compact it
        ind = compact_solution(ind)
        population.append(ind)
        
    # Remove duplicates in population (optional, but good for diversity)
    # Represent as frozenset for hashing
    unique_pop = []
    seen = set()
    for ind in population:
        key = frozenset(ind)
        if key not in seen:
            seen.add(key)
            unique_pop.append(ind)
    population = unique_pop
    
    if not population:
        # Fallback
        return [all_blocks[i] for i in range(100)]

    # Find initial best
    best_ind = min(population, key=evaluate)
    best_score = evaluate(best_ind)
    
    # 2. Genetic Algorithm Loop
    # Parameters
    max_generations = 50
    offspring_size = len(population) * 2
    
    for gen in range(max_generations):
        if time.time() > deadline - 10:
            break
            
        offspring = []
        
        for _ in range(offspring_size):
            if time.time() > deadline - 10:
                break
            
            # Selection: Tournament
            # Pick 2 random individuals, choose the better one
            p1 = random.choice(population)
            p2 = random.choice(population)
            if evaluate(p1) <= evaluate(p2):
                parent1 = p1
            else:
                parent1 = p2
            
            p1 = random.choice(population)
            p2 = random.choice(population)
            if evaluate(p1) <= evaluate(p2):
                parent2 = p1
            else:
                parent2 = p2
            
            # Crossover: Uniform
            # Combine blocks from parent1 and parent2
            # We take a random subset of each? Or just union?
            # Union can be large. Intersection can be small/invalid.
            # Let's try: Take random blocks from P1 and P2 to form a new set.
            # Or: Start with P1, then swap some blocks with P2.
            
            child = list(parent1)
            
            # Mutation: Swap a random block in child with a random block from parent2 or random global
            num_swaps = random.randint(1, 3)
            for _ in range(num_swaps):
                if not child:
                    break
                idx_to_swap = random.randint(0, len(child) - 1)
                
                # Choose new block
                if random.random() < 0.5 and parent2:
                    new_block = random.choice(parent2)
                else:
                    new_block = random.randint(0, num_blocks - 1)
                
                child[idx_to_swap] = new_block
            
            # Also add a random block occasionally
            if random.random() < 0.2:
                child.append(random.randint(0, num_blocks - 1))
            
            # Remove duplicates
            child = list(set(child))
            
            # Compact (Repair)
            child = compact_solution(child)
            
            if child:
                offspring.append(child)
        
        if not offspring:
            break
            
        # Combine population and offspring
        new_population = population + offspring
        
        # Truncate to pop_size
        # Sort by fitness
        # To avoid re-evaluating everything, we could cache, but evaluate is fast enough for small sets?
        # Actually, evaluate is O(|S| * 35). |S| ~ 70. 70*35 = 2450 ops.
        # 40 individuals * 2450 = ~100k ops. Very fast.
        
        new_population.sort(key=evaluate)
        population = new_population[:pop_size]
        
        # Update best
        current_best = population[0]
        current_score = evaluate(current_best)
        if current_score < best_score:
            best_score = current_score
            best_ind = current_best
            
    # Final Polish on Best
    final_sol = compact_solution(best_ind)
    
    # Verify validity
    counts = [0] * num_triples
    for b_idx in final_sol:
        for tri_idx in block_triple_indices[b_idx]:
            counts[tri_idx] += 1
            
    if any(c == 0 for c in counts):
        # If somehow invalid, fallback to a valid greedy solution
        # We should have kept a valid one in population, but just in case
        # Re-run a quick greedy
        fallback = greedy_construct(seed_offset=999)
        fallback = compact_solution(fallback)
        if fallback:
            final_sol = fallback
        else:
            final_sol = best_ind # Last resort

    result_blocks = [all_blocks[i] for i in final_sol]
    return result_blocks
