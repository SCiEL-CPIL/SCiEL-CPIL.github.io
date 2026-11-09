"""Genetic algorithm approach for covering design C(19,8,5)."""
import itertools
import random
import numpy as np
from collections import Counter
import time

def construct(v, k, t):
    """
    Construct a covering design using a Genetic Algorithm:
    1. Initialize a population with greedy solutions.
    2. Evolve via Crossover (merge & reduce) and Mutation (swap/replace).
    3. Select best individuals to form the next generation.
    """
    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Map each k-subset to the set of t-subset indices it covers
    # This is crucial for speed in GA operations
    # C(19,8) = 75582 blocks. Each covers 56 t-subsets.
    # We can store this as a dictionary: block_tuple -> set(t_indices)
    # Memory: 75582 * 56 * 8 bytes ~ 34MB. Acceptable.
    block_cover_cache = {}
    
    def get_block_coverage(block):
        """Returns set of t-subset indices covered by a block."""
        if block not in block_cover_cache:
            cov = set()
            for ts in itertools.combinations(block, t):
                cov.add(t_subset_to_idx[ts])
            block_cover_cache[block] = cov
        return block_cover_cache[block]

    def greedy_cover():
        """Generate a valid covering using greedy heuristic."""
        uncovered = set(range(num_t_subsets))
        blocks = []
        
        while uncovered:
            # Pick an uncovered t-subset
            # To improve diversity in GA, we can randomize the choice slightly
            # but for initialization, standard greedy is fine.
            target_idx = random.choice(list(uncovered))
            target_ts = all_t_subsets[target_idx]
            
            # Find best block containing target_ts
            best_block = None
            best_count = -1
            
            remaining_points = [p for p in range(v) if p not in target_ts]
            need = k - t
            
            # Sample if the space is too large, but C(11,3)=165 is small
            for combo in itertools.combinations(remaining_points, need):
                block = tuple(sorted(target_ts + tuple(combo)))
                cov = get_block_coverage(block)
                count = len(cov & uncovered)
                if count > best_count:
                    best_count = count
                    best_block = block
            
            if best_block:
                blocks.append(best_block)
                cov = get_block_coverage(best_block)
                uncovered -= cov
        
        return blocks

    def reduce_covering(blocks):
        """Remove redundant blocks from a covering."""
        # Calculate coverage counts for each t-subset
        # coverage_count[i] = number of blocks covering t-subset i
        # If a block is removed, we need to check if any t-subset becomes uncovered.
        # Efficient way:
        # 1. Count how many times each t-subset is covered.
        # 2. For each block, check if it covers any t-subset that is covered ONLY by it.
        
        # This O(N * C(k,t)) check per block is okay for small N.
        # Let's implement a simpler iterative removal.
        changed = True
        current_blocks = list(blocks)
        
        while changed:
            changed = False
            for i in range(len(current_blocks)):
                block_i = current_blocks[i]
                cov_i = get_block_coverage(block_i)
                
                # Check if any t-subset in cov_i is NOT covered by any other block
                is_redundant = True
                for ts_idx in cov_i:
                    # Check other blocks
                    found = False
                    for j in range(len(current_blocks)):
                        if j == i: continue
                        if ts_idx in get_block_coverage(current_blocks[j]):
                            found = True
                            break
                    if not found:
                        is_redundant = False
                        break
                
                if is_redundant:
                    current_blocks.pop(i)
                    changed = True
                    break
        return current_blocks

    def crossover(parent1, parent2):
        """
        Crossover: Merge two coverings and reduce.
        This creates a new valid covering that is a subset of the union.
        """
        # Union of blocks
        merged_blocks = list(set(parent1) | set(parent2))
        # Reduce to remove redundancies
        return reduce_covering(merged_blocks)

    def mutate(blocks):
        """
        Mutation: 
        1. Try to replace a block with a better one (if it helps reduce size or is neutral).
        2. Try to remove redundant blocks.
        """
        # First, try reduction
        reduced = reduce_covering(blocks)
        
        # If reduction didn't help, try swapping one block
        if len(reduced) == len(blocks):
             # Pick a random block to replace
             if not reduced: return reduced
             idx = random.randint(0, len(reduced) - 1)
             old_block = reduced[idx]
             
             # Find a replacement block that covers at least as much unique coverage
             # We want to maintain validity.
             # Simple heuristic: pick a random t-subset covered by old_block, 
             # and find a new block covering it that covers more of the "weakly covered" t-subsets.
             
             cov_old = get_block_coverage(old_block)
             # Calculate coverage counts in the current solution
             # This is expensive, so we only do it if we have time/budget.
             # For simplicity in this step, we just pick a random valid block
             # that covers at least one t-subset from old_block.
             
             # To ensure validity, we must ensure that any t-subset ONLY covered by old_block
             # is also covered by the new block.
             
             # Let's identify "critical" t-subsets for old_block
             critical_ts = set()
             for ts_idx in cov_old:
                 # Check if covered by others
                 covered_by_other = False
                 for j in range(len(reduced)):
                     if j == idx: continue
                     if ts_idx in get_block_coverage(reduced[j]):
                         covered_by_other = True
                         break
                 if not covered_by_other:
                     critical_ts.add(ts_idx)
             
             # If no critical t-subsets, we can replace with any block that covers at least one of old_block's t-subsets
             # to potentially allow further reduction? No, if we replace, we must cover critical_ts.
             
             if critical_ts:
                 # New block MUST cover all critical_ts
                 # Find intersection of all k-subsets covering critical_ts?
                 # This is complex. 
                 # Alternative: If critical_ts is non-empty, we cannot simply replace unless the new block covers all of them.
                 # Let's just return the reduced version if we can't find a safe swap easily.
                 # A safer mutation: Randomly swap a non-critical block?
                 # Actually, if a block has critical t-subsets, it's "essential".
                 # We can try to find a block that covers critical_ts and maybe some others.
                 
                 # Let's pick a random critical ts
                 target_ts_idx = random.choice(list(critical_ts))
                 target_ts = all_t_subsets[target_ts_idx]
                 
                 # Find a block containing target_ts
                 remaining_points = [p for p in range(v) if p not in target_ts]
                 need = k - t
                 candidates = []
                 for combo in itertools.combinations(remaining_points, need):
                     new_block = tuple(sorted(target_ts + tuple(combo)))
                     cov_new = get_block_coverage(new_block)
                     # Must cover all critical_ts
                     if critical_ts.issubset(cov_new):
                         candidates.append(new_block)
                 
                 if candidates:
                     new_block = random.choice(candidates)
                     reduced[idx] = new_block
                     # Try reduction again
                     reduced = reduce_covering(reduced)
        
        return reduced

    # --- Main GA Loop ---
    
    # Parameters
    pop_size = 20
    generations = 50
    time_limit = 110  # seconds
    
    # Initialize Population
    population = []
    start_time = time.time()
    
    for i in range(pop_size):
        if time.time() - start_time > time_limit * 0.2:
            break
        # Add some randomness to greedy? 
        # The greedy function above uses random.choice for target, so it's already stochastic.
        pop_ind = greedy_cover()
        pop_ind = reduce_covering(pop_ind)
        population.append(pop_ind)
    
    if not population:
        # Fallback
        return greedy_cover()
    
    best_ind = min(population, key=len)
    best_size = len(best_ind)
    
    for gen in range(generations):
        if time.time() - start_time > time_limit:
            break
            
        next_population = []
        
        # Selection: Elitism
        # Keep the best individual
        next_population.append(best_ind)
        
        # Crossover and Mutation to fill population
        while len(next_population) < pop_size:
            if time.time() - start_time > time_limit:
                break
                
            # Select parents (Tournament selection)
            p1 = random.choice(population)
            p2 = random.choice(population)
            
            # Crossover
            child = crossover(p1, p2)
            
            # Mutation
            if random.random() < 0.7: # 70% mutation rate
                child = mutate(child)
            
            next_population.append(child)
        
        # Evaluate next population and update best
        for ind in next_population:
            if len(ind) < best_size:
                best_size = len(ind)
                best_ind = ind
        
        # Replace population with next generation (with some overlap for diversity?)
        # Simple: replace entirely, but keep best
        population = next_population
        
    return list(best_ind)
