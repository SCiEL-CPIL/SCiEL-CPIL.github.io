"""Genetic algorithm approach for covering design C(19,8,5)."""
import itertools
import random
import numpy as np
from collections import Counter
import time

def construct(v, k, t):
    """
    Construct a covering design using a Genetic Algorithm approach:
    1. Initialize with a greedy solution.
    2. Evolve the population using mutation (block swaps/adds/removes) and crossover (union + prune).
    3. Optimize via local search within the GA loop.
    """
    # Generate all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index for fast lookup
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute coverage for all possible k-subsets? 
    # C(19,8) = 75582. Each covers 56 t-subsets.
    # Storing 75582 * 56 ints is ~4M ints. Manageable.
    # But generating them takes time. Let's do it lazily or precompute if fast enough.
    # Precomputation time: 75582 * 56 operations. ~4M ops. Very fast in Python if optimized?
    # Let's precompute a dictionary: block_tuple -> list of t_subset_indices
    
    block_to_t_indices = {}
    all_blocks = list(itertools.combinations(range(v), k))
    for b in all_blocks:
        # b is a tuple of k points
        t_indices = []
        for ts in itertools.combinations(b, t):
            t_indices.append(t_subset_to_idx[ts])
        block_to_t_indices[b] = t_indices
        
    # Helper to get coverage set quickly
    def get_coverage(block):
        return set(block_to_t_indices[tuple(block)])
        
    # Greedy initialization
    def greedy_cover():
        uncovered = set(range(num_t_subsets))
        blocks = []
        
        # For efficiency, we pick the best block for a specific uncovered t-subset
        # We iterate through all blocks? No, that's 75k.
        # We pick a random uncovered t-subset, then find the best block containing it.
        
        while uncovered:
            # Pick a random uncovered t-subset to focus on
            target_idx = random.choice(list(uncovered))
            target_ts = all_t_subsets[target_idx]
            
            # Find all blocks containing target_ts
            # Instead of scanning all 75k blocks, generate candidates
            # target_ts has t points. Need k-t more points from v-t remaining.
            target_set = set(target_ts)
            remaining_points = [p for p in range(v) if p not in target_set]
            need = k - t
            
            best_block = None
            best_gain = -1
            
            for combo in itertools.combinations(remaining_points, need):
                block = tuple(sorted(target_ts + tuple(combo)))
                # Check if this block is in our precomputed map (it should be)
                if block not in block_to_t_indices:
                    continue # Safety check, shouldn't happen
                
                # Count how many uncovered t-subsets this block covers
                # Optimization: Only count if it covers target_idx? Yes, it must.
                cov = block_to_t_indices[block]
                gain = 0
                for idx in cov:
                    if idx in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = list(block)
            
            if best_block is None:
                # Should not happen if v,k,t are valid
                break
                
            blocks.append(best_block)
            # Remove covered t-subsets
            cov = block_to_t_indices[tuple(best_block)]
            for idx in cov:
                uncovered.discard(idx)
        
        return blocks

    # Get initial greedy solution
    # Run a few times to get a diverse start? Or just one?
    # One is fast. Let's get one.
    initial_blocks = greedy_cover()
    
    # Function to calculate fitness
    def calculate_fitness(blocks):
        covered = set()
        for b in blocks:
            covered.update(block_to_t_indices[tuple(b)])
        uncovered_count = num_t_subsets - len(covered)
        # Penalize uncovered heavily, then minimize size
        return uncovered_count * 10000 + len(blocks)
        
    # Local search: Remove redundant blocks
    def remove_redundant(blocks):
        # Convert to set of tuples for faster intersection?
        # We need to check if a block's coverage is subset of others' union.
        # Efficient way: 
        # 1. Count frequency of each t-subset coverage.
        # 2. If a t-subset is covered by only 1 block, that block is essential.
        # 3. If a block has no essential t-subsets, it is redundant.
        
        # This logic is slightly flawed: a block might cover 2 unique t-subsets, 
        # but if we remove it, those 2 become uncovered. 
        # Redundant means: For ALL t-subsets in block B, there is ANOTHER block B' covering them.
        
        # Efficient check:
        # For each block i:
        #   unique_covered = 0
        #   for each t-subset ts in block i:
        #       if count_covering[ts] == 1:
        #           unique_covered += 1
        #   if unique_covered == 0:
        #       block i is redundant.
        
        # We need to maintain count_covering.
        
        def check_and_remove(blocks):
            if not blocks:
                return blocks
            
            # Calculate coverage counts
            cover_counts = Counter()
            for b in blocks:
                for idx in block_to_t_indices[tuple(b)]:
                    cover_counts[idx] += 1
            
            new_blocks = []
            for b in blocks:
                is_redundant = True
                for idx in block_to_t_indices[tuple(b)]:
                    if cover_counts[idx] == 1:
                        is_redundant = False
                        break
                if not is_redundant:
                    new_blocks.append(b)
            
            return new_blocks
        
        changed = True
        while changed:
            old_len = len(blocks)
            blocks = check_and_remove(blocks)
            if len(blocks) < old_len:
                changed = True
            else:
                changed = False
                
        return blocks

    # Initialize population
    pop_size = 20
    population = []
    
    # Add the greedy solution
    greedy_clean = remove_redundant(initial_blocks)
    population.append(greedy_clean)
    
    # Add mutated versions
    for _ in range(pop_size - 1):
        # Mutation: Start with greedy, randomly remove some blocks, then greedily fill?
        # Or just random swaps.
        # Let's try: Take greedy, remove 1-3 random blocks, then run a few steps of greedy repair.
        cand = list(greedy_clean)
        
        # Randomly remove a few blocks
        num_removals = random.randint(1, 3)
        for _ in range(num_removals):
            if cand:
                idx = random.randrange(len(cand))
                cand.pop(idx)
                
        # Greedy repair: Add blocks to cover uncovered
        covered = set()
        for b in cand:
            covered.update(block_to_t_indices[tuple(b)])
        uncovered = set(range(num_t_subsets)) - covered
        
        # Simple greedy repair
        steps = 0
        while uncovered and steps < 50:
            target_idx = random.choice(list(uncovered))
            target_ts = all_t_subsets[target_idx]
            target_set = set(target_ts)
            remaining_points = [p for p in range(v) if p not in target_set]
            need = k - t
            
            best_block = None
            best_gain = -1
            
            for combo in itertools.combinations(remaining_points, need):
                block = tuple(sorted(target_ts + tuple(combo)))
                if block in [tuple(b) for b in cand]: # Avoid duplicates
                    continue
                cov = block_to_t_indices[block]
                gain = 0
                for idx in cov:
                    if idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = list(block)
                    
            if best_block:
                cand.append(best_block)
                cov = block_to_t_indices[tuple(best_block)]
                for idx in cov:
                    uncovered.discard(idx)
            else:
                break
            steps += 1
            
        cand = remove_redundant(cand)
        population.append(cand)
        
    # GA Loop
    best_fitness = min(calculate_fitness(ind) for ind in population)
    best_individual = min(population, key=calculate_fitness)
    
    start_time = time.time()
    time_limit = 115.0 # Leave 5s buffer
    
    generation = 0
    while time.time() - start_time < time_limit:
        generation += 1
        
        # Selection: Tournament
        def select_tournament(pop):
            t_size = 3
            candidates = random.sample(pop, t_size)
            return min(candidates, key=calculate_fitness)
            
        new_population = []
        
        # Elitism: Keep the best individual
        best_ind = min(population, key=calculate_fitness)
        new_population.append(best_ind)
        
        while len(new_population) < pop_size:
            p1 = select_tournament(population)
            p2 = select_tournament(population)
            
            # Crossover: Union
            # Combine blocks from p1 and p2
            child_blocks = list(p1) + list(p2)
            # Remove duplicates
            seen = set()
            unique_child = []
            for b in child_blocks:
                tb = tuple(b)
                if tb not in seen:
                    seen.add(tb)
                    unique_child.append(b)
            
            # Prune redundant
            child_blocks = remove_redundant(unique_child)
            
            # Mutation
            if random.random() < 0.8: # Mutation rate
                # Try to replace a block to improve coverage or reduce size
                if child_blocks:
                    idx = random.randrange(len(child_blocks))
                    old_block = child_blocks[idx]
                    
                    # Strategy: Find a block that covers the most 'rare' subsets
                    # Or just swap with a random block?
                    # Let's try: Pick a random t-subset covered by old_block.
                    # Find a different block covering it.
                    old_cov = block_to_t_indices[tuple(old_block)]
                    if old_cov:
                        target_idx = random.choice(list(old_cov))
                        target_ts = all_t_subsets[target_idx]
                        target_set = set(target_ts)
                        remaining_points = [p for p in range(v) if p not in target_set]
                        need = k - t
                        
                        best_new_block = None
                        # Sample a few candidates to save time
                        candidates = random.sample(list(itertools.combinations(remaining_points, need)), min(50, len(list(itertools.combinations(remaining_points, need)))))
                        
                        for combo in candidates:
                            new_b = tuple(sorted(target_ts + tuple(combo)))
                            if new_b == tuple(old_block):
                                continue
                            # Check if this swap is beneficial
                            # Rough check: Does it cover the target? Yes.
                            # We'll accept it and let fitness decide.
                            best_new_block = list(new_b)
                            break # Just take the first valid one for speed? 
                            # Better: Take one that covers more unique things?
                            # For now, random swap.
                        
                        if best_new_block:
                            child_blocks[idx] = best_new_block
                            child_blocks = remove_redundant(child_blocks)
                            
            new_population.append(child_blocks)
            
        population = new_population
        
        # Update best
        current_best = min(population, key=calculate_fitness)
        current_best_fit = calculate_fitness(current_best)
        if current_best_fit < best_fitness:
            best_fitness = current_best_fit
            best_individual = current_best
            
        # If we found a valid covering (uncovered=0), try to minimize size further
        # The fitness function handles this.
        
    # Final cleanup
    best_individual = remove_redundant(best_individual)
    
    return best_individual
