import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    # 1. Precomputation
    all_tsets = list(itertools.combinations(range(v), t))
    num_tsets = len(all_tsets)
    tset_to_idx = {tset: i for i, tset in enumerate(all_tsets)}

    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)

    # For each block, find the indices of the t-subsets it covers
    t_subsets_per_block = len(list(itertools.combinations(range(k), t)))
    
    # Build BlockTsets: (num_blocks, t_subsets_per_block)
    BlockTsets = np.zeros((num_blocks, t_subsets_per_block), dtype=np.int32)
    
    # Reverse index: for each t-set, which blocks cover it
    ts_to_blocks = [[] for _ in range(num_tsets)]

    for i, block in enumerate(all_blocks):
        count = 0
        for sub in itertools.combinations(block, t):
            idx = tset_to_idx[sub]
            BlockTsets[i, count] = idx
            ts_to_blocks[idx].append(i)
            count += 1

    # List of arrays for easier access in python loops
    block_tsets_list = [BlockTsets[i] for i in range(num_blocks)]

    def polish_blocks(blocks):
        """Remove redundant blocks from the end of the list."""
        if not blocks:
            return blocks

        # Count how many times each t-set is covered
        cover_count = np.zeros(num_tsets, dtype=np.int32)
        for b in blocks:
            cover_count[block_tsets_list[b]] += 1

        # Iterate from end to start
        i = len(blocks) - 1
        while i >= 0:
            if time.time() > deadline - 2.0:
                break
            b = blocks[i]
            # Check if any t-set is covered only by this block
            can_remove = True
            for ts in block_tsets_list[b]:
                if cover_count[ts] == 1:
                    can_remove = False
                    break
            
            if can_remove:
                # Remove block b
                for ts in block_tsets_list[b]:
                    cover_count[ts] -= 1
                blocks.pop(i)
            else:
                i -= 1
        return blocks

    def local_search_swap(blocks, max_swaps=50):
        """
        Try to replace a block with another block that covers all the t-sets
        uniquely covered by the original block.
        Returns (blocks, improved_count)
        """
        if not blocks:
            return blocks, 0

        # Compute cover counts
        cover_count = np.zeros(num_tsets, dtype=np.int32)
        for b in blocks:
            cover_count[block_tsets_list[b]] += 1

        swaps_made = 0
        # We iterate through the blocks. If we make a swap, we restart the loop
        # because the cover counts have changed.
        
        while swaps_made < max_swaps:
            if time.time() > deadline - 2.0:
                break
            
            # We can randomize the order of checking blocks to diversify
            indices = list(range(len(blocks)))
            random.shuffle(indices)
            
            found_swap = False
            for idx_pos in indices:
                i = idx_pos
                if i >= len(blocks): continue
                
                b = blocks[i]
                
                # Find t-sets uniquely covered by block b
                unique_ts = []
                for ts in block_tsets_list[b]:
                    if cover_count[ts] == 1:
                        unique_ts.append(ts)
                
                if not unique_ts:
                    continue

                # Find candidate blocks that cover ALL unique_ts
                # Start with the blocks covering the first unique t-set
                candidates = set(ts_to_blocks[unique_ts[0]])
                for ts_idx in unique_ts[1:]:
                    candidates.intersection_update(ts_to_blocks[ts_idx])
                    if not candidates:
                        break

                if not candidates:
                    continue

                # Remove the block itself from candidates
                candidates.discard(b)

                if not candidates:
                    continue

                # Pick a candidate. 
                # Heuristic: Pick the one that minimizes the number of NEW unique t-sets it creates?
                # Or just pick randomly. Let's try minimizing "new uniqueness".
                # Actually, any valid replacement works for the count.
                # Let's pick the one that covers the most t-sets that are currently covered by >1 block?
                # No, that increases redundancy.
                # Let's just pick randomly to explore.
                
                candidate_list = list(candidates)
                c = candidate_list[random.randint(0, len(candidate_list) - 1)]
                
                # Perform swap
                # Update cover_count
                for ts in block_tsets_list[b]:
                    cover_count[ts] -= 1
                for ts in block_tsets_list[c]:
                    cover_count[ts] += 1
                
                blocks[i] = c
                swaps_made += 1
                found_swap = True
                break # Restart the loop after a change
            
            if not found_swap:
                break

        return blocks, swaps_made

    def run_greedy(seed=None):
        """
        Greedy set cover with random tie-breaking.
        """
        if seed is not None:
            random.seed(seed)
        
        uncovered = np.ones(num_tsets, dtype=bool)
        blocks = []
        
        # Pre-compute initial gains? No, gains change.
        # Gain of a block is the number of uncovered t-sets it covers.
        
        while uncovered.any():
            if time.time() > deadline - 5.0:
                return None
            
            # Calculate gain for all blocks
            # gains[b] = sum(uncovered[BlockTsets[b]])
            gains = uncovered[BlockTsets].sum(axis=1)
            
            max_gain = gains.max()
            if max_gain == 0:
                break
            
            # Find all blocks with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Randomly select one to diversify
            best_block_idx = candidates[random.randint(0, len(candidates) - 1)]
            
            blocks.append(best_block_idx)
            
            # Update uncovered
            uncovered[BlockTsets[best_block_idx]] = False

        return blocks

    def genetic_algorithm(pop_size=10, generations=20, mutation_rate=0.1):
        """
        A simple genetic algorithm to refine a set of blocks.
        Fitness: -len(blocks) (minimize size).
        Operators:
        - Selection: Tournament
        - Crossover: Uniform (keep blocks from one or the other, deduplicate)
        - Mutation: Swap a block with a random one (if valid) or remove a redundant one.
        """
        if time.time() > deadline - 10.0:
            return None

        # Initialize population with the current best and some random greedy solutions
        population = []
        
        # Add best so far
        if 'best_blocks' in locals() and best_blocks is not None:
            population.append(best_blocks[:])
        
        # Add some greedy solutions
        while len(population) < pop_size:
            sol = run_greedy(seed=random.randint(0, 10000))
            if sol is None:
                break
            sol = polish_blocks(sol)
            population.append(sol)
        
        if not population:
            return None

        best_in_pop = min(population, key=len)
        
        for gen in range(generations):
            if time.time() > deadline - 5.0:
                break
            
            new_population = []
            
            # Elitism: keep the best
            new_population.append(best_in_pop[:])
            
            # Generate offspring
            while len(new_population) < pop_size:
                if time.time() > deadline - 5.0:
                    break
                
                # Selection
                p1 = random.choice(population)
                p2 = random.choice(population)
                
                # Crossover: Uniform
                # Combine blocks from p1 and p2
                combined = list(set(p1 + p2))
                
                # If combined doesn't cover everything, it's invalid.
                # We need to check coverage.
                # For efficiency, we can just polish it. If it's not a cover, polish won't fix it.
                # So we must ensure it's a valid cover.
                # Simple check: does combined cover all t-sets?
                # We can compute cover_count.
                cover_count = np.zeros(num_tsets, dtype=np.int32)
                for b in combined:
                    cover_count[block_tsets_list[b]] += 1
                
                if not np.all(cover_count > 0):
                    # Invalid. Try to repair by adding blocks from greedy?
                    # Or just discard and pick another pair.
                    # For simplicity, discard.
                    continue
                
                offspring = polish_blocks(combined)
                
                # Mutation
                if random.random() < mutation_rate:
                    # Try to swap a block
                    if offspring:
                        idx = random.randint(0, len(offspring) - 1)
                        b = offspring[idx]
                        # Find a replacement
                        # Find unique t-sets of b
                        unique_ts = []
                        for ts in block_tsets_list[b]:
                            if cover_count[ts] == 1:
                                unique_ts.append(ts)
                        
                        if unique_ts:
                            # Find candidate
                            candidates = set(ts_to_blocks[unique_ts[0]])
                            for ts_idx in unique_ts[1:]:
                                candidates.intersection_update(ts_to_blocks[ts_idx])
                            candidates.discard(b)
                            if candidates:
                                c = random.choice(list(candidates))
                                offspring[idx] = c
                                offspring = polish_blocks(offspring)
                        else:
                            # Can remove
                            offspring.pop(idx)
                            offspring = polish_blocks(offspring)
                
                new_population.append(offspring)
            
            if not new_population:
                break
            
            population = new_population
            current_best = min(population, key=len)
            if len(current_best) < len(best_in_pop):
                best_in_pop = current_best[:]
        
        return best_in_pop

    # 2. Main Search Loop
    best_blocks = None
    best_len = float('inf')

    # Initial deterministic greedy (seed 0)
    sol = run_greedy(seed=0)
    if sol is not None:
        sol = polish_blocks(sol)
        sol, _ = local_search_swap(sol, max_swaps=10)
        sol = polish_blocks(sol)
        best_blocks = sol[:]
        best_len = len(sol)

    # Randomized restarts
    # We want to spend most of the time here.
    restart_id = 1
    while time.time() < deadline - 10.0:
        sol = run_greedy(seed=restart_id)
        restart_id += 1
        
        if sol is None:
            break
        
        sol = polish_blocks(sol)
        
        # If this solution is better, or close enough, try local search
        if len(sol) <= best_len + 3:
            sol, _ = local_search_swap(sol, max_swaps=20)
            sol = polish_blocks(sol)
            
            if len(sol) < best_len:
                best_len = len(sol)
                best_blocks = sol[:]
                # Try GA on the new best
                if time.time() < deadline - 15.0:
                    ga_result = genetic_algorithm(pop_size=5, generations=10, mutation_rate=0.2)
                    if ga_result is not None and len(ga_result) < best_len:
                        best_len = len(ga_result)
                        best_blocks = ga_result[:]
                        # Polish again
                        best_blocks = polish_blocks(best_blocks)
                        best_blocks, _ = local_search_swap(best_blocks, max_swaps=10)
                        best_blocks = polish_blocks(best_blocks)
                        if len(best_blocks) < best_len:
                            best_len = len(best_blocks)

    # Final polish on best solution
    if best_blocks is not None:
        best_blocks = polish_blocks(best_blocks)
        best_blocks, _ = local_search_swap(best_blocks, max_swaps=20)
        best_blocks = polish_blocks(best_blocks)
        
        # One more GA attempt if time permits
        if time.time() < deadline - 5.0:
            ga_result = genetic_algorithm(pop_size=10, generations=20, mutation_rate=0.1)
            if ga_result is not None and len(ga_result) < len(best_blocks):
                best_blocks = polish_blocks(ga_result)
                best_blocks, _ = local_search_swap(best_blocks, max_swaps=10)
                best_blocks = polish_blocks(best_blocks)

    if best_blocks is None:
        return []

    result = [list(all_blocks[b]) for b in best_blocks]
    return result
