import itertools
import time
import random
import sys
import math

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using a Genetic Algorithm.
    
    Strategy:
    1. Generate an initial population of valid covers using a fast greedy algorithm.
    2. Evolve the population using:
       - Selection: Tournament selection based on number of blocks (lower is better).
       - Crossover: Uniform crossover between two parents (randomly picking blocks from either).
       - Mutation: 
         - Swap: Replace a block with a random one (if valid or repairable).
         - Remove: Remove a redundant block.
         - Add: Add a random block.
       - Repair: Ensure the offspring is a valid cover by adding missing blocks via greedy.
    3. Maintain a "hall of fame" of the best unique solutions found.
    
    This differs from SA by maintaining a diverse population and using crossover to
    combine structural features of different good solutions, potentially escaping
    local minima more effectively than single-trajectory methods.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    # Map each t-subset to an index for fast lookup
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute t-subsets for all possible k-blocks
    # C(19, 8) = 75582 blocks. Each has C(8, 5) = 56 t-subsets.
    # Total ~4.2M operations. Should be fast enough.
    all_k_blocks = list(itertools.combinations(range(v), k))
    block_tsubs_cache = {}
    for b in all_k_blocks:
        block_tsubs_cache[b] = [t_subset_to_idx[ts] for ts in itertools.combinations(b, t)]
    
    # Helper to calculate coverage count for a list of blocks
    def get_cover_counts(blocks):
        counts = [0] * num_t_subsets
        for b in blocks:
            for idx in block_tsubs_cache[b]:
                counts[idx] += 1
        return counts

    # Helper to check if a set of blocks is a valid cover
    def is_valid_cover(counts):
        return all(c > 0 for c in counts)

    # Helper to repair a cover by adding missing blocks greedily
    def repair_cover(blocks, counts):
        blocks = list(blocks)
        counts = list(counts)
        block_set = set(blocks)
        
        # Find uncovered t-subsets
        uncovered_indices = [i for i, c in enumerate(counts) if c == 0]
        
        while uncovered_indices:
            # Pick a random uncovered t-subset
            target_idx = random.choice(uncovered_indices)
            target_subset = all_t_subsets[target_idx]
            
            # Find a block that covers this subset and maximizes additional coverage
            # We prefer blocks that cover many uncovered subsets.
            best_block = None
            best_gain = -1
            
            # Optimization: We can filter candidates. But for now, scan all?
            # C(19,8) is 75k. Scanning 75k * 56 per step is ~4M ops.
            # If we need 100 repairs, that's 400M ops. Might be slow.
            # Let's try a faster heuristic: pick a block containing the target subset.
            
            # Generate candidates containing target_subset
            # target_subset has t points. We need k-t more points from v-t.
            # C(v-t, k-t) candidates.
            # For v=19, t=5, k=8: C(14, 3) = 364 candidates. Much faster.
            
            points_in_target = set(target_subset)
            available_points = [x for x in range(v) if x not in points_in_target]
            
            candidates = []
            for extra in itertools.combinations(available_points, k - t):
                b = tuple(sorted(list(target_subset) + list(extra)))
                candidates.append(b)
            
            random.shuffle(candidates)
            
            for b in candidates:
                if b in block_set:
                    continue
                gain = 0
                for idx in block_tsubs_cache[b]:
                    if counts[idx] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b
            
            if best_block is not None:
                blocks.append(best_block)
                block_set.add(best_block)
                for idx in block_tsubs_cache[best_block]:
                    counts[idx] += 1
                # Remove covered from uncovered list
                # This is O(N) update. We can just rebuild uncovered_indices occasionally or keep it simple.
                # For simplicity and correctness, let's just check the condition in the while loop.
                # To avoid O(N) scan every time, we can just let the while loop re-evaluate.
                # But `uncovered_indices` is a list. We should filter it.
                uncovered_indices = [i for i in uncovered_indices if counts[i] == 0]
            else:
                # Should not happen if k>=t
                break
        
        return blocks, counts

    # Initial Greedy Construction for Population Initialization
    def greedy_initial():
        cover_count = [0] * num_t_subsets
        blocks = []
        uncovered_set = set(range(num_t_subsets))
        
        points = list(range(v))
        
        while uncovered_set:
            target_idx = random.choice(list(uncovered_set))
            target_subset = all_t_subsets[target_idx]
            
            block_points = set(target_subset)
            remaining_points = [x for x in points if x not in block_points]
            random.shuffle(remaining_points)
            
            for _ in range(k - t):
                best_point = -1
                best_gain = -1
                
                for x in remaining_points:
                    gain = 0
                    for rest in itertools.combinations(block_points, t - 1):
                        new_sub = tuple(sorted(rest + (x,)))
                        idx = t_subset_to_idx.get(new_sub)
                        if idx is not None and idx in uncovered_set:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                
                if best_point != -1:
                    block_points.add(best_point)
                    remaining_points.remove(best_point)
                else:
                    if remaining_points:
                        p = remaining_points.pop(0)
                        block_points.add(p)
            
            block = tuple(sorted(block_points))
            
            if block not in blocks:
                blocks.append(block)
                for ts_idx in block_tsubs_cache[block]:
                    if ts_idx in uncovered_set:
                        uncovered_set.remove(ts_idx)
        
        return blocks

    # Initialize Population
    pop_size = 10
    population = []
    hall_of_fame = set()
    best_len = float('inf')
    best_blocks = None
    
    random.seed(42)
    
    for _ in range(pop_size):
        b = greedy_initial()
        # Reduce
        counts = get_cover_counts(b)
        changed = True
        while changed:
            changed = False
            indices = list(range(len(b)))
            random.shuffle(indices)
            for i in indices:
                if i >= len(b): continue
                blk = b[i]
                removable = True
                for idx in block_tsubs_cache[blk]:
                    if counts[idx] == 1:
                        removable = False
                        break
                if removable:
                    for idx in block_tsubs_cache[blk]:
                        counts[idx] -= 1
                    b.pop(i)
                    changed = True
                    break
        
        if len(b) < best_len:
            best_len = len(b)
            best_blocks = list(b)
        
        key = tuple(sorted(b))
        if key not in hall_of_fame:
            hall_of_fame.add(key)
            population.append(b)
        
        # If population is full, stop adding duplicates
        if len(population) >= pop_size:
            break

    # Ensure we have enough population
    while len(population) < pop_size:
        b = greedy_initial()
        key = tuple(sorted(b))
        if key not in hall_of_fame:
            hall_of_fame.add(key)
            population.append(b)
        else:
            # If duplicate, just add it anyway to maintain size, or retry
            population.append(b)

    # GA Loop
    generation = 0
    while time.time() < deadline - 5.0:
        generation += 1
        new_population = []
        
        # Elitism: keep the best few
        population.sort(key=lambda x: len(x))
        elite_count = max(2, pop_size // 5)
        for i in range(elite_count):
            new_population.append(population[i])
        
        # Generate offspring
        while len(new_population) < pop_size:
            # Selection
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            
            # Crossover: Uniform
            # Combine blocks from both parents
            # Start with a random subset of parent1 and parent2
            child_blocks = set()
            for b in parent1:
                if random.random() < 0.5:
                    child_blocks.add(b)
            for b in parent2:
                if random.random() < 0.5:
                    child_blocks.add(b)
            
            # If child is empty or too small, ensure it's not empty
            if not child_blocks:
                child_blocks.add(random.choice(parent1))
            
            child_list = list(child_blocks)
            
            # Mutation
            # 1. Try to remove redundant blocks
            counts = get_cover_counts(child_list)
            changed = True
            while changed and time.time() < deadline - 5.0:
                changed = False
                if not child_list: break
                i = random.randint(0, len(child_list) - 1)
                blk = child_list[i]
                removable = True
                for idx in block_tsubs_cache[blk]:
                    if counts[idx] == 1:
                        removable = False
                        break
                if removable:
                    for idx in block_tsubs_cache[blk]:
                        counts[idx] -= 1
                    child_list.pop(i)
                    changed = True
            
            # 2. Try to swap a block (if valid)
            if child_list and random.random() < 0.3:
                i = random.randint(0, len(child_list) - 1)
                old_blk = child_list[i]
                new_blk = random.choice(all_k_blocks)
                
                if new_blk not in child_list:
                    # Check if swap maintains validity
                    # This is expensive to check fully. 
                    # Simple heuristic: if new block covers any unique t-subset of old block, it's risky.
                    # Let's just do the swap and repair if invalid.
                    # Actually, let's just add the new block and remove old, then repair.
                    
                    # Remove old
                    for idx in block_tsubs_cache[old_blk]:
                        counts[idx] -= 1
                    child_list[i] = new_blk
                    for idx in block_tsubs_cache[new_blk]:
                        counts[idx] += 1
                    
                    # Check validity
                    if not all(c > 0 for c in counts):
                        # Repair
                        child_list, counts = repair_cover(child_list, counts)
            
            # 3. Try to add a random block (if it helps reduce length later? No, adds length)
            # Maybe skip add mutation in GA as it increases cost.
            
            # Final Reduction
            changed = True
            while changed and time.time() < deadline - 5.0:
                changed = False
                if not child_list: break
                i = random.randint(0, len(child_list) - 1)
                blk = child_list[i]
                removable = True
                for idx in block_tsubs_cache[blk]:
                    if counts[idx] == 1:
                        removable = False
                        break
                if removable:
                    for idx in block_tsubs_cache[blk]:
                        counts[idx] -= 1
                    child_list.pop(i)
                    changed = True
            
            # Update Best
            if len(child_list) < best_len:
                best_len = len(child_list)
                best_blocks = list(child_list)
            
            key = tuple(sorted(child_list))
            if key not in hall_of_fame:
                hall_of_fame.add(key)
                new_population.append(child_list)
            else:
                # If duplicate, still add to new_population to maintain size, 
                # but maybe with slight mutation?
                # For now, just add it. Selection will handle quality.
                new_population.append(child_list)
            
            # Limit population growth in this loop
            if len(new_population) >= pop_size:
                break
        
        population = new_population
        
        # Optional: Local search on the best individual
        if best_blocks and time.time() < deadline - 10.0:
            # Quick local search on best_blocks
            b = list(best_blocks)
            counts = get_cover_counts(b)
            improved = True
            while improved and time.time() < deadline - 10.0:
                improved = False
                indices = list(range(len(b)))
                random.shuffle(indices)
                for i in indices:
                    if i >= len(b): continue
                    blk = b[i]
                    removable = True
                    for idx in block_tsubs_cache[blk]:
                        if counts[idx] == 1:
                            removable = False
                            break
                    if removable:
                        for idx in block_tsubs_cache[blk]:
                            counts[idx] -= 1
                        b.pop(i)
                        improved = True
                        break
            
            if len(b) < best_len:
                best_len = len(b)
                best_blocks = b

    return best_blocks
