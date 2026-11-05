import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start = time.time()
    deadline = start + 118
    rng = random.Random(1234567890)

    # Precompute all blocks and t-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    t_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(t_subsets)
    sub_idx = {s: i for i, s in enumerate(t_subsets)}

    # Precompute block masks
    # Using Python ints for bitsets is efficient for ~10k bits
    block_masks = [0] * n_blocks
    for bi, blk in enumerate(all_blocks):
        mask = 0
        for sub in itertools.combinations(blk, t):
            si = sub_idx[sub]
            mask |= 1 << si
        block_masks[bi] = mask

    def greedy_construction(rng, target_size=None):
        """Greedy: always pick block maximizing new coverage."""
        uncovered = (1 << n_subsets) - 1
        chosen = []
        chosen_set = set()
        
        while uncovered:
            best_bi = -1
            best_gain = 0
            best_mask = 0
            
            # Optimization: If target_size is reached, we might still need to finish if uncovered
            # But the standard greedy doesn't stop early unless uncovered is 0.
            # The original code had a target_size stop, but then filled up.
            # Let's just do standard greedy until covered.
            
            # Vectorized gain calculation might be faster?
            # gains = [ (block_masks[bi] & uncovered).bit_count() for bi in range(n_blocks) if bi not in chosen_set]
            # But iterating in Python is slow.
            
            # We can filter candidates
            candidates = [bi for bi in range(n_blocks) if bi not in chosen_set]
            if not candidates:
                break
                
            # To speed up, we can sample if n_blocks is huge, but here n_blocks is manageable (C(19,7) = 50388)
            # 50388 * 150 iterations * 120s is tight.
            # Let's stick to the loop but optimize it.
            
            for bi in candidates:
                gain = (block_masks[bi] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_bi = bi
                    best_mask = block_masks[bi]
                    if best_gain == uncovered.bit_count():
                        break
            
            if best_bi < 0:
                break
                
            chosen.append(best_bi)
            chosen_set.add(best_bi)
            uncovered &= ~best_mask
            
        return chosen, uncovered == 0

    def prune(blocks_list):
        """Remove redundant blocks."""
        chosen = list(blocks_list)
        chosen_set = set(chosen)
        counts = [0] * n_subsets
        for bi in chosen:
            mask = block_masks[bi]
            while mask:
                low = mask & -mask
                si = low.bit_length() - 1
                counts[si] += 1
                mask ^= low
        
        changed = True
        while changed:
            changed = False
            for i, bi in enumerate(chosen):
                mask = block_masks[bi]
                redundant = True
                while mask:
                    low = mask & -mask
                    si = low.bit_length() - 1
                    if counts[si] == 1:
                        redundant = False
                        break
                    mask ^= low
                if redundant:
                    mask = block_masks[bi]
                    while mask:
                        low = mask & -mask
                        si = low.bit_length() - 1
                        counts[si] -= 1
                        mask ^= low
                    chosen.pop(i)
                    chosen_set.discard(bi)
                    changed = True
                    break
        return chosen

    def crossover(p1, p2):
        """
        Custom crossover: 
        1. Union of blocks from p1 and p2.
        2. Greedily select blocks from this union to cover all subsets.
        3. If not covered, repair with global greedy.
        """
        union_set = set(p1) | set(p2)
        union_list = list(union_set)
        # Shuffle to introduce randomness? Or sort by potential?
        # Let's shuffle to explore different orderings
        rng.shuffle(union_list)
        
        uncovered = (1 << n_subsets) - 1
        chosen = []
        chosen_set = set()
        
        # Phase 1: Greedy selection from the union
        while uncovered and union_list:
            best_bi = -1
            best_gain = 0
            best_mask = 0
            # We iterate through the remaining candidates in the union
            # To make it faster, we can just pick the first one with max gain
            # Since union size is ~200-300, this is fast.
            for idx, bi in enumerate(union_list):
                if bi in chosen_set:
                    continue
                gain = (block_masks[bi] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_bi = bi
                    best_mask = block_masks[bi]
                    if best_gain == uncovered.bit_count():
                        break
            
            if best_bi < 0:
                break
            
            chosen.append(best_bi)
            chosen_set.add(best_bi)
            union_list.remove(best_bi) # Remove from candidate list
            uncovered &= ~best_mask
        
        # Phase 2: Repair if necessary
        if uncovered:
            # Use global greedy for remaining
            while uncovered:
                best_bi = -1
                best_gain = 0
                best_mask = 0
                for bi in range(n_blocks):
                    if bi in chosen_set:
                        continue
                    gain = (block_masks[bi] & uncovered).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_bi = bi
                        best_mask = block_masks[bi]
                if best_bi < 0:
                    break
                chosen.append(best_bi)
                chosen_set.add(best_bi)
                uncovered &= ~best_mask
        
        return chosen

    def mutate(blocks_list, mutation_rate=0.1):
        """
        Mutation: Randomly remove some blocks and repair.
        """
        chosen = list(blocks_list)
        if not chosen:
            return chosen
        
        n_remove = max(1, int(len(chosen) * mutation_rate))
        # Remove random blocks
        to_remove = set(rng.sample(chosen, min(n_remove, len(chosen))))
        chosen = [b for b in chosen if b not in to_remove]
        
        # Repair
        counts = [0] * n_subsets
        for bi in chosen:
            mask = block_masks[bi]
            while mask:
                low = mask & -mask
                si = low.bit_length() - 1
                counts[si] += 1
                mask ^= low
        
        uncovered = 0
        for si in range(n_subsets):
            if counts[si] == 0:
                uncovered |= 1 << si
        
        while uncovered:
            best_bi = -1
            best_gain = 0
            for bi in range(n_blocks):
                if bi in chosen: # Using list check is slow, but n is small. Better use set?
                    continue
                gain = (block_masks[bi] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_bi = bi
            if best_bi < 0:
                break
            chosen.append(best_bi)
            mask = block_masks[best_bi]
            while mask:
                low = mask & -mask
                si = low.bit_length() - 1
                counts[si] += 1
                mask ^= low
            uncovered &= ~block_masks[best_bi]
            
        return prune(chosen)

    def local_search(blocks_list, time_budget=5):
        """
        1-out/1-in swap search.
        """
        chosen = list(blocks_list)
        chosen_set = set(chosen)
        counts = [0] * n_subsets
        for bi in chosen:
            mask = block_masks[bi]
            while mask:
                low = mask & -mask
                si = low.bit_length() - 1
                counts[si] += 1
                mask ^= low
                
        current_time = time.time()
        improved = True
        while improved and time.time() - current_time < time_budget:
            improved = False
            for i, bi_out in enumerate(chosen):
                if time.time() - current_time > time_budget:
                    break
                # Check if removable
                mask_out = block_masks[bi_out]
                removable = True
                m = mask_out
                while m:
                    low = m & -m
                    si = low.bit_length() - 1
                    if counts[si] == 1:
                        removable = False
                        break
                    m ^= low
                if not removable:
                    continue
                
                # Find best replacement
                best_bi_in = -1
                best_gain = 0
                # We want a block that covers the subsets that bi_out covers, to maintain coverage
                # Actually, we just need to ensure that after removal, all subsets covered by bi_out
                # are still covered by others OR by the new block.
                # Since bi_out is removable, all its subsets are covered by >=2 blocks.
                # So we can just pick a block that adds the most NEW coverage? No, that increases size.
                # We want to pick a block that allows us to keep the size same or reduce it.
                # If we swap 1 for 1, size is same.
                # If we swap 1 for 0 (just remove), size reduces.
                
                # Try removing just bi_out first?
                # If we remove bi_out, is the covering still valid? Yes, by removable check.
                # So we can just remove it and prune?
                # The prune step in the main loop handles that.
                # Here, let's try swapping with a block that might allow further pruning.
                
                # Simple strategy: Try to find a block that covers the most subsets that are
                # currently covered ONLY by bi_out? No, there are none (removable).
                # So any swap maintains validity.
                # To improve, we want to find a block that makes other blocks redundant.
                
                # Heuristic: Pick a random block not in chosen? Or one with high overlap with chosen?
                # Let's try a few random candidates.
                candidates = rng.sample([b for b in range(n_blocks) if b not in chosen_set], min(10, n_blocks - len(chosen_set)))
                for bi_in in candidates:
                    # Perform swap
                    remove_block(bi_out)
                    chosen[i] = bi_in
                    add_block(bi_in)
                    
                    # Try to prune
                    pruned, _ = prune(chosen)
                    if len(pruned) < len(chosen):
                        # Accept
                        chosen = pruned
                        chosen_set = set(chosen)
                        # Recalculate counts
                        counts = [0] * n_subsets
                        for bi in chosen:
                            mask = block_masks[bi]
                            while mask:
                                low = mask & -mask
                                si = low.bit_length() - 1
                                counts[si] += 1
                                mask ^= low
                        improved = True
                        break
                    else:
                        # Revert
                        remove_block(bi_in)
                        chosen[i] = bi_out
                        add_block(bi_out)
                
                if improved:
                    break
            
        return chosen

    def remove_block(bi):
        mask = block_masks[bi]
        while mask:
            low = mask & -mask
            si = low.bit_length() - 1
            counts[si] -= 1
            mask ^= low
        chosen_set.discard(bi)

    def add_block(bi):
        mask = block_masks[bi]
        while mask:
            low = mask & -mask
            si = low.bit_length() - 1
            counts[si] += 1
            mask ^= low
        chosen_set.add(bi)

    # Main Genetic Algorithm Loop
    pop_size = 20
    population = []
    
    # Initialize population
    for i in range(pop_size):
        if time.time() > deadline - 10:
            break
        chosen, valid = greedy_construction(rng)
        if valid:
            pruned = prune(chosen)
            population.append(pruned)
            
    if not population:
        # Fallback
        chosen, _ = greedy_construction(rng)
        return [list(all_blocks[bi]) for bi in chosen]

    best_individual = min(population, key=len)
    best_size = len(best_individual)
    
    while time.time() < deadline - 5:
        new_population = []
        
        # Elitism: Keep best individual
        new_population.append(best_individual[:])
        
        # Generate offspring
        while len(new_population) < pop_size:
            if time.time() > deadline - 5:
                break
            
            # Selection: Tournament selection
            p1_idx = rng.randrange(len(population))
            p2_idx = rng.randrange(len(population))
            p1 = population[p1_idx]
            p2 = population[p2_idx]
            
            # Crossover
            child = crossover(p1, p2)
            
            # Mutation
            if rng.random() < 0.5:
                child = mutate(child, mutation_rate=0.1)
            
            # Local Search
            if time.time() < deadline - 5:
                child = local_search(child, time_budget=2)
            
            new_population.append(child)
            
        population = new_population
        
        # Update best
        current_best = min(population, key=len)
        if len(current_best) < best_size:
            best_size = len(current_best)
            best_individual = current_best[:]
            
    # Final prune
    best_individual = prune(best_individual)
    
    return [list(all_blocks[bi]) for bi in best_individual]
