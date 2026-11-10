import itertools
import random
import time
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 150s limit
    
    # 1. Precomputation
    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    all_triples = list(itertools.combinations(points, t))
    n_triples = len(all_triples)
    triple_index = {trip: i for i, trip in enumerate(all_triples)}
    
    # Map each block to the list of triple indices it covers
    block_triples = []
    for block in all_blocks:
        covered = []
        for trip in itertools.combinations(block, t):
            covered.append(triple_index[trip])
        block_triples.append(covered)
        
    # Precompute bitmask for each block for fast intersection and counting
    # n_triples is ~2024, fits in Python int.
    block_masks = [0] * n_blocks
    for i, trips in enumerate(block_triples):
        m = 0
        for t_idx in trips:
            m |= (1 << t_idx)
        block_masks[i] = m
    
    # Precompute block overlap scores: for each pair of blocks, how many triples they share
    # This is too expensive for all pairs (346k^2), so we'll compute on the fly or use a different strategy
    # Instead, precompute for each block the count of each triple it covers (always 1, so not needed)

    # 2. Greedy Initialization
    def greedy_init(rng):
        uncovered_mask = (1 << n_triples) - 1
        selected = []
        
        while uncovered_mask:
            best_gain = -1
            best_block = -1
            
            # Randomized greedy: scan all blocks, pick best gain.
            # To add diversity, we can randomize the order or tie-breaking.
            # Here we use a simple scan with random tie-breaking.
            # Optimization: If we find a block covering all remaining uncovered, we can stop?
            # No, we need to cover ALL. But if gain == bit_count(uncovered_mask), we are done.
            
            # Iterate through blocks. 
            # To make it faster, we can skip blocks that don't intersect uncovered_mask?
            # block_masks[i] & uncovered_mask
            # If 0, gain is 0.
            
            # We want to maximize gain.
            # We can just iterate all.
            # 346k blocks.
            # In Python, this loop might take ~0.1-0.2s per iteration if optimized?
            # 70 iterations * 0.2s = 14s. Acceptable.
            
            # To speed up, we can use a list of indices and shuffle? No, we need max.
            # We can use a priority queue? 
            # But gains change as uncovered_mask changes.
            
            # Let's just scan.
            # Optimization: pre-filter blocks that are "promising"? Hard to do dynamically.
            
            # We can break early if we find a block that covers everything?
            # Max possible gain is uncovered_mask.bit_count().
            max_possible = uncovered_mask.bit_count()
            
            for i in range(n_blocks):
                inter = block_masks[i] & uncovered_mask
                if inter:
                    gain = inter.bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_block = i
                        if best_gain == max_possible:
                            break # Can't do better
                    elif gain == best_gain and gain > 0:
                        # Random tie-breaking
                        if rng.random() < 0.1:
                            best_block = i
                            # Don't break, keep looking for strictly better? 
                            # Or just accept this one? 
                            # Standard greedy picks first max. 
                            # Randomized greedy might pick a different max.
                            # Let's just keep the first max found to be deterministic-ish 
                            # but rely on initial shuffle? No, we scan 0..N.
                            # Let's just pick the first one with max gain to be fast.
                            # The randomness comes from multiple restarts or SA.
                            pass
            
            if best_block == -1:
                break # Should not happen
                
            selected.append(best_block)
            uncovered_mask &= ~block_masks[best_block]
            
        return selected

    # Run a few greedy restarts to find a good initial solution
    best_initial = None
    best_initial_len = float('inf')
    
    for _ in range(10):
        if time.time() > deadline - 30:
            break
        rng = random.Random()
        sel = greedy_init(rng)
        if len(sel) < best_initial_len:
            best_initial_len = len(sel)
            best_initial = sel
            
    if best_initial is None:
        # Fallback: single greedy
        best_initial = greedy_init(random.Random())
        
    # 3. Greedy Reduction: Try to remove redundant blocks
    def greedy_reduce(block_set):
        """Try to remove blocks that are redundant (all their triples are covered by other blocks)."""
        if not block_set:
            return set()
        
        current_set = set(block_set)
        cover_count = [0] * n_triples
        for idx in current_set:
            for t_idx in block_triples[idx]:
                cover_count[t_idx] += 1
        
        changed = True
        while changed:
            changed = False
            for idx in list(current_set):
                # Check if all triples in this block are covered by other blocks
                redundant = True
                for t_idx in block_triples[idx]:
                    if cover_count[t_idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    # Remove this block
                    for t_idx in block_triples[idx]:
                        cover_count[t_idx] -= 1
                    current_set.remove(idx)
                    changed = True
        
        return current_set
    
    # Apply greedy reduction to initial solution
    best_initial = greedy_reduce(set(best_initial))
    best_initial_len = len(best_initial)
    
    # 4. Simulated Annealing / Local Search
    # Objective: Minimize (number_of_blocks + penalty * uncovered)
    # We want to minimize block count.
    
    current_set = set(best_initial)
    
    # Coverage counts
    cover_count = [0] * n_triples
    for idx in current_set:
        for t_idx in block_triples[idx]:
            cover_count[t_idx] += 1
            
    uncovered_count = sum(1 for c in cover_count if c == 0)
    
    # Cost function:
    # If valid: cost = len(current_set)
    # If invalid: cost = len(current_set) + 1000 * uncovered_count
    # We want to drive uncovered_count to 0 first, then minimize size.
    
    def calc_cost():
        if uncovered_count == 0:
            return len(current_set)
        else:
            return len(current_set) + 1000 * uncovered_count

    current_cost = calc_cost()
    best_set = set(current_set)
    best_cost = current_cost
    
    # SA Parameters
    T = 5.0
    cooling_rate = 0.9995
    min_T = 0.1
    
    all_indices = list(range(n_blocks))
    
    # Precompute for each block, the list of blocks that share at least one triple with it
    # This is expensive, so we'll use a different strategy: for swap moves, 
    # we'll sample candidate blocks that are likely to be good replacements
    
    while time.time() < deadline - 5:
        if T < min_T:
            # Restart with higher T to escape local minima
            T = 5.0
            
        # Choose a move
        move_type = random.random()
        
        delta_cost = 0
        old_block = -1
        new_block = -1
        valid_move = True
        
        if move_type < 0.3:
            # Add a block that covers many uncovered triples
            if uncovered_count == 0:
                # If all covered, just add a random block (will likely be rejected)
                new_block = random.choice(all_indices)
                if new_block in current_set:
                    valid_move = False
                else:
                    for t_idx in block_triples[new_block]:
                        if cover_count[t_idx] == 0:
                            uncovered_count -= 1
                            delta_cost -= 1000
                        cover_count[t_idx] += 1
                    delta_cost += 1
                    current_set.add(new_block)
            else:
                # Find blocks that cover uncovered triples
                # Sample a few random blocks and pick the one with best gain
                best_gain = -1
                best_block = -1
                for _ in range(20):
                    candidate = random.choice(all_indices)
                    if candidate in current_set:
                        continue
                    gain = 0
                    for t_idx in block_triples[candidate]:
                        if cover_count[t_idx] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_block = candidate
                if best_block == -1 or best_gain == 0:
                    valid_move = False
                else:
                    new_block = best_block
                    for t_idx in block_triples[new_block]:
                        if cover_count[t_idx] == 0:
                            uncovered_count -= 1
                            delta_cost -= 1000
                        cover_count[t_idx] += 1
                    delta_cost += 1
                    current_set.add(new_block)
                
        elif move_type < 0.6:
            # Remove a block (only if it doesn't create uncovered triples)
            if not current_set:
                valid_move = False
            else:
                # Try to remove a block that is "redundant" (all its triples have count > 1)
                # Sample a few blocks and pick one that is safe to remove
                best_candidate = -1
                for _ in range(10):
                    candidate = random.choice(list(current_set))
                    safe = True
                    for t_idx in block_triples[candidate]:
                        if cover_count[t_idx] <= 1:
                            safe = False
                            break
                    if safe:
                        best_candidate = candidate
                        break
                if best_candidate == -1:
                    valid_move = False
                else:
                    old_block = best_candidate
                    for t_idx in block_triples[old_block]:
                        cover_count[t_idx] -= 1
                        if cover_count[t_idx] == 0:
                            uncovered_count += 1
                            delta_cost += 1000
                    delta_cost -= 1
                    current_set.remove(old_block)
                    
        else:
            # Swap: replace one block with another
            if not current_set:
                valid_move = False
            else:
                # Pick a block to remove
                old_block = random.choice(list(current_set))
                
                # Find a good replacement: a block not in current_set that covers
                # many of the same triples as old_block, especially uncovered ones
                # after removing old_block
                
                # First, compute what would be uncovered if we remove old_block
                would_uncover = []
                for t_idx in block_triples[old_block]:
                    if cover_count[t_idx] <= 1:
                        would_uncover.append(t_idx)
                
                if not would_uncover:
                    # Safe to remove, find a block that covers some of old_block's triples
                    # to maintain coverage quality
                    best_gain = -1
                    best_block = -1
                    for _ in range(30):
                        candidate = random.choice(all_indices)
                        if candidate in current_set:
                            continue
                        gain = 0
                        for t_idx in block_triples[candidate]:
                            if cover_count[t_idx] > 0:  # Already covered, but we're removing old_block
                                # Check if this triple is still covered after removing old_block
                                if cover_count[t_idx] > 1 or t_idx not in block_triples[old_block]:
                                    gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_block = candidate
                    if best_block == -1:
                        valid_move = False
                    else:
                        new_block = best_block
                        # Remove old
                        for t_idx in block_triples[old_block]:
                            cover_count[t_idx] -= 1
                            if cover_count[t_idx] == 0:
                                uncovered_count += 1
                                delta_cost += 1000
                        # Add new
                        for t_idx in block_triples[new_block]:
                            if cover_count[t_idx] == 0:
                                uncovered_count -= 1
                                delta_cost -= 1000
                            cover_count[t_idx] += 1
                        delta_cost += 0
                        current_set.remove(old_block)
                        current_set.add(new_block)
                else:
                    # Must cover the would_uncover triples
                    best_gain = -1
                    best_block = -1
                    for _ in range(30):
                        candidate = random.choice(all_indices)
                        if candidate in current_set:
                            continue
                        gain = 0
                        for t_idx in block_triples[candidate]:
                            if cover_count[t_idx] == 0:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_block = candidate
                    if best_block == -1 or best_gain == 0:
                        valid_move = False
                    else:
                        new_block = best_block
                        # Remove old
                        for t_idx in block_triples[old_block]:
                            cover_count[t_idx] -= 1
                            if cover_count[t_idx] == 0:
                                uncovered_count += 1
                                delta_cost += 1000
                        # Add new
                        for t_idx in block_triples[new_block]:
                            if cover_count[t_idx] == 0:
                                uncovered_count -= 1
                                delta_cost -= 1000
                            cover_count[t_idx] += 1
                        delta_cost += 0
                        current_set.remove(old_block)
                        current_set.add(new_block)

        if not valid_move:
            continue
            
        new_cost = current_cost + delta_cost
        
        # Metropolis Acceptance
        if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_set = set(current_set)
        else:
            # Reject: Undo
            if move_type < 0.3:
                # Undo Add
                current_set.remove(new_block)
                for t_idx in block_triples[new_block]:
                    cover_count[t_idx] -= 1
                    if cover_count[t_idx] == 0:
                        uncovered_count += 1
            elif move_type < 0.6:
                # Undo Remove
                current_set.add(old_block)
                for t_idx in block_triples[old_block]:
                    cover_count[t_idx] += 1
                    if cover_count[t_idx] == 0:
                        uncovered_count -= 1
            else:
                # Undo Swap
                current_set.remove(new_block)
                current_set.add(old_block)
                # Re-add old coverage
                for t_idx in block_triples[old_block]:
                    cover_count[t_idx] += 1
                    if cover_count[t_idx] == 0:
                        uncovered_count -= 1
                # Re-remove new coverage
                for t_idx in block_triples[new_block]:
                    cover_count[t_idx] -= 1
                    if cover_count[t_idx] == 0:
                        uncovered_count += 1

        T *= cooling_rate

    # 5. Final Polish: Apply greedy reduction to best_set
    best_set = greedy_reduce(best_set)
    
    # Verify best_set
    final_cover = [0] * n_triples
    for idx in best_set:
        for t_idx in block_triples[idx]:
            final_cover[t_idx] += 1
            
    if any(c == 0 for c in final_cover):
        # Fallback to current_set if it's valid, or greedy
        current_cover = [0] * n_triples
        for idx in current_set:
            for t_idx in block_triples[idx]:
                current_cover[t_idx] += 1
        if not any(c == 0 for c in current_cover):
            best_set = current_set
        else:
            # Return greedy
            best_set = set(best_initial)

    final_blocks = [list(all_blocks[i]) for i in best_set]
    return final_blocks
