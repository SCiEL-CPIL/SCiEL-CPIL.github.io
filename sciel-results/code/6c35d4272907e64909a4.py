import itertools
import random
import time
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Precompute all t-subsets (triples for t=3)
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    subset_index = {s: i for i, s in enumerate(all_subsets)}
    
    # Precompute all possible k-blocks and their covered subset indices
    # For v=24, k=7, this is C(24,7) = 346,104 blocks.
    # Each block covers C(7,3) = 35 subsets.
    
    block_list = []
    block_masks = []
    
    # Optimization: Precompute bit for each triple
    triple_bit = {sub: 1 << i for i, sub in enumerate(all_subsets)}
    
    # Generate all blocks
    # This is the most expensive part upfront, but necessary for global search
    for block in itertools.combinations(range(v), k):
        mask = 0
        for sub in itertools.combinations(block, t):
            mask |= triple_bit[sub]
        block_list.append(block)
        block_masks.append(mask)
        
    n_blocks = len(block_list)
    
    # Helper to check if a set of block indices forms a valid cover
    def is_valid_cover(indices):
        union_mask = 0
        for idx in indices:
            union_mask |= block_masks[idx]
        return union_mask == (1 << n_subsets) - 1

    # Helper to get coverage counts for a set of blocks
    def get_coverage_counts(indices):
        counts = [0] * n_subsets
        for idx in indices:
            block = block_list[idx]
            for sub in itertools.combinations(block, t):
                counts[subset_index[sub]] += 1
        return counts

    # Initial Solution: Greedy Set Cover
    # We need a good starting point.
    # Standard greedy: pick block with max gain.
    
    def greedy_cover():
        uncovered_mask = (1 << n_subsets) - 1
        selected_indices = []
        used_indices = set()
        
        while uncovered_mask:
            best_gain = 0
            best_candidates = []
            
            # Iterate through all blocks
            for i in range(n_blocks):
                if i in used_indices:
                    continue
                gain = (uncovered_mask & block_masks[i]).bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [i]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(i)
                    
            if not best_candidates:
                break
                
            # Pick randomly from best to add diversity if needed, or just first
            # For initial greedy, deterministic is fine, but let's add slight randomness
            # to avoid getting stuck in a very specific local structure?
            # Actually, for SA start, a deterministic greedy is a good baseline.
            chosen_idx = best_candidates[0]
            
            selected_indices.append(chosen_idx)
            used_indices.add(chosen_idx)
            uncovered_mask &= ~block_masks[chosen_idx]
            
        return selected_indices

    # Run initial greedy
    current_indices = greedy_cover()
    
    # Post-process initial greedy: Remove redundant blocks
    def remove_redundant(indices):
        if not indices:
            return indices
        
        # Calculate coverage counts
        counts = get_coverage_counts(indices)
        
        # Check for redundancy
        # A block is redundant if all its triples are covered by at least one other block.
        # i.e., for all triples in block, count > 1.
        
        new_indices = []
        # We do a single pass. If we remove a block, counts for its triples decrease.
        # This might make other blocks redundant.
        # To be safe and fast, we can do multiple passes until stable.
        
        changed = True
        current_set = set(indices)
        
        while changed:
            changed = False
            next_set = set()
            for idx in current_set:
                block = block_list[idx]
                is_essential = False
                for sub in itertools.combinations(block, t):
                    if counts[subset_index[sub]] == 1:
                        is_essential = True
                        break
                
                if is_essential:
                    next_set.add(idx)
                else:
                    # Remove it
                    changed = True
                    for sub in itertools.combinations(block, t):
                        counts[subset_index[sub]] -= 1
            
            current_set = next_set
            
        return list(current_set)

    current_indices = remove_redundant(current_indices)
    
    best_indices = current_indices[:]
    best_score = len(current_indices)
    
    rng = random.Random(42)
    
    # Simulated Annealing
    # Objective: Minimize number of blocks.
    # Moves:
    # 1. Remove a redundant block (if any exist).
    # 2. Swap a block: Remove one, add another.
    #    - If we remove a non-redundant block, we must add one that covers all lost triples.
    #    - Or, we can allow invalid states and penalize them?
    #    - Standard SA for set cover often uses "repair" or accepts invalid states with penalty.
    #    - Let's stick to valid states for simplicity and robustness, or use a small penalty.
    #    - Given the time limit, strict validity is safer.
    
    # Let's use a strategy that allows invalid states but heavily penalizes them,
    # or restricts moves to valid ones.
    # Restricting to valid moves is harder.
    # Let's try:
    # Move 1: Remove block i if it is redundant.
    # Move 2: Swap block i with block j.
    #   - Remove i.
    #   - Check if coverage is still valid. If yes, accept.
    #   - If no, check if j covers all triples that were ONLY covered by i.
    #   - If yes, swap is valid.
    
    # To make this fast, we maintain `counts` and `current_indices`.
    
    counts = get_coverage_counts(current_indices)
    current_set = set(current_indices)
    
    temperature = 100.0
    cooling_rate = 0.995
    
    # Track best
    global_best_indices = current_indices[:]
    global_best_score = best_score
    
    # Precompute for each block, which other blocks can "replace" it?
    # This is complex. Let's just sample.
    
    # Number of iterations
    # We have ~120s left.
    # Each move evaluation should be fast.
    
    while time.time() < deadline - 5.0:
        # Generate a move
        if rng.random() < 0.1 and len(current_indices) > 0:
            # Try to remove a redundant block
            # Find a redundant block
            redundant_block = None
            for idx in current_indices:
                block = block_list[idx]
                is_redundant = True
                for sub in itertools.combinations(block, t):
                    if counts[subset_index[sub]] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    redundant_block = idx
                    break
            
            if redundant_block is not None:
                # Remove it
                block = block_list[redundant_block]
                for sub in itertools.combinations(block, t):
                    counts[subset_index[sub]] -= 1
                
                current_indices.remove(redundant_block)
                current_set.remove(redundant_block)
                
                new_score = len(current_indices)
                if new_score < global_best_score:
                    global_best_score = new_score
                    global_best_indices = current_indices[:]
                    
                # Always accept valid improvement
                # No temperature needed for strict improvement?
                # SA usually accepts worse states.
                # Since we only generate valid moves here (removal of redundant),
                # we always accept.
                continue
        
        # Try Swap
        if len(current_indices) == 0:
            break
            
        # Pick a block to remove
        i_idx = rng.choice(current_indices)
        i_block = block_list[i_idx]
        
        # Identify triples that would become uncovered if we remove i_block
        # i.e., triples where counts == 1
        critical_triples = []
        for sub in itertools.combinations(i_block, t):
            if counts[subset_index[sub]] == 1:
                critical_triples.append(sub)
        
        # If no critical triples, we can just remove it (handled above? No, above was random check)
        # If critical_triples is empty, it's a redundant block.
        if not critical_triples:
             # This is effectively a removal move
             for sub in itertools.combinations(i_block, t):
                 counts[subset_index[sub]] -= 1
             current_indices.remove(i_idx)
             current_set.remove(i_idx)
             
             new_score = len(current_indices)
             if new_score < global_best_score:
                 global_best_score = new_score
                 global_best_indices = current_indices[:]
             continue
             
        # We need to find a block j that covers ALL critical_triples
        # And j is not in current_set
        
        # To speed up, we can precompute for each triple, which blocks cover it?
        # That's a huge structure.
        
        # Alternative: Sample random blocks and check.
        # Since we need to cover a small number of critical triples (usually 1-5),
        # the probability that a random block covers them is non-trivial.
        
        # Let's sample K random blocks
        found_swap = False
        for _ in range(50): # 50 samples
            j_idx = rng.randint(0, n_blocks - 1)
            if j_idx in current_set:
                continue
                
            j_block = block_list[j_idx]
            
            # Check if j_block covers all critical_triples
            covers_all = True
            # Convert critical_triples to a set of indices for faster lookup?
            # Or just check membership in j_block?
            # critical_triples is a list of tuples of 3 ints.
            # j_block is a tuple of 7 ints.
            # We can check if all elements of sub are in j_block.
            
            for sub in critical_triples:
                if not all(x in j_block for x in sub):
                    covers_all = False
                    break
            
            if covers_all:
                # Valid swap!
                # Execute swap
                # Remove i_idx
                for sub in itertools.combinations(i_block, t):
                    counts[subset_index[sub]] -= 1
                current_indices.remove(i_idx)
                current_set.remove(i_idx)
                
                # Add j_idx
                for sub in itertools.combinations(j_block, t):
                    counts[subset_index[sub]] += 1
                current_indices.append(j_idx)
                current_set.add(j_idx)
                
                found_swap = True
                break
        
        if not found_swap:
            # Try a "perturbation" move?
            # Or just accept a worse state?
            # Since we only generate valid moves, we can't get worse.
            # This is basically a local search that only improves or stays same.
            # This might get stuck.
            # To escape, we need to allow invalid states or larger jumps.
            
            # Let's try a "2-swap" or "add" move?
            # Adding a block increases score, so only useful if it enables future removals.
            # This is complex.
            
            # Let's try to restart if stuck?
            # Or increase temperature and allow "random valid swaps" even if they don't improve?
            # Since all valid swaps keep size same, they are neutral.
            # We should accept neutral moves to explore.
            
            # Let's modify the logic:
            # If we find a valid swap, we accept it with probability 1 (or temp-based).
            # Since size doesn't change, it's neutral.
            # We should accept neutral moves to walk the landscape.
            
            # Let's re-evaluate:
            # My current loop only accepts improvements (removals) or valid swaps (neutral).
            # If I find a valid swap, I should accept it to move in the landscape.
            # The code above executes the swap and continues.
            # So it does accept neutral moves.
            
            # What if no swap is found?
            # We stay put.
            
            pass
        
        # Cooling
        temperature *= cooling_rate
        
        # Optional: If we are stuck for a long time, restart?
        # For now, just keep going.

    # Final cleanup
    # Ensure global_best_indices is valid and minimal
    final_indices = remove_redundant(global_best_indices)
    
    # Verify
    if not is_valid_cover(final_indices):
        # Fallback to initial greedy if something went wrong
        final_indices = remove_redundant(greedy_cover())
        
    return [list(block_list[i]) for i in final_indices]

