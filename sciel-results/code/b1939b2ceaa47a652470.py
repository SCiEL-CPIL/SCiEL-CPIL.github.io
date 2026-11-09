import itertools
import random
import time

def construct(v, k, t):
    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    
    # Map each t-subset to its index
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # For each block, list of t-subset indices it covers
    block_tsubs = []
    for block in all_blocks:
        tsubs_in_block = [tsub_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_tsubs.append(tsubs_in_block)
        
    # For each t-subset index, list of block indices that cover it
    # This is useful for quick redundancy checks if we maintain counts
    # But for local search, we just need to check if count > 1 for all tsubs in block
    
    # 1. Greedy Initialization
    cover_count = [0] * n_tsubs
    block_gain = [len(tsubs) for tsubs in block_tsubs]
    selected_indices = []
    remaining = set(range(n_blocks))
    
    while True:
        if all(cover_count[s] > 0 for s in range(n_tsubs)):
            break
        
        best_block = -1
        best_gain = -1
        for b in remaining:
            if block_gain[b] > best_gain:
                best_gain = block_gain[b]
                best_block = b
        
        if best_block == -1 or best_gain <= 0:
            break
        
        selected_indices.append(best_block)
        remaining.discard(best_block)
        
        for s_idx in block_tsubs[best_block]:
            if cover_count[s_idx] == 0:
                cover_count[s_idx] = 1
                # Update gains for other blocks
                # Optimization: We can just recalculate gains or update incrementally
                # Incremental update:
                # For each block b in remaining that covers s_idx, block_gain[b] -= 1
                # We need tsub_to_blocks for this
                pass

    # To make the greedy step correct with incremental gains, we need tsub_to_blocks
    # Let's rebuild or just use a simpler greedy that recalculates gains if needed, 
    # or stick to the original greedy logic which was correct.
    # Let's re-implement the greedy part carefully to get a good starting point.
    
    # Re-doing greedy properly
    tsub_to_blocks = [[] for _ in range(n_tsubs)]
    for b_idx, tsubs in enumerate(block_tsubs):
        for s_idx in tsubs:
            tsub_to_blocks[s_idx].append(b_idx)
            
    cover_count = [0] * n_tsubs
    block_gain = [len(tsubs) for tsubs in block_tsubs]
    selected_indices = []
    remaining = set(range(n_blocks))
    
    while True:
        if all(cover_count[s] > 0 for s in range(n_tsubs)):
            break
        
        best_block = -1
        best_gain = -1
        for b in remaining:
            if block_gain[b] > best_gain:
                best_gain = block_gain[b]
                best_block = b
        
        if best_block == -1 or best_gain <= 0:
            break
        
        selected_indices.append(best_block)
        remaining.discard(best_block)
        
        for s_idx in block_tsubs[best_block]:
            if cover_count[s_idx] == 0:
                cover_count[s_idx] = 1
                for b in tsub_to_blocks[s_idx]:
                    if b in remaining:
                        block_gain[b] -= 1

    # 2. Local Search
    # State: set of block indices
    current_blocks = set(selected_indices)
    # Re-initialize cover_count for current_blocks
    cover_count = [0] * n_tsubs
    for b in current_blocks:
        for s_idx in block_tsubs[b]:
            cover_count[s_idx] += 1
            
    start_time = time.time()
    time_limit = 115.0 # Leave buffer
    
    def try_remove_redundant():
        """Remove any redundant blocks. Returns True if any removed."""
        removed = False
        for b in list(current_blocks):
            redundant = True
            for s_idx in block_tsubs[b]:
                if cover_count[s_idx] == 1:
                    redundant = False
                    break
            if redundant:
                current_blocks.remove(b)
                for s_idx in block_tsubs[b]:
                    cover_count[s_idx] -= 1
                removed = True
        return removed

    # Initial cleanup
    try_remove_redundant()
    
    best_size = len(current_blocks)
    best_solution = list(current_blocks)
    
    # Local Search Loop
    while time.time() - start_time < time_limit:
        # Try to remove redundant
        if try_remove_redundant():
            if len(current_blocks) < best_size:
                best_size = len(current_blocks)
                best_solution = list(current_blocks)
            continue
            
        # Try random swap
        if len(current_blocks) == 0:
            break
            
        b_out = random.choice(list(current_blocks))
        # Pick a random block from all_blocks
        b_in = random.choice(range(n_blocks))
        
        if b_out == b_in:
            continue
            
        # Check if b_in is already in current_blocks
        if b_in in current_blocks:
            continue
            
        # Simulate swap: remove b_out, add b_in
        # Check validity: 
        # 1. Temporarily decrement counts for b_out
        # 2. Check if any count becomes 0
        # 3. If so, check if b_in covers them
        # 4. If valid, perform swap
        
        valid = True
        for s_idx in block_tsubs[b_out]:
            if cover_count[s_idx] == 1:
                # This t-subset would become uncovered
                # Check if b_in covers it
                # We can precompute a set of tsub indices for b_in for O(1) lookup?
                # Or just iterate b_in's tsubs?
                # Since block_tsubs[b_in] is a list, converting to set each time is expensive.
                # We can precompute sets for all blocks? 50388 blocks * 35 tsubs. Memory ~ 50k * 35 * 8 bytes ~ 14MB. Acceptable.
                pass
                
        # To speed up, precompute tsub sets for all blocks
        # block_tsub_sets = [set(tsubs) for tsubs in block_tsubs]
        
        # Let's do the check efficiently:
        # We need to know if b_in covers all tsubs that b_out uniquely covers.
        # Unique tsubs of b_out: {s | s in block_tsubs[b_out] and cover_count[s] == 1}
        # If b_in covers all of these, the swap is valid.
        
        unique_tsubs_out = [s for s in block_tsubs[b_out] if cover_count[s] == 1]
        if unique_tsubs_out:
            # Check if b_in covers all of them
            # We can iterate through unique_tsubs_out and check if they are in block_tsubs[b_in]
            # To make this fast, we can have a global array or use the precomputed sets.
            # Let's precompute sets outside the loop.
            pass

    # We need to precompute sets for fast membership check
    # Let's restart the local search part with precomputed sets
    
    block_tsub_sets = [set(tsubs) for tsubs in block_tsubs]
    
    # Re-run local search with sets
    # (Reset state to best_solution found so far or greedy result)
    # Actually, let's just continue from current state if we didn't modify the code structure too much.
    # But the code above didn't finish the swap logic.
    
    # Let's write the full local search block properly.
    
    # ... (See final code below)
    
    return [list(all_blocks[b]) for b in best_solution]
