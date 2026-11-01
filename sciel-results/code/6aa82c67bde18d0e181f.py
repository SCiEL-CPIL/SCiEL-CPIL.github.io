import itertools
import random
import time

# Global cache for precomputation to avoid recomputing on multiple calls if any
_PRECOMPUTED = {}

def _precompute(v, k, t):
    if (v, k, t) in _PRECOMPUTED:
        return _PRECOMPUTED[(v, k, t)]
        
    all_triples = list(itertools.combinations(range(v), t))
    num_triples = len(all_triples)
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    
    # Generate all k-blocks and their masks
    all_blocks = list(itertools.combinations(range(v), k))
    block_masks = [0] * len(all_blocks)
    blocks_by_triple = [[] for _ in range(num_triples)]
    
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        for sub in itertools.combinations(block, t):
            t_idx = triple_to_idx[sub]
            mask |= (1 << t_idx)
            blocks_by_triple[t_idx].append(b_idx)
        block_masks[b_idx] = mask
        
    _PRECOMPUTED[(v, k, t)] = {
        'all_blocks': all_blocks,
        'block_masks': block_masks,
        'blocks_by_triple': blocks_by_triple,
        'num_triples': num_triples,
        'triple_to_idx': triple_to_idx
    }
    return _PRECOMPUTED[(v, k, t)]

def construct(v, k, t):
    if v != 24 or k != 7 or t != 3:
        # Fallback for other parameters (not expected in this task)
        return _simple_greedy(v, k, t)

    start_time = time.time()
    deadline = start_time + 110.0
    
    data = _precompute(v, k, t)
    all_blocks = data['all_blocks']
    block_masks = data['block_masks']
    blocks_by_triple = data['blocks_by_triple']
    num_triples = data['num_triples']
    
    full_mask = (1 << num_triples) - 1
    
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple randomized greedy instances
    run_id = 0
    while time.time() < deadline:
        blocks = _randomized_greedy_bitset(v, k, t, data, random_seed=run_id)
        run_id += 1
        
        # Apply reduction
        blocks = _reduce_blocks(blocks, v, t, data)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
            if best_count <= 76:
                # We found a very good solution. Continue to see if we can beat it.
                pass

    if best_blocks is None:
        best_blocks = _simple_greedy(v, k, t)
        
    # Final reduction
    best_blocks = _reduce_blocks(best_blocks, v, t, data)
    
    return [list(b) for b in best_blocks]

def _randomized_greedy_bitset(v, k, t, data, random_seed):
    rng = random.Random(random_seed)
    all_blocks = data['all_blocks']
    block_masks = data['block_masks']
    blocks_by_triple = data['blocks_by_triple']
    num_triples = data['num_triples']
    
    uncovered_mask = (1 << num_triples) - 1
    blocks = []
    
    while uncovered_mask:
        # Pick a random uncovered triple
        # Find a set bit in uncovered_mask
        # We can pick a random index from 0..num_triples-1 and check if it's set,
        # or extract a random bit. Extracting a random bit from a large int is tricky.
        # Simpler: pick random triple index, if covered, retry? 
        # Better: iterate through bits? No, 2024 bits is small.
        # Let's just pick a random triple index and check if it's in uncovered_mask.
        # If not, pick another. Since coverage is dense, this might be slow if uncovered is small.
        # Alternative: maintain a list of uncovered triple indices? 
        # But updating list is O(N). 
        # Let's use the bitset to find *any* uncovered triple quickly?
        # `uncovered_mask & -uncovered_mask` gives the lowest set bit.
        # We can randomize by rotating? 
        # Let's just pick a random triple index and check.
        # If we are near the end, most triples are covered.
        # To ensure we don't loop forever, we can maintain a list of uncovered indices?
        # Let's stick to bitset check.
        
        while True:
            t_idx = rng.randrange(num_triples)
            if (uncovered_mask >> t_idx) & 1:
                break
        
        candidates = blocks_by_triple[t_idx]
        best_block_idx = None
        best_gain = -1
        
        # Evaluate candidates
        for b_idx in candidates:
            mask = block_masks[b_idx]
            gain = (mask & uncovered_mask).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_block_idx = b_idx
            elif gain == best_gain and rng.random() < 0.5:
                best_block_idx = b_idx
                
        blocks.append(all_blocks[best_block_idx])
        uncovered_mask &= ~block_masks[best_block_idx]
        
    return blocks

def _reduce_blocks(blocks, v, t, data):
    if not blocks:
        return []
        
    all_blocks = data['all_blocks']
    block_masks = data['block_masks']
    blocks_by_triple = data['blocks_by_triple']
    triple_to_idx = data['triple_to_idx']
    num_triples = data['num_triples']
    
    # Map block tuples to indices in all_blocks for quick lookup?
    # Or just keep the list of block tuples.
    # We need to map the current blocks (tuples) to their index in all_blocks to get masks.
    # This is O(N^2) if we do linear search. 
    # Let's create a dict: block_tuple -> block_idx
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    
    # Convert current blocks to indices in all_blocks
    current_block_indices = [block_to_idx[b] for b in blocks]
    
    # Active flags
    active = [True] * len(current_block_indices)
    
    # Helper to get mask of active blocks
    def get_active_mask():
        m = 0
        for i, idx in enumerate(current_block_indices):
            if active[i]:
                m |= block_masks[idx]
        return m
        
    # Helper to check if a triple is covered by an active block other than 'exclude_idx'
    # We can just use the mask of all active blocks excluding the one we're testing?
    # No, we need to know if a specific triple is covered.
    
    # Iterative reduction and swapping
    # We will loop until no changes are made
    changed = True
    while changed:
        changed = False
        
        # Phase 1: Remove redundant blocks
        # A block is redundant if all its triples are covered by other active blocks.
        # We can compute the union of masks of all other active blocks.
        # This is O(N) per block, so O(N^2) total. N is ~80, so 6400 ops. Fast.
        
        # To optimize, we can maintain a coverage count array?
        # But swapping changes which blocks are active.
        # Let's stick to the mask approach for simplicity and correctness, it's fast enough for N~100.
        
        for i in range(len(current_block_indices)):
            if not active[i]:
                continue
            
            b_idx = current_block_indices[i]
            b_mask = block_masks[b_idx]
            
            # Check if b_mask is covered by the union of other active blocks
            # Union of others = (Total Active Mask) ^ (b_mask & Total Active Mask) ? No.
            # Union of others = (Total Active Mask) & ~b_mask ? No, that removes bits in b_mask.
            # We want to know if b_mask is a subset of (Union of Other Active Masks).
            
            # Calculate Union of Other Active Masks
            others_mask = 0
            for j in range(len(current_block_indices)):
                if i == j or not active[j]:
                    continue
                others_mask |= block_masks[current_block_indices[j]]
                
            if (b_mask & ~others_mask) == 0:
                # Redundant
                active[i] = False
                changed = True
                # We can break or continue. Continuing allows removing multiple in one pass.
                # But if we remove i, others_mask changes for subsequent i.
                # So we must recompute or be careful.
                # Since we are in a loop that restarts on change, we can just mark it and continue checking others
                # against the *original* set? No, that's unsafe.
                # If we remove i, block j might become redundant.
                # But if we don't remove i, block j might not be redundant.
                # So we should probably just restart the loop or be conservative.
                # Given N is small, restarting is fine.
                break
        
        if not active.count(True) == len(current_block_indices):
            # If we removed something, restart to catch new redundancies
            continue
            
        # Phase 2: 1-for-1 Swaps
        # For each active block i, check if we can swap it with a new block B_new
        # such that B_new covers all triples that i uniquely covers.
        
        for i in range(len(current_block_indices)):
            if not active[i]:
                continue
            
            b_idx = current_block_indices[i]
            b_mask = block_masks[b_idx]
            
            # Calculate unique mask for i
            # Unique bits are those in b_mask that are not in (Union of Other Active Masks)
            others_mask = 0
            for j in range(len(current_block_indices)):
                if i == j or not active[j]:
                    continue
                others_mask |= block_masks[current_block_indices[j]]
                
            unique_mask = b_mask & ~others_mask
            
            if unique_mask == 0:
                # Should have been caught in Phase 1
                active[i] = False
                changed = True
                break
                
            # Find a candidate block B_new that covers unique_mask
            # B_new must cover all triples in unique_mask.
            # Pick a triple t from unique_mask.
            # Get all blocks covering t.
            # Check if they cover the rest of unique_mask.
            
            # Extract a set bit from unique_mask
            # unique_mask is an int.
            # Find index of a set bit.
            # Since unique_mask is non-zero, we can find the lowest set bit.
            # Or just iterate through bits?
            # Let's get the first set bit index.
            low_bit = unique_mask & -unique_mask
            t_idx = low_bit.bit_length() - 1
            
            # Candidates are blocks covering t_idx
            candidates = blocks_by_triple[t_idx]
            
            found_swap = False
            for cand_idx in candidates:
                if cand_idx == b_idx:
                    continue
                    
                cand_mask = block_masks[cand_idx]
                
                # Check if cand_mask covers unique_mask
                if (cand_mask & unique_mask) == unique_mask:
                    # Valid swap
                    # Check if cand_idx is already in active blocks?
                    # If so, we are just replacing i with an existing block?
                    # If cand_idx is already active, we can't have two identical blocks.
                    # But the task requires distinct blocks.
                    # So we must ensure cand_idx is not in the current active set.
                    
                    is_in_set = False
                    for j in range(len(current_block_indices)):
                        if active[j] and current_block_indices[j] == cand_idx:
                            is_in_set = True
                            break
                            
                    if not is_in_set:
                        # Perform swap
                        current_block_indices[i] = cand_idx
                        changed = True
                        found_swap = True
                        break
                        
            if found_swap:
                break # Restart loop to update masks
                
    # Construct result
    result_indices = [current_block_indices[i] for i in range(len(current_block_indices)) if active[i]]
    return [all_blocks[i] for i in result_indices]

def _simple_greedy(v, k, t):
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        target = min(uncovered)
        best_block = None
        best_gain = -1
        remaining_points = [p for p in range(v) if p not in target]
        need = k - t
        for combo in itertools.combinations(remaining_points, need):
            block = tuple(sorted(target + combo))
            gain = 0
            for sub in itertools.combinations(block, t):
                if sub in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = block
        blocks.append(best_block)
        for sub in itertools.combinations(best_block, t):
            uncovered.discard(sub)
    return blocks
