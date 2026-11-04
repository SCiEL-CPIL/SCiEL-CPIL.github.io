import itertools
import random
import time

def _remove_redundant_blocks(v, k, t, blocks):
    """
    Remove blocks that do not uniquely cover any t-subset.
    Returns a new list of blocks.
    """
    if not blocks:
        return []
    
    # Map each t-subset to the list of block indices that cover it
    # To be efficient, we can just check for each block if all its t-subsets are covered by others.
    
    # First, create a set of all blocks for quick lookup? No, we need indices.
    # Let's count coverage.
    # num_blocks = len(blocks)
    # covered_by = {tsub: set() for tsub in all_tsubs} # Too much memory? 3876 * small set. OK.
    
    # Better: 
    # For each block i, calculate unique_count.
    # unique_count[i] = number of t-subsets in block i that are NOT in any other block j.
    
    # To do this efficiently:
    # 1. Create a dictionary: tsub -> count of blocks covering it.
    # 2. Initialize count to 0.
    # 3. For each block, increment count for its t-subsets.
    # 4. For each block, if all its t-subsets have count > 1, it is redundant? 
    #    No, if count > 1, it means at least one OTHER block covers it. 
    #    So if for ALL t-subsets in block i, count > 1, then block i is redundant.
    
    all_tsubs_in_blocks = {}
    for b in blocks:
        for tsub in itertools.combinations(b, t):
            all_tsubs_in_blocks.setdefault(tuple(tsub), 0)
            all_tsubs_in_blocks[tuple(tsub)] += 1
            
    redundant_indices = []
    for i, b in enumerate(blocks):
        is_redundant = True
        for tsub in itertools.combinations(b, t):
            if all_tsubs_in_blocks.get(tuple(tsub), 0) <= 1:
                is_redundant = False
                break
        if is_redundant:
            redundant_indices.append(i)
            
    if not redundant_indices:
        return blocks
        
    # Remove redundant blocks
    # Note: Removing a block might make another block redundant.
    # We should iterate this process.
    
    new_blocks = [b for i, b in enumerate(blocks) if i not in redundant_indices]
    
    # Repeat until stable
    while len(new_blocks) < len(blocks):
        blocks = new_blocks
        # Recalculate
        all_tsubs_in_blocks = {}
        for b in blocks:
            for tsub in itertools.combinations(b, t):
                all_tsubs_in_blocks.setdefault(tuple(tsub), 0)
                all_tsubs_in_blocks[tuple(tsub)] += 1
                
        redundant_indices = []
        for i, b in enumerate(blocks):
            is_redundant = True
            for tsub in itertools.combinations(b, t):
                if all_tsubs_in_blocks.get(tuple(tsub), 0) <= 1:
                    is_redundant = False
                    break
            if is_redundant:
                redundant_indices.append(i)
                
        if not redundant_indices:
            break
        new_blocks = [b for i, b in enumerate(blocks) if i not in redundant_indices]
        
    return new_blocks

def _build_greedy(v, k, t, all_tsubs, rng, deadline):
    uncovered = set(all_tsubs)
    blocks = []
    
    while uncovered:
        if time.time() > deadline:
            break
            
        # Pick a random uncovered t-subset
        # To speed up, we can pick one that is "hard" to cover?
        # For now, random.
        seed = rng.choice(list(uncovered))
        block = list(seed)
        block_set = set(block)
        
        while len(block) < k:
            if time.time() > deadline:
                break
                
            best_gain = -1
            best_point = None
            t_minus_1 = t - 1
            # Precompute (t-1) subsets of current block
            block_t1 = set(itertools.combinations(block, t_minus_1))
            
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for comb in block_t1:
                    # The t-subset is comb + (x,)
                    # Need to check if it's in uncovered
                    # comb is a tuple of t-1 elements. x is an int.
                    # Sort to form a canonical tuple
                    # Optimization: 
                    # Instead of sorting every time, we can precompute?
                    # For small t, sorting is fast.
                    tsub = tuple(sorted(comb + (x,)))
                    if tsub in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            if best_point is None:
                # No point adds coverage? Should not happen if uncovered is not empty and block is not full?
                # If block is not full, there must be some way to cover seed?
                # Actually, if we picked seed, and we are building block, 
                # if no point increases gain, it means all potential t-subsets involving current block points 
                # and any new point are already covered? 
                # But seed is uncovered. Seed is in block. 
                # So seed is covered by the block we are building? 
                # Wait, the block is not complete yet. 
                # The gain calculation checks if the t-subset is in `uncovered`.
                # If seed is in block, and we add points, does seed become "covered" in the sense of the gain calc?
                # No, gain calc looks for NEW t-subsets covered by adding x.
                # Seed itself is a t-subset. It is in `uncovered`.
                # When we form the block, we haven't "removed" seed from uncovered yet.
                # But seed is a subset of the current block (if block size >= t).
                # The gain logic: "tsub is in uncovered".
                # If we add x, we check if `comb + x` is in uncovered.
                # If `comb + x` is the seed, and seed is in uncovered, then gain should be at least 1?
                # Yes, if `comb` is a subset of seed and `x` is the missing point?
                # Wait, seed is in `block`. So `seed` is a subset of `block`.
                # The t-subsets covered by the final block will include seed.
                # But during construction, we are adding points one by one.
                # The `uncovered` set is static during the block construction (we only update it after the block is done).
                # So if seed is in `uncovered`, and we form a t-subset that equals seed, it will be counted as gain.
                # So `best_gain` should be at least 1 if we can complete the seed?
                # Actually, if seed is in block, we don't need to "gain" it? 
                # No, we need to cover it. 
                # The gain counts how many *uncovered* t-subsets will be covered.
                # Since seed is uncovered, covering it is a gain.
                # So best_point should be found.
                
                # If best_point is None, it means no x gives gain > -1.
                # This implies gain == 0 for all x?
                # If gain is 0, it means no new t-subsets are covered.
                # But we must cover seed.
                # Maybe the logic is flawed?
                # If seed is in block, and block size is t, then the block IS the seed.
                # But k=7, t=4. Block size starts at 4.
                # If block size is 4, it is the seed. 
                # We need to add 3 more points.
                # Adding points will cover MORE t-subsets.
                # So gain should be > 0.
                
                # If best_point is None, pick random.
                candidates = [x for x in range(v) if x not in block_set]
                if not candidates:
                    break
                best_point = rng.choice(candidates)
                
            block.append(best_point)
            block_set.add(best_point)
            
        block.sort()
        blocks.append(block)
        
        # Update uncovered
        for tsub in itertools.combinations(block, t):
            uncovered.discard(tuple(tsub))
            
    return blocks

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0
    
    all_tsubs = [tuple(c) for c in itertools.combinations(range(v), t)]
    
    best_blocks = None
    best_len = float('inf')
    
    # Run multiple restarts
    # Use different random seeds
    num_restarts = 100
    for i in range(num_restarts):
        if time.time() > deadline - 10.0: # Leave time for validation/cleanup
            break
            
        rng = random.Random(1000 + i)
        blocks = _build_greedy(v, k, t, all_tsubs, rng, deadline)
        
        # Remove redundant blocks
        blocks = _remove_redundant_blocks(v, k, t, blocks)
        
        if len(blocks) < best_len:
            best_blocks = blocks
            best_len = len(blocks)
            
    if best_blocks is None:
        # Fallback: single greedy
        rng = random.Random(0)
        best_blocks = _build_greedy(v, k, t, all_tsubs, rng, deadline)
        best_blocks = _remove_redundant_blocks(v, k, t, best_blocks)
        
    return best_blocks
