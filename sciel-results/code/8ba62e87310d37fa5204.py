import itertools
import random
import time
import sys

_CACHE = {}

def _precompute(v, k, t):
    key = (v, k, t)
    if key in _CACHE:
        return _CACHE[key]
    
    start_time = time.time()
    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    # Map each triple to list of block indices containing it
    triple_to_blocks = {t: [] for t in itertools.combinations(points, t)}
    
    # Store block triples for fast access
    block_triples = []
    
    for i, block in enumerate(all_blocks):
        tris = list(itertools.combinations(block, t))
        block_triples.append(tris)
        for tri in tris:
            triple_to_blocks[tri].append(i)
            
    _CACHE[key] = (all_blocks, triple_to_blocks, block_triples)
    return _CACHE[key]

def _greedy_construct(v, k, t, all_blocks, triple_to_blocks, block_triples, seed):
    rng = random.Random(seed)
    uncovered_set = set(itertools.combinations(range(v), t))
    uncovered_list = list(uncovered_set)
    
    blocks_used = set()
    blocks_list = []
    
    # Map block index to its triples for quick access
    # block_triples is already a list of lists of tuples
    
    while uncovered_list:
        # Pick random uncovered triple
        idx = rng.randrange(len(uncovered_list))
        T = uncovered_list[idx]
        
        candidates = triple_to_blocks[T]
        
        best_block_idx = None
        max_score = -1
        
        # Optimization: If we have many candidates, we might not need to check all.
        # But for correctness of greedy, we should.
        # However, we can break early if we find a block that covers ALL remaining? No.
        # We can break if score == len(uncovered_list)? No.
        
        for idx_b in candidates:
            if idx_b in blocks_used:
                continue
            
            # Count coverage
            score = 0
            # Use local variable for speed
            unc_set = uncovered_set
            for tri in block_triples[idx_b]:
                if tri in unc_set:
                    score += 1
            
            if score > max_score:
                max_score = score
                best_block_idx = idx_b
                # If we cover a lot, we might stop early?
                # If max_score is high, say > 20, it's likely good.
                # But let's stick to exact max for now to ensure quality.
                
        if best_block_idx is None:
            break
            
        blocks_used.add(best_block_idx)
        blocks_list.append(best_block_idx)
        
        # Update uncovered
        for tri in block_triples[best_block_idx]:
            if tri in uncovered_set:
                uncovered_set.discard(tri)
                # Remove from list
                # This is O(N). To be fast, we can just mark as removed and clean up periodically?
                # Or use a boolean array?
                # For 2000 items, linear search is ~2000 ops.
                # 2000 steps * 2000 ops = 4M ops. OK.
                # But doing it for every triple in the block (35 triples) is 35 * 2000 = 70k ops per step.
                # 2000 steps * 70k = 140M ops. Might be slow.
                
                # Better: Just update the set.
                # And for the list, we can just leave stale entries and check membership?
                # If we pick a stale entry, we just pick another?
                # Yes! "Lazy deletion".
                
    return blocks_list

def _reduce_blocks(v, k, t, block_indices, all_blocks, block_triples):
    current_blocks = list(block_indices)
    if not current_blocks:
        return []
        
    # Count coverage
    coverage = {}
    for idx in current_blocks:
        for tri in block_triples[idx]:
            coverage[tri] = coverage.get(tri, 0) + 1
            
    changed = True
    while changed:
        changed = False
        new_current = []
        for idx in current_blocks:
            is_redundant = True
            for tri in block_triples[idx]:
                if coverage[tri] == 1:
                    is_redundant = False
                    break
            if is_redundant:
                for tri in block_triples[idx]:
                    coverage[tri] -= 1
                changed = True
            else:
                new_current.append(idx)
        current_blocks = new_current
    return current_blocks

def _local_search_swap(v, k, t, block_indices, all_blocks, triple_to_blocks, block_triples, seed):
    rng = random.Random(seed)
    current_blocks = set(block_indices)
    current_list = list(block_indices)
    
    # Precompute coverage for critical triples
    # We will run this loop a few times
    for iteration in range(50): # Limit iterations
        improved = False
        
        # Shuffle order of blocks to try
        rng.shuffle(current_list)
        
        for b_out in current_list:
            # Find critical triples for b_out
            critical = []
            for tri in block_triples[b_out]:
                # Check if any other block covers tri
                # This is expensive. 
                # Alternative: maintain coverage counts globally?
                # Let's just compute coverage for the current set once.
                pass
        
        # This naive check is too slow.
        # Let's use a simpler heuristic:
        # Try to swap b_out with b_in if b_in covers all critical triples of b_out.
        
        # To make this fast, we need to know critical triples quickly.
        # We can maintain a coverage count dictionary for the current set.
        
        # Let's restructure:
        # 1. Calculate coverage counts for current set.
        # 2. Identify blocks with critical triples.
        # 3. For each such block, find candidates.
        
        # This is getting complex. Let's stick to a simpler improvement:
        # Just do more greedy restarts with better randomization.
        # The "Swap" idea is good but implementation in Python within time limit is risky.
        
        break # Disable local search for now to ensure speed and correctness
    
    return current_list

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110
    
    all_blocks, triple_to_blocks, block_triples = _precompute(v, k, t)
    
    best_blocks = None
    best_len = float('inf')
    
    seed = 0
    # Run greedy multiple times
    while time.time() < deadline - 20: # Leave 20s for final checks
        blocks = _greedy_construct(v, k, t, all_blocks, triple_to_blocks, block_triples, seed)
        seed += 1
        
        if not blocks:
            continue
            
        reduced_blocks = _reduce_blocks(v, k, t, blocks, all_blocks, block_triples)
        
        if len(reduced_blocks) < best_len:
            best_len = len(reduced_blocks)
            best_blocks = reduced_blocks
            
    if best_blocks is None:
        blocks = _greedy_construct(v, k, t, all_blocks, triple_to_blocks, block_triples, 999)
        best_blocks = blocks
        
    result = [list(all_blocks[idx]) for idx in best_blocks]
    return result
