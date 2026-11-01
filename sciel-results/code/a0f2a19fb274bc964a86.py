import itertools
import random
import time

def construct(v, k, t):
    # Parameters for C(24, 7, 3)
    # v=24, k=7, t=3
    
    if v != 24 or k != 7 or t != 3:
        # Fallback for other parameters, though task specifies 24,7,3
        # Simple greedy fallback
        return _simple_greedy(v, k, t)
        
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Precompute all triples
    all_triples = list(itertools.combinations(range(v), t))
    triple_set = set(all_triples)
    
    # Precompute mapping from point to triples containing it? 
    # Might be useful for optimization, but let's stick to the candidate generation strategy.
    
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple randomized greedy instances
    while time.time() < deadline:
        blocks = _randomized_greedy(v, k, t, triple_set, random_seed=random.randint(0, 10**6))
        
        # Apply reduction
        blocks = _reduce_blocks(blocks, v, t)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
            # Optional: if we hit the known best (76), we can stop or try to beat it
            if best_count <= 76:
                # We found a very good solution. Try to verify if we can do better with more time?
                # Or just break if we are confident.
                # Let's continue to see if we can find < 76 (unlikely but possible) or just stabilize.
                pass

    if best_blocks is None:
        # Should not happen, but safe fallback
        best_blocks = _simple_greedy(v, k, t)
        
    # Final reduction on the best result just in case
    best_blocks = _reduce_blocks(best_blocks, v, t)
    
    # Ensure blocks are lists of ints, distinct, etc.
    return [list(b) for b in best_blocks]

def _simple_greedy(v, k, t):
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        # Pick first uncovered
        target = min(uncovered)
        # Find best block containing target
        best_block = None
        best_gain = -1
        # Generate all k-subsets containing target
        # target is a tuple of t elements
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

def _randomized_greedy(v, k, t, triple_set, random_seed):
    rng = random.Random(random_seed)
    uncovered = set(triple_set)
    blocks = []
    
    # Cache for candidate generation? 
    # For a triple T, the candidates are all k-subsets containing T.
    # We can precompute the list of "extensions" for each triple?
    # There are 2024 triples. Each has C(21, 4) = 5985 extensions.
    # 2024 * 5985 ~ 12M tuples of 4 ints. Memory might be high but maybe ok?
    # 12M * 4 * 8 bytes = ~400MB. Too high.
    # Let's generate on the fly.
    
    while uncovered:
        # Pick a random uncovered triple to start
        # Using list(uncovered) is O(N). 
        # Since uncovered shrinks, we can just pick a random index from the set?
        # Python sets don't support random access.
        # We can maintain a list of uncovered triples and remove them, but removal is O(N).
        # Alternatively, just pick min(uncovered) for determinism in the "randomized" part?
        # No, randomization is key.
        # Let's convert to list periodically? Or just use `next(iter(uncovered))`?
        # `next(iter(uncovered))` is fast but not random.
        # To get a random element from a set in Python without converting to list (O(N)):
        # There is no built-in O(1) random access.
        # However, N is small (max 2024). Converting to list is O(2024).
        # Doing this every step (100 steps) is 200k ops. Negligible.
        
        unc_list = list(uncovered)
        target = unc_list[rng.randrange(len(unc_list))]
        
        # Find best block containing target
        best_block = None
        best_gain = -1
        
        target_set = set(target)
        remaining_points = [p for p in range(v) if p not in target_set]
        need = k - t
        
        # Optimization: We want to maximize the number of *uncovered* triples in the new block.
        # The new block will contain `target` (which is uncovered) and 34 other triples.
        # So gain is at least 1.
        
        for combo in itertools.combinations(remaining_points, need):
            block = tuple(sorted(target + combo))
            
            # Calculate gain
            gain = 0
            # We know 'target' is in uncovered.
            gain += 1 
            # Check the other 34 triples
            # itertools.combinations(block, t) generates 35 triples.
            # We can skip 'target' to save one check, but it's negligible.
            for sub in itertools.combinations(block, t):
                if sub in uncovered:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_block = block
            elif gain == best_gain:
                # Tie-breaking: random pick among best
                if rng.random() < 0.5:
                    best_block = block
                    
        blocks.append(best_block)
        
        # Update uncovered
        for sub in itertools.combinations(best_block, t):
            uncovered.discard(sub)
            
    return blocks

def _reduce_blocks(blocks, v, t):
    """
    Remove redundant blocks.
    A block is redundant if every t-subset in it is covered by at least one other block.
    """
    if not blocks:
        return []
        
    # Map each triple to the list of block indices that cover it
    # triple_to_blocks: dict[tuple, list[int]]
    triple_to_blocks = {}
    
    for i, block in enumerate(blocks):
        for sub in itertools.combinations(block, t):
            if sub not in triple_to_blocks:
                triple_to_blocks[sub] = []
            triple_to_blocks[sub].append(i)
            
    # Identify blocks that can be removed
    # A block i is removable if for all sub in block_i, len(triple_to_blocks[sub]) > 1
    # Note: if we remove a block, the counts change.
    # So we should iterate until stable.
    
    active = [True] * len(blocks)
    changed = True
    while changed:
        changed = False
        for i in range(len(blocks)):
            if not active[i]:
                continue
            # Check if block i is redundant given the currently active blocks
            redundant = True
            block = blocks[i]
            for sub in itertools.combinations(block, t):
                # Count active blocks covering sub
                count = 0
                for j in triple_to_blocks[sub]:
                    if active[j]:
                        count += 1
                        if count > 1:
                            break
                if count == 1:
                    # This sub is only covered by block i (among active ones)
                    # So block i is NOT redundant
                    redundant = False
                    break
            if redundant:
                active[i] = False
                changed = True
                
    # Rebuild list of active blocks
    return [blocks[i] for i in range(len(blocks)) if active[i]]
