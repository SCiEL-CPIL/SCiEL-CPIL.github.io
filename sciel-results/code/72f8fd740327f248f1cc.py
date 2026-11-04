import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s buffer
    
    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    t_sub_set = set(all_t_subs)
    
    best_blocks = None
    best_size = float('inf')
    
    # We will run multiple randomized greedy attempts
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        blocks = greedy_cover(v, k, t, t_sub_set, random_seed=attempts)
        
        # Post-optimization: Remove redundant blocks
        if blocks:
            blocks = reduce_blocks(blocks, t, t_sub_set)
            
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = [list(b) for b in blocks]
            
        # If we hit a very good bound, we might want to stop early?
        # Lower bound is 264. Best known 335.
        # If we get close to 335, we should keep trying to improve.
        # But we must ensure we return something valid.
        
    if best_blocks is None:
        # Fallback: simple greedy
        best_blocks = greedy_cover(v, k, t, t_sub_set, random_seed=0)
        best_blocks = reduce_blocks(best_blocks, t, t_sub_set)
        
    return [list(b) for b in best_blocks]

def greedy_cover(v, k, t, t_sub_set, random_seed=0):
    rng = random.Random(random_seed)
    uncovered = set(t_sub_set)
    blocks = []
    
    while uncovered:
        # Pick a random uncovered t-subset to start the block
        # Or pick the one that is "hardest" to cover? 
        # For now, random selection from uncovered to diversify
        seed = rng.choice(list(uncovered))
        block = set(seed)
        
        # Fill the rest of the block
        while len(block) < k:
            candidates = [x for x in range(v) if x not in block]
            if not candidates:
                break
                
            # Calculate gain for each candidate
            # gain(x) = number of uncovered t-subsets that contain x and the other (t-1) elements from the current block?
            # No, the t-subset must be a subset of the *final* block.
            # If we add x, the new t-subsets covered are those of the form {x} U S where S is a (t-1)-subset of the current block.
            # Wait, the current block has size m. We are adding x to make size m+1.
            # The new t-subsets covered are those containing x and (t-1) elements from the current block.
            # Let's verify.
            # Current block B. New block B' = B U {x}.
            # t-subsets of B' are:
            # 1. t-subsets of B (already covered or not)
            # 2. t-subsets containing x: {x} U S where S is a (t-1)-subset of B.
            
            best_x = None
            best_gain = -1
            gains = []
            for x in candidates:
                gain = 0
                for s in itertools.combinations(block, t-1):
                    # Check if {x} U s is in uncovered
                    # We need to form the tuple and check
                    # Optimization: pre-compute combinations of block?
                    pass 
                
                # This inner loop is slow in pure Python.
                # Let's use a slightly faster check or just stick to simple logic if k, t are small.
                # v=19, k=8, t=5.
                # combinations(block, 4) is at most C(7,4)=35.
                # Candidates ~ 19.
                # 35 * 19 = 665 checks per step.
                # Steps ~ 8.
                # Blocks ~ 300.
                # 300 * 8 * 665 = ~1.5 million operations. This is fast.
                
                gain = 0
                for s in itertools.combinations(block, t-1):
                    # Create the tuple for set lookup
                    # s is a tuple of t-1 ints.
                    # We need to add x and sort.
                    # Since we are checking membership in a set of tuples, we need the sorted tuple.
                    # tuple(sorted(s + (x,)))
                    new_sub = tuple(sorted(s + (x,)))
                    if new_sub in uncovered:
                        gain += 1
                gains.append((gain, x))
                
            # Sort by gain, pick best (with random tie-breaking)
            gains.sort(key=lambda x: x[0], reverse=True)
            max_gain = gains[0][0]
            # Pick randomly from top gains to add diversity
            top_gains = [x for g, x in gains if g == max_gain]
            x = rng.choice(top_gains)
            block.add(x)
            
        block_list = sorted(block)
        blocks.append(block_list)
        
        # Update uncovered
        for sub in itertools.combinations(block_list, t):
            uncovered.discard(sub)
            
    return blocks

def reduce_blocks(blocks, t, t_sub_set):
    if not blocks:
        return []
        
    # Count how many times each t-subset is covered
    # Map: t-subset -> count
    # But we only care about subsets in the blocks.
    # We can just count for all t-subsets in the union of blocks.
    
    counts = {}
    for b in blocks:
        for sub in itertools.combinations(b, t):
            counts[sub] = counts.get(sub, 0) + 1
            
    # A block is redundant if all its t-subsets have count > 1
    # (meaning they are covered by at least one other block)
    # Note: If a subset is covered by 2 blocks, removing one still leaves it covered.
    
    changed = True
    current_blocks = [list(b) for b in blocks]
    
    while changed:
        changed = False
        new_blocks = []
        for b in current_blocks:
            is_redundant = True
            for sub in itertools.combinations(b, t):
                if counts.get(sub, 0) == 1:
                    is_redundant = False
                    break
            if not is_redundant:
                new_blocks.append(b)
            else:
                # Block is redundant, remove it
                changed = True
                # Update counts for this block's subsets
                for sub in itertools.combinations(b, t):
                    counts[sub] -= 1
        current_blocks = new_blocks
        
    return current_blocks
