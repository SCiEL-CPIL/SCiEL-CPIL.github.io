import itertools
import time

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using a greedy algorithm 
    followed by post-processing to remove redundant blocks.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    uncovered = set(all_t_subsets)
    
    # Precompute (t-1)-subsets for each t-subset? 
    # Actually, in the gain calculation, we iterate over (t-1)-subsets of the current block B.
    # So we just need to generate combinations of B of size t-1.
    
    blocks = []
    
    # Greedy Phase
    while uncovered:
        if time.time() > deadline:
            break
            
        # Pick an uncovered t-subset to start the block
        # Using the first one is fine, or we could pick one that is "harder" to cover.
        # For simplicity and speed, pick the first.
        seed_subset = next(iter(uncovered))
        block = list(seed_subset)
        
        # Extend the block to size k
        while len(block) < k:
            if time.time() > deadline:
                break
                
            best_gain = -1
            best_point = -1
            block_set = set(block)
            
            # Generate all (t-1)-subsets of the current block
            # These are the "bases" for new t-subsets involving a new point x
            t_minus_1_subsets = list(itertools.combinations(block, t - 1))
            
            for x in range(v):
                if x in block_set:
                    continue
                
                gain = 0
                for base in t_minus_1_subsets:
                    # The new t-subset would be base + (x,)
                    # We need to check if it's in uncovered
                    # To avoid sorting every time, we can check membership directly
                    # But 'uncovered' contains sorted tuples.
                    # So we must form the sorted tuple.
                    new_t_sub = tuple(sorted(base + (x,)))
                    if new_t_sub in uncovered:
                        gain += 1
                        # Optimization: if gain is already very high, we might break?
                        # Max possible gain is len(t_minus_1_subsets).
                        if gain == len(t_minus_1_subsets):
                            break
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point != -1:
                block.append(best_point)
            else:
                # Fallback: just pick any point not in block
                for x in range(v):
                    if x not in block_set:
                        block.append(x)
                        break
        
        block.sort()
        blocks.append(block)
        
        # Update uncovered set
        # Remove all t-subsets of the new block from uncovered
        block_t_subsets = list(itertools.combinations(block, t))
        uncovered.difference_update(block_t_subsets)

    # Post-processing: Remove redundant blocks
    if time.time() < deadline:
        # Count coverage for each t-subset
        cover_count = {subset: 0 for subset in all_t_subsets}
        for block in blocks:
            for sub in itertools.combinations(block, t):
                cover_count[sub] += 1
        
        removed = True
        while removed and time.time() < deadline:
            removed = False
            for i in range(len(blocks) - 1, -1, -1):
                if time.time() > deadline:
                    break
                block = blocks[i]
                redundant = True
                for sub in itertools.combinations(block, t):
                    if cover_count[sub] <= 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove block
                    for sub in itertools.combinations(block, t):
                        cover_count[sub] -= 1
                    blocks.pop(i)
                    removed = True
    
    return blocks
