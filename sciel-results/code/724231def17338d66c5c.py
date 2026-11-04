import itertools
import random
import time

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t).
    Uses a greedy approach with randomized seed selection and post-optimization.
    """
    start_time = time.time()
    deadline = 115.0  # Safety margin for 120s limit
    
    # Precompute all t-subsets for fast lookup and counting
    all_t_subsets = list(itertools.combinations(range(v), t))
    uncovered = set(all_t_subsets)
    blocks = []
    
    # Helper to generate all t-subsets of a block
    def get_block_subsets(block):
        return list(itertools.combinations(block, t))

    # Helper to calculate gain of adding x to current block B
    # Returns number of uncovered t-subsets covered by B U {x} that were not in B
    def calc_gain(B, x, uncovered):
        gain = 0
        # New subsets must contain x and t-1 elements from B
        for rest in itertools.combinations(B, t - 1):
            subset = tuple(sorted(rest + (x,)))
            if subset in uncovered:
                gain += 1
        return gain

    # Construct the best block containing seed_subset
    def construct_best_block(seed_subset, uncovered):
        current_block = list(seed_subset)
        # We need to add k - t points
        for _ in range(k - t):
            best_x = None
            best_gain = -1
            # Candidates are points not in current_block
            for x in range(v):
                if x in current_block:
                    continue
                g = calc_gain(current_block, x, uncovered)
                if g > best_gain:
                    best_gain = g
                    best_x = x
            if best_x is None:
                break
            current_block.append(best_x)
        
        current_block.sort()
        return tuple(current_block)

    # Main greedy loop
    while uncovered:
        if time.time() > deadline - 5.0:
            break # Risk of timeout, but we hope we are close or fast enough
        
        # Sample candidates for the seed subset
        # If uncovered is small, take all. Otherwise sample.
        sample_size = min(50, len(uncovered))
        candidates = random.sample(list(uncovered), sample_size)
        
        best_block = None
        max_gain = -1
        
        for seed in candidates:
            block = construct_best_block(seed, uncovered)
            # Calculate total gain of this block
            gain = 0
            for subset in get_block_subsets(block):
                if subset in uncovered:
                    gain += 1
            
            if gain > max_gain:
                max_gain = gain
                best_block = block
        
        if best_block is None:
            break
        
        blocks.append(list(best_block))
        
        # Update uncovered
        for subset in get_block_subsets(best_block):
            uncovered.discard(subset)

    # Post-processing: Remove redundant blocks
    # This step is safe and fast
    if time.time() < deadline - 2.0:
        # Build a coverage count dictionary
        # cover_count[subset] = number of blocks covering it
        cover_count = {subset: 0 for subset in all_t_subsets}
        
        # Precompute subsets for each block to speed up removal check
        block_subsets = []
        for b in blocks:
            subs = list(itertools.combinations(tuple(b), t))
            block_subsets.append(subs)
            for s in subs:
                cover_count[s] += 1
        
        # Try to remove blocks
        changed = True
        while changed and time.time() < deadline - 1.0:
            changed = False
            for i in range(len(blocks)):
                # Check if block i is redundant
                # A block is redundant if for every subset T in block i, cover_count[T] > 1
                redundant = True
                for s in block_subsets[i]:
                    if cover_count[s] <= 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove block i
                    for s in block_subsets[i]:
                        cover_count[s] -= 1
                    blocks.pop(i)
                    block_subsets.pop(i)
                    changed = True
                    break # Restart loop as indices changed

    return blocks
