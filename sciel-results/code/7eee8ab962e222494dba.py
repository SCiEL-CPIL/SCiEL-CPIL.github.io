import itertools

def construct(v, k, t):
    points = list(range(v))
    uncovered = set(itertools.combinations(points, t))
    blocks = []
    
    # Pre-compute all possible extensions for each possible anchor?
    # No, anchors change.
    
    while uncovered:
        # Convert to list to allow indexing and deterministic sampling
        # Sorting is expensive, so just take the first 10 in iteration order
        # To make it more robust, let's take the 10 with the smallest values?
        # min() is O(N). 
        # Let's just take a deterministic subset.
        # Since 'uncovered' is a set, iteration order is stable for a given insertion sequence.
        # We insert all initially, then remove. Removal doesn't change order of remaining in CPython sets?
        # Actually, set iteration order is not guaranteed to be stable across versions, 
        # but within a run it is consistent.
        
        # To be safe and deterministic, let's pick the anchor as the min element.
        # But sampling 10 is better.
        # Let's sort the uncovered set to get the "first" 10 lexicographically.
        # N=3876. Sort is fast.
        sorted_uncovered = sorted(uncovered)
        
        # Pick the first 10
        anchors = sorted_uncovered[:10]
        
        best_block = None
        best_gain = -1
        
        for anchor in anchors:
            # Find points not in anchor
            anchor_set = set(anchor)
            remaining_points = [p for p in points if p not in anchor_set]
            
            for ext in itertools.combinations(remaining_points, k - t):
                candidate_block = tuple(sorted(anchor + ext))
                
                gain = 0
                # Check coverage
                for sub in itertools.combinations(candidate_block, t):
                    if sub in uncovered:
                        gain += 1
                        # Early exit if we hit max possible? Max is 35.
                        # If gain > best_gain, we update.
                        # If gain == 35, it's max.
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = candidate_block
        
        if best_block is not None:
            blocks.append(list(best_block))
            for sub in itertools.combinations(best_block, t):
                uncovered.discard(sub)
                
    return blocks
