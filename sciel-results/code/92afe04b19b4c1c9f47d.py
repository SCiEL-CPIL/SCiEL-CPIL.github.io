import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initialize uncovered sets
    # For v=19, k=7, t=4, C(19,4) = 3876. This is small.
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    
    # Points 0..v-1
    points = list(range(v))
    
    while uncovered:
        if time.time() > deadline:
            break
            
        # Sample some "hardest" uncovered subsets to use as anchors
        # Using lexicographically first ones is a good deterministic heuristic for "hardness"
        # as they tend to be uncovered longer if not targeted.
        # Let's take the first 10 uncovered subsets.
        # Converting to list to get first N is O(N) if we sort, but uncovered is a set.
        # We can just take an arbitrary 10, or sort them. Sorting 3876 items is fast.
        # However, as uncovered shrinks, sorting is faster.
        # To be efficient, we can just take 10 random ones or the first 10 from a sorted list.
        # Let's sort to be consistent with "lexicographically first".
        sorted_uncovered = sorted(uncovered)
        
        num_anchors = min(10, len(sorted_uncovered))
        anchors = sorted_uncovered[:num_anchors]
        
        best_block = None
        best_gain = -1
        
        for anchor in anchors:
            current_block = list(anchor)
            
            # Greedily fill the block
            while len(current_block) < k:
                best_point = -1
                max_gain_for_point = -1
                
                # Calculate gain for each candidate point
                # We can precompute the (t-1)-subsets of current_block
                # to speed up the check
                t_minus_1_subsets = list(itertools.combinations(current_block, t - 1))
                
                for x in points:
                    if x in current_block:
                        continue
                    
                    gain = 0
                    for sub in t_minus_1_subsets:
                        # Create the new t-subset
                        # sub is a tuple of t-1 elements
                        # We need to add x and sort
                        new_t_sub = tuple(sorted(sub + (x,)))
                        if new_t_sub in uncovered:
                            gain += 1
                            
                    if gain > max_gain_for_point:
                        max_gain_for_point = gain
                        best_point = x
                        
                if best_point == -1:
                    break # Should not happen if v > k
                
                current_block.append(best_point)
            
            current_block.sort()
            
            # Evaluate how many currently uncovered subsets this block covers
            # We need to count how many of the original 'uncovered' set are subsets of current_block
            # Note: 'uncovered' hasn't been updated yet, so this is the true gain for this iteration.
            block_set = set(current_block)
            # A t-subset T is covered if T is a subset of current_block
            # Since T has size t, we can check if all elements of T are in block_set
            # Or generate all t-subsets of current_block and check if they are in uncovered
            
            covered_count = 0
            # Number of t-subsets in a k-block is C(k,t). For k=7, t=4, this is 35.
            for t_sub in itertools.combinations(current_block, t):
                if t_sub in uncovered:
                    covered_count += 1
                    
            if covered_count > best_gain:
                best_gain = covered_count
                best_block = current_block
                
        if best_block is None:
            break # Safety
            
        blocks.append(best_block)
        
        # Update uncovered
        block_set = set(best_block)
        # Remove all t-subsets of best_block from uncovered
        # We can iterate through all t-subsets of best_block
        to_remove = []
        for t_sub in itertools.combinations(best_block, t):
            if t_sub in uncovered:
                to_remove.append(t_sub)
                
        for t_sub in to_remove:
            uncovered.discard(t_sub)
            
    return blocks
