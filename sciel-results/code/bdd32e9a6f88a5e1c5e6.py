import itertools
import time

def construct(v, k, t):
    # Deadline guard: 120s limit, aim for 110s to be safe
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    uncovered = set(all_t_subsets)
    blocks = []
    
    # To speed up gain calculation, we can precompute nothing globally, 
    # but we can optimize the inner loop.
    
    while uncovered:
        if time.time() > deadline:
            break
            
        # Start with an uncovered t-subset
        first_uncovered = min(uncovered)
        block = list(first_uncovered)
        
        # Build the block
        while len(block) < k:
            # Precompute (t-1)-subsets of current block
            # Note: block is sorted initially, and we append, so we keep it sorted?
            # The seed doesn't keep it sorted during construction, but min(uncovered) is sorted.
            # Let's keep block sorted to make combinations generation consistent?
            # Actually, combinations don't care about input order for the *set* of combos, 
            # but the resulting tuples are in input order.
            # For `uncovered` lookup, we need sorted tuples.
            
            # Current block elements
            b_set = set(block)
            
            # Generate (t-1)-subsets of current block
            # If len(block) < t-1, gain is 0 for all? 
            # But we start with t elements, so len(block) >= t.
            # Wait, the loop starts with len(block) = t.
            # So we are adding the (t+1)-th element.
            # The new t-subsets formed will have x and t-1 elements from block.
            
            t_minus_1_combos = list(itertools.combinations(block, t - 1))
            
            best_x = -1
            max_gain = -1
            
            # Iterate candidates
            for x in range(v):
                if x in b_set:
                    continue
                
                # Calculate gain
                # Count how many (t-1)-combos from block, when combined with x, 
                # form a t-subset that is in uncovered.
                gain = 0
                for combo in t_minus_1_combos:
                    # Create the t-subset
                    # combo is a tuple of t-1 elements.
                    # We need to insert x into it and sort.
                    # Since t is small (4), manual sorting or tuple sorting is fast.
                    new_t = tuple(sorted(combo + (x,)))
                    if new_t in uncovered:
                        gain += 1
                        
                if gain > max_gain:
                    max_gain = gain
                    best_x = x
            
            # If no gain found (shouldn't happen if uncovered is not empty and we start with uncovered),
            # just pick the smallest available
            if best_x == -1:
                for x in range(v):
                    if x not in b_set:
                        best_x = x
                        break
            
            block.append(best_x)
            b_set.add(best_x)
        
        # Finalize block
        block.sort()
        blocks.append(block)
        
        # Remove all t-subsets of this block from uncovered
        for t_sub in itertools.combinations(block, t):
            uncovered.discard(t_sub)
            
    return blocks
