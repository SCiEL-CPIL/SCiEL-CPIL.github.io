import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Pre-compute all t-subsets for fast checking? 
    # Or just use sets.
    
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    freq = [0] * v
    
    # Helper to check if a tuple is a subset of a list
    def is_subset(sub, sup_list):
        sup_set = set(sup_list)
        return all(x in sup_set for x in sub)
        
    while uncovered:
        if time.time() > deadline:
            break
            
        # 1. Select seed t-subset
        # Strategy: Minimize sum of frequencies of points in the subset
        # This prioritizes "cold" points
        min_score = float('inf')
        seed_subset = None
        for sub in uncovered:
            score = sum(freq[x] for x in sub)
            if score < min_score:
                min_score = score
                seed_subset = sub
                # Optional: early exit if score is 0? No, might be better to pick one with some structure.
                # But min is good.
        
        if seed_subset is None:
            break
            
        current_block = list(seed_subset)
        
        # 2. Fill the block to size k
        while len(current_block) < k:
            if time.time() > deadline:
                break
                
            best_point = -1
            best_gain = -1
            best_freq = float('inf')
            
            # Candidates are points not in current_block
            for x in range(v):
                if x in current_block:
                    continue
                
                # Calculate gain: number of uncovered t-subsets covered by adding x
                # A t-subset S is covered if S \ {x} is a subset of current_block
                # and S is in uncovered.
                gain = 0
                # Iterate over all (t-1) subsets of current_block
                for rest in itertools.combinations(current_block, t - 1):
                    # Form the t-subset candidate
                    # rest is a tuple of t-1 elements
                    # We need to check if rest + (x,) is in uncovered
                    # Since rest is sorted (from combinations) and x is arbitrary, 
                    # we must sort to match the key in uncovered set
                    cand = tuple(sorted(rest + (x,)))
                    if cand in uncovered:
                        gain += 1
                
                # Tie-breaking: prefer lower frequency
                if gain > best_gain or (gain == best_gain and freq[x] < best_freq):
                    best_gain = gain
                    best_freq = freq[x]
                    best_point = x
            
            if best_point == -1:
                # Should not happen if uncovered is not empty and v > k
                break
                
            current_block.append(best_point)
        
        if len(current_block) == k:
            current_block.sort()
            blocks.append(current_block)
            
            # Update uncovered
            for sub in itertools.combinations(current_block, t):
                if sub in uncovered:
                    uncovered.remove(sub)
                    
            # Update frequencies
            for x in current_block:
                freq[x] += 1
        
        if time.time() > deadline:
            break

    # 3. Redundancy Removal Pass
    # We want to remove blocks that are not essential.
    # A block B is essential if there exists a t-subset S of B such that S is NOT covered by any other block.
    # If no such S exists, B is redundant.
    
    changed = True
    while changed:
        changed = False
        if time.time() > deadline:
            break
            
        new_blocks = []
        for i, block in enumerate(blocks):
            # Check if block is redundant
            redundant = True
            for sub in itertools.combinations(block, t):
                # Check if sub is covered by any OTHER block
                covered_by_other = False
                for j, other_block in enumerate(blocks):
                    if i == j:
                        continue
                    # Check if sub is subset of other_block
                    # Optimization: if len(other_block) < t, skip (not possible here)
                    # Simple check
                    if all(x in other_block for x in sub):
                        covered_by_other = True
                        break
                if not covered_by_other:
                    # This subset is only covered by 'block', so 'block' is essential
                    redundant = False
                    break
            
            if not redundant:
                new_blocks.append(block)
            
        if len(new_blocks) < len(blocks):
            blocks = new_blocks
            changed = True

    return blocks
