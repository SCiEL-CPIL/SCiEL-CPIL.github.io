import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    all_points = range(v)
    # Initialize uncovered
    uncovered = set(itertools.combinations(all_points, t))
    blocks = []
    
    # To speed up redundancy check later, we can track cover counts
    # But for the greedy phase, we just need 'uncovered'
    
    while uncovered:
        if time.time() > deadline:
            break
            
        # Pick the lexicographically smallest uncovered subset
        seed_subset = min(uncovered)
        
        block = list(seed_subset)
        
        while len(block) < k:
            best_point = -1
            best_gain = -1
            
            for x in all_points:
                if x in block:
                    continue
                
                gain = 0
                # Check 4-subsets of current block
                for rest in itertools.combinations(block, t-1):
                    # Create 5-subset containing x and rest
                    # Since rest is a tuple from combinations, it's sorted.
                    # We need to merge x into it and check if it's in uncovered
                    # Simple way: create a set, add x, convert to tuple
                    # Or manual merge since t is small (5)
                    sub_set = set(rest)
                    sub_set.add(x)
                    sub = tuple(sorted(sub_set))
                    if sub in uncovered:
                        gain += 1
                
                if gain > best_gain or (gain == best_gain and (best_point == -1 or x < best_point)):
                    best_gain = gain
                    best_point = x
            
            if best_point == -1:
                break
            block.append(best_point)
        
        block.sort()
        blocks.append(block)
        
        # Update uncovered
        for sub in itertools.combinations(block, t):
            uncovered.discard(sub)
            
    # Post-processing: Remove redundant blocks
    if time.time() < deadline:
        # Calculate cover counts
        cover_count = {}
        for block in blocks:
            for sub in itertools.combinations(block, t):
                cover_count[sub] = cover_count.get(sub, 0) + 1
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            to_remove = []
            for i, block in enumerate(blocks):
                redundant = True
                for sub in itertools.combinations(block, t):
                    if cover_count[sub] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(i)
            
            if to_remove:
                # Remove blocks
                for i in reversed(to_remove):
                    block = blocks.pop(i)
                    for sub in itertools.combinations(block, t):
                        cover_count[sub] -= 1
                changed = True
                
    return blocks
