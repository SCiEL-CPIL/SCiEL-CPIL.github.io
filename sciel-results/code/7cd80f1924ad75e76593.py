import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145
    
    best_blocks = None
    best_size = float('inf')
    
    # Precompute combinations of remaining points for speed?
    # v=19, k=7, t=4.
    # S has size 4. Remaining 3 from 15.
    
    while time.time() < deadline:
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        
        while uncovered:
            # Pick a random uncovered subset to anchor the search
            # This helps in randomizing the path
            anchor = random.choice(list(uncovered))
            
            # Generate candidate blocks containing the anchor
            # Anchor is a tuple of t elements
            remaining_points = [x for x in range(v) if x not in anchor]
            
            best_candidate = None
            max_gain = -1
            
            # Iterate over combinations of size k-t from remaining_points
            for comb in itertools.combinations(remaining_points, k - t):
                candidate = tuple(sorted(anchor + comb))
                
                # Calculate gain
                gain = 0
                for sub in itertools.combinations(candidate, t):
                    if sub in uncovered:
                        gain += 1
                
                if gain > max_gain:
                    max_gain = gain
                    best_candidate = candidate
            
            if best_candidate is None:
                break
                
            blocks.append(best_candidate)
            
            # Update uncovered
            for sub in itertools.combinations(best_candidate, t):
                if sub in uncovered:
                    uncovered.discard(sub)
        
        # Deletion Phase
        cover_count = {}
        for b in blocks:
            for sub in itertools.combinations(b, t):
                cover_count[sub] = cover_count.get(sub, 0) + 1
        
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for b in blocks:
                essential = False
                for sub in itertools.combinations(b, t):
                    if cover_count[sub] == 1:
                        essential = True
                        break
                if essential:
                    new_blocks.append(b)
                else:
                    changed = True
                    for sub in itertools.combinations(b, t):
                        cover_count[sub] -= 1
            
            if changed:
                blocks = new_blocks
        
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks

    if best_blocks is None:
        return []
        
    return [list(b) for b in best_blocks]
