import itertools
import time
import random

def construct(v, k, t):
    # Parameters
    points = range(v)
    all_t_subsets = list(itertools.combinations(points, t))
    
    best_blocks = None
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    while time.time() < deadline:
        # 1. Randomized Greedy Construction
        uncovered = set(all_t_subsets)
        blocks = []
        
        while uncovered:
            # Pick a random uncovered t-subset to start the block
            start_t = random.choice(list(uncovered))
            current_block = list(start_t)
            
            # Greedily add points to current_block
            while len(current_block) < k:
                best_gain = -1
                best_candidates = []
                
                for x in points:
                    if x in current_block:
                        continue
                    
                    # Calculate gain: number of uncovered t-subsets covered by adding x
                    # A t-subset s is covered by current_block + [x] if s is a subset of it.
                    # Since s has size t, and current_block has size i, 
                    # s must contain some elements from current_block and x.
                    
                    # Efficient check:
                    # Generate all t-subsets of current_block + [x]
                    # and count how many are in uncovered.
                    # But we only care about NEW ones covered by x.
                    # Actually, any t-subset of (current_block + [x]) that is in uncovered 
                    # will be covered.
                    # We don't need to subtract those covered by current_block alone 
                    # because current_block is not a final block yet, 
                    # BUT if a t-subset is subset of current_block, it is already 
                    # "destined" to be covered. 
                    # However, the gain of adding x is specifically the t-subsets 
                    # that require x to be covered.
                    # i.e., t-subsets s such that s subset of (current_block + [x]) 
                    # AND x in s.
                    
                    # Let's just count all t-subsets of (current_block + [x]) in uncovered.
                    # This is a valid heuristic: prefer x that completes many subsets.
                    
                    temp_block = current_block + [x]
                    gain = 0
                    for s in itertools.combinations(temp_block, t):
                        if s in uncovered:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_candidates = [x]
                    elif gain == best_gain and gain > 0:
                        best_candidates.append(x)
                
                if not best_candidates:
                    # Should not happen if uncovered is not empty and k > t
                    # But if gain is 0 for all, pick random
                    best_candidates = [x for x in points if x not in current_block]
                
                # Randomly pick from best candidates to add diversity
                next_x = random.choice(best_candidates)
                current_block.append(next_x)
            
            current_block.sort()
            blocks.append(current_block)
            
            # Update uncovered
            for s in itertools.combinations(current_block, t):
                uncovered.discard(s)
        
        # 2. Local Search: Deletion
        # Compute cover counts
        cover_count = {s: 0 for s in all_t_subsets}
        for b in blocks:
            for s in itertools.combinations(b, t):
                cover_count[s] += 1
        
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for b in blocks:
                # Check if b is redundant
                redundant = True
                for s in itertools.combinations(b, t):
                    if cover_count[s] == 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove b
                    for s in itertools.combinations(b, t):
                        cover_count[s] -= 1
                    changed = True
                else:
                    new_blocks.append(b)
            
            blocks = new_blocks
        
        # Update best
        if best_blocks is None or len(blocks) < len(best_blocks):
            best_blocks = blocks
            
    return best_blocks
