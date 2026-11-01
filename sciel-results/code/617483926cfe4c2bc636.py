import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Encode t-subsets as bitmasks
    all_t_subsets = set()
    for comb in itertools.combinations(range(v), t):
        mask = 0
        for x in comb:
            mask |= (1 << x)
        all_t_subsets.add(mask)
        
    best_blocks = None
    best_count = float('inf')
    
    # Number of restarts depends on time, but let's just loop until deadline
    while True:
        if time.time() > deadline - 5: # 5s margin for final cleanup
            break
            
        uncovered = set(all_t_subsets)
        blocks = []
        
        while uncovered:
            if time.time() > deadline - 5:
                break
            block = []
            for _ in range(k):
                best_x = -1
                best_gain = -1
                candidates = []
                for x in range(v):
                    if x in block:
                        continue
                    new_block = block + [x]
                    gain = 0
                    for t_comb in itertools.combinations(new_block, t):
                        mask = 0
                        for x in t_comb:
                            mask |= (1 << x)
                        if mask in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        candidates = [x]
                    elif gain == best_gain:
                        candidates.append(x)
                
                # Random tie-breaking
                block.append(random.choice(candidates))
                
            block.sort()
            blocks.append(block)
            
            for t_comb in itertools.combinations(block, t):
                mask = 0
                for x in t_comb:
                    mask |= (1 << x)
                uncovered.discard(mask)
                
        # If we broke out of inner loop early due to time, discard this run
        if uncovered:
            continue
            
        # Local Search: Remove redundant blocks
        cover_count = {mask: 0 for mask in all_t_subsets}
        for b in blocks:
            for t_comb in itertools.combinations(b, t):
                mask = 0
                for x in t_comb:
                    mask |= (1 << x)
                cover_count[mask] += 1
        
        new_blocks = []
        for b in blocks:
            redundant = True
            for t_comb in itertools.combinations(b, t):
                mask = 0
                for x in t_comb:
                    mask |= (1 << x)
                if cover_count[mask] == 1:
                    redundant = False
                    break
            if redundant:
                for t_comb in itertools.combinations(b, t):
                    mask = 0
                    for x in t_comb:
                        mask |= (1 << x)
                    cover_count[mask] -= 1
            else:
                new_blocks.append(b)
        
        blocks = new_blocks
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [b[:] for b in blocks]
            
        # If we found a very good solution, we can stop early?
        # The lower bound is 131. Best known 151.
        # If we find 151, it's good.
        # Let's keep searching for a bit longer to see if we can beat 151.
        
    if best_blocks is None:
        # Fallback: return empty or a trivial invalid set? 
        # The contract says "Valid when...". If we timeout, we should return the best found.
        # If no valid covering was found, we might return an empty list which is invalid.
        # But the greedy is fast, so it should find one.
        return []
        
    return best_blocks
