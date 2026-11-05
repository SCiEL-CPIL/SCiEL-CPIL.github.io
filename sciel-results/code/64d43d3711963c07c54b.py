import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    uncovered = set(all_t_subsets)
    blocks = []
    
    if not uncovered:
        return blocks
        
    # Greedy Construction
    while uncovered:
        # Pick the first uncovered subset to start the block
        first_uncovered = min(uncovered)
        block = [first_uncovered[0]]
        
        # Precompute point frequencies in uncovered for heuristic
        freq = {x: 0 for x in range(v)}
        for sub in uncovered:
            for x in sub:
                freq[x] += 1
        
        while len(block) < k:
            candidates = [x for x in range(v) if x not in block]
            
            if len(block) < t - 1:
                # Heuristic: pick point with highest frequency in uncovered
                # Tie-break by smallest index
                x = max(candidates, key=lambda p: (freq[p], -p))
            else:
                best_gain = -1
                best_x = -1
                core_combos = list(itertools.combinations(block, t - 1))
                
                for x in candidates:
                    gain = 0
                    for core in core_combos:
                        new_subset = tuple(sorted(core + (x,)))
                        if new_subset in uncovered:
                            gain += 1
                            
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        if gain == len(core_combos):
                            break
                            
                x = best_x
            
            block.append(x)
        
        block.sort()
        blocks.append(block)
        
        # Update uncovered
        for sub in itertools.combinations(block, t):
            uncovered.discard(sub)
            
        if time.time() > deadline:
            break

    # Minimization Phase
    if time.time() < deadline and blocks:
        counts = {}
        for block in blocks:
            for sub in itertools.combinations(block, t):
                counts[sub] = counts.get(sub, 0) + 1
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            new_blocks = []
            for block in blocks:
                redundant = True
                for sub in itertools.combinations(block, t):
                    if counts[sub] <= 1:
                        redundant = False
                        break
                
                if redundant:
                    for sub in itertools.combinations(block, t):
                        counts[sub] -= 1
                    changed = True
                else:
                    new_blocks.append(block)
            
            blocks = new_blocks

    return blocks
