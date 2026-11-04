import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    
    best_blocks = None
    best_count = float('inf')
    
    while time.time() < deadline:
        uncovered = set(all_t_subsets)
        blocks = []
        
        while uncovered:
            # Pick first uncovered subset as anchor
            anchor = next(iter(uncovered))
            anchor_set = set(anchor)
            
            current_points = list(anchor)
            current_set = set(current_points)
            
            remaining_points = [x for x in range(v) if x not in current_set]
            
            for _ in range(k - t):
                best_gain = -1
                candidates = []
                
                for x in remaining_points:
                    gain = 0
                    # Iterate over (t-1) subsets of current_set
                    for y in itertools.combinations(current_set, t - 1):
                        sub = tuple(sorted((x,) + y))
                        if sub in uncovered:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        candidates = [x]
                    elif gain == best_gain:
                        candidates.append(x)
                
                # Randomly pick from candidates with max gain
                if candidates:
                    best_point = random.choice(candidates)
                    current_points.append(best_point)
                    current_set.add(best_point)
                    remaining_points.remove(best_point)
            
            final_block = tuple(sorted(current_points))
            blocks.append(final_block)
            
            for sub in itertools.combinations(final_block, t):
                uncovered.discard(sub)
        
        # Cleanup
        coverage_count = {sub: 0 for sub in all_t_subsets}
        for b in blocks:
            for sub in itertools.combinations(b, t):
                coverage_count[sub] += 1
        
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                b = blocks[i]
                is_redundant = True
                for sub in itertools.combinations(b, t):
                    if coverage_count[sub] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    for sub in itertools.combinations(b, t):
                        coverage_count[sub] -= 1
                    blocks.pop(i)
                    changed = True
                    break
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks[:]
            
        if time.time() > deadline:
            break

    if best_blocks is None:
        return []
        
    return [list(b) for b in best_blocks]
