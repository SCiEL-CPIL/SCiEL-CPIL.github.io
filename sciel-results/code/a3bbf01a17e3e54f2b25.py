import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    time_limit = 145.0  # 5s safety margin
    
    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(all_points, t))
    
    best_blocks = None
    best_count = float('inf')
    
    def run_greedy():
        uncovered = set(all_t_subsets)
        blocks = []
        
        # Randomize the order in which we consider starting subsets
        shuffled_subs = list(all_t_subsets)
        random.shuffle(shuffled_subs)
        
        for sub in shuffled_subs:
            if sub in uncovered:
                block = list(sub)
                
                while len(block) < k:
                    gains = []
                    # Precompute (t-1) subsets of current block
                    # To avoid recomputing for each x
                    rest_combos = list(itertools.combinations(block, t - 1))
                    
                    for x in all_points:
                        if x in block:
                            continue
                        gain = 0
                        for rest in rest_combos:
                            cand = tuple(sorted(rest + (x,)))
                            if cand in uncovered:
                                gain += 1
                        gains.append((gain, x))
                    
                    if not gains:
                        break
                        
                    max_gain = max(g[0] for g in gains)
                    candidates = [x for g, x in gains if g == max_gain]
                    block.append(random.choice(candidates))
                
                block.sort()
                blocks.append(block)
                
                # Update uncovered
                for s in itertools.combinations(block, t):
                    uncovered.discard(s)
                    
                if not uncovered:
                    break
                    
        return blocks

    def remove_redundant(blocks):
        if not blocks:
            return blocks
        block_subsets = [set(itertools.combinations(b, t)) for b in blocks]
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                redundant = True
                for sub in block_subsets[i]:
                    covered = False
                    for j in range(len(blocks)):
                        if i == j: continue
                        if sub in block_subsets[j]:
                            covered = True
                            break
                    if not covered:
                        redundant = False
                        break
                if redundant:
                    blocks.pop(i)
                    block_subsets.pop(i)
                    changed = True
                    break
        return blocks

    while time.time() - start_time < time_limit:
        blocks = run_greedy()
        blocks = remove_redundant(blocks)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks

    if best_blocks is None:
        return []
        
    return best_blocks
