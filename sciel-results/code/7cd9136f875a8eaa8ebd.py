import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    t_sub_set = set(all_t_subs)
    
    def greedy_cover_randomized():
        uncovered = set(t_sub_set)
        blocks = []
        
        while uncovered:
            seed_sub = random.choice(list(uncovered))
            block = list(seed_sub)
            
            while len(block) < k:
                candidates = [x for x in range(v) if x not in block]
                
                if not candidates:
                    break
                    
                gains = []
                for x in candidates:
                    count = 0
                    if len(block) >= t - 1:
                        for rest in itertools.combinations(block, t - 1):
                            s = tuple(sorted(rest + (x,)))
                            if s in uncovered:
                                count += 1
                    gains.append(count)
                
                max_gain = max(gains)
                
                if max_gain == 0:
                    x = random.choice(candidates)
                else:
                    best_candidates = [c for c, g in zip(candidates, gains) if g == max_gain]
                    x = random.choice(best_candidates)
                
                block.append(x)
                
            block.sort()
            blocks.append(block)
            
            for sub in itertools.combinations(block, t):
                uncovered.discard(sub)
                
        return blocks

    def remove_redundant(blocks):
        if not blocks:
            return blocks
        
        # Shuffle blocks to randomize removal order
        current_blocks = list(blocks)
        random.shuffle(current_blocks)
        
        # Init counts
        counts = {}
        for b in current_blocks:
            for sub in itertools.combinations(b, t):
                counts[sub] = counts.get(sub, 0) + 1
        
        while True:
            redundant_idx = -1
            for i, b in enumerate(current_blocks):
                is_red = True
                for sub in itertools.combinations(b, t):
                    if counts[sub] <= 1:
                        is_red = False
                        break
                if is_red:
                    redundant_idx = i
                    break
            
            if redundant_idx == -1:
                break
            
            b_to_remove = current_blocks.pop(redundant_idx)
            for sub in itertools.combinations(b_to_remove, t):
                counts[sub] -= 1
        
        return current_blocks

    best_cover = None
    best_size = float('inf')
    
    while time.time() < deadline:
        cover = greedy_cover_randomized()
        cover = remove_redundant(cover)
        
        if len(cover) < best_size:
            best_size = len(cover)
            best_cover = cover

    return best_cover if best_cover else []
