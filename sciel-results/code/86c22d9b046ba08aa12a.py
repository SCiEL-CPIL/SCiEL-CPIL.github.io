import itertools
import time
import random

def construct(v, k, t):
    all_points = range(v)
    
    def bits_to_mask(bits):
        m = 0
        for b in bits:
            m |= (1 << b)
        return m

    def mask_to_points(mask):
        return [i for i in range(v) if (mask >> i) & 1]

    # Pre-compute all t-subset masks
    all_t_masks = []
    for combo in itertools.combinations(all_points, t):
        all_t_masks.append(bits_to_mask(combo))
        
    def run_greedy(seed):
        random.seed(seed)
        uncovered = set(all_t_masks)
        blocks = []
        coverage_count = {m: 0 for m in all_t_masks}

        def add_block(block_pts):
            nonlocal uncovered, blocks, coverage_count
            blocks.append(sorted(block_pts))
            for sub in itertools.combinations(block_pts, t):
                m = bits_to_mask(sub)
                coverage_count[m] += 1
                if coverage_count[m] == 1:
                    if m in uncovered:
                        uncovered.remove(m)

        def remove_block(block_pts):
            nonlocal uncovered, blocks, coverage_count
            target = sorted(block_pts)
            for i, b in enumerate(blocks):
                if b == target:
                    blocks.pop(i)
                    break
            for sub in itertools.combinations(block_pts, t):
                m = bits_to_mask(sub)
                coverage_count[m] -= 1
                if coverage_count[m] == 0:
                    uncovered.add(m)

        def prune_blocks():
            nonlocal uncovered, blocks, coverage_count
            changed = True
            while changed:
                changed = False
                for i in range(len(blocks) - 1, -1, -1):
                    b = blocks[i]
                    essential = False
                    for sub in itertools.combinations(b, t):
                        m = bits_to_mask(sub)
                        if coverage_count[m] == 1:
                            essential = True
                            break
                    if not essential:
                        remove_block(b)
                        changed = True

        if k <= v:
            while uncovered:
                # Pick random uncovered t-subset
                start_mask = random.choice(list(uncovered))
                start_pts = mask_to_points(start_mask)
                
                block_pts = start_pts[:]
                
                while len(block_pts) < k:
                    best_gain = -1
                    best_candidates = []
                    
                    t_minus_1_subs = list(itertools.combinations(block_pts, t-1))
                    candidates = [x for x in range(v) if x not in block_pts]
                    
                    for x in candidates:
                        gain = 0
                        for sub in t_minus_1_subs:
                            m = 0
                            for p in sub:
                                m |= (1 << p)
                            m |= (1 << x)
                            if m in uncovered:
                                gain += 1
                        
                        if gain > best_gain:
                            best_gain = gain
                            best_candidates = [x]
                        elif gain == best_gain:
                            best_candidates.append(x)
                            
                    if best_candidates:
                        best_x = random.choice(best_candidates)
                    else:
                        best_x = random.choice(candidates)
                    
                    block_pts.append(best_x)
                
                add_block(block_pts)
            
            prune_blocks()
            return blocks

    best_blocks = None
    best_len = float('inf')
    start_time = time.time()
    max_time = 110
    
    seed = 0
    while time.time() - start_time < max_time:
        blocks = run_greedy(seed)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
        seed += 1

    if best_blocks is None:
        best_blocks = run_greedy(0)

    return best_blocks
