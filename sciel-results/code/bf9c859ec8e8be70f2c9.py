import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    points = list(range(v))
    all_tsubs = list(itertools.combinations(points, t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)
    
    all_kblocks = list(itertools.combinations(points, k))
    n_blocks = len(all_kblocks)
    
    block_masks = [0] * n_blocks
    tsub_to_blocks = [[] for _ in range(n_tsubs)]
    
    for i, B in enumerate(all_kblocks):
        mask = 0
        for ts in itertools.combinations(B, t):
            j = tsub_to_idx[ts]
            mask |= (1 << j)
            tsub_to_blocks[j].append(i)
        block_masks[i] = mask
        
    all_ones_mask = (1 << n_tsubs) - 1
    
    def popcount(x):
        # Use bit_count if available, else bin count
        if hasattr(int, 'bit_count'):
            return x.bit_count()
        return bin(x).count('1')

    def greedy_construct(order=None):
        uncovered = all_ones_mask
        selected = []
        if order is None:
            order = list(range(n_blocks))
            
        while uncovered:
            best_block = -1
            max_gain = 0
            for idx in order:
                gain = popcount(block_masks[idx] & uncovered)
                if gain > max_gain:
                    max_gain = gain
                    best_block = idx
                    if gain == popcount(uncovered):
                        break
            if best_block == -1:
                break
            selected.append(best_block)
            uncovered &= ~block_masks[best_block]
        return selected

    def optimize(blocks):
        if not blocks:
            return blocks
            
        blocks = list(set(blocks))
        
        # 1. Redundancy Removal
        changed = True
        while changed:
            changed = False
            new_blocks = []
            n = len(blocks)
            for i in range(n):
                b = blocks[i]
                others_mask = 0
                for j in range(n):
                    if i != j:
                        others_mask |= block_masks[blocks[j]]
                if block_masks[b] & ~others_mask:
                    new_blocks.append(b)
                else:
                    changed = True
            blocks = new_blocks
            
        # 2. 2-Swap Optimization
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            n = len(blocks)
            for i in range(n):
                if time.time() > deadline - 5:
                    return blocks
                for j in range(i + 1, n):
                    if time.time() > deadline - 5:
                        return blocks
                    b1 = blocks[i]
                    b2 = blocks[j]
                    
                    others_mask = 0
                    for l in range(n):
                        if l != i and l != j:
                            others_mask |= block_masks[blocks[l]]
                    
                    unique_mask = (block_masks[b1] | block_masks[b2]) & ~others_mask
                    
                    if unique_mask == 0:
                        blocks = [blocks[l] for l in range(n) if l != i and l != j]
                        improved = True
                        break
                    else:
                        t_idx = (unique_mask & -unique_mask).bit_length() - 1
                        candidates = tsub_to_blocks[t_idx]
                        
                        for b3 in candidates:
                            if b3 == b1 or b3 == b2:
                                continue
                            if unique_mask & ~block_masks[b3] == 0:
                                new_blocks = []
                                for l in range(n):
                                    if l != i and l != j:
                                        new_blocks.append(blocks[l])
                                new_blocks.append(b3)
                                blocks = new_blocks
                                improved = True
                                break
                        if improved:
                            break
                if improved:
                    break
        return blocks

    # Initial Greedy
    initial_blocks = greedy_construct()
    optimized_blocks = optimize(initial_blocks)
    best_blocks = optimized_blocks
    
    # Randomized Greedy Restarts
    if time.time() < deadline - 20:
        for _ in range(20):
            if time.time() > deadline - 10:
                break
            order = list(range(n_blocks))
            random.shuffle(order)
            sel = greedy_construct(order)
            opt_sel = optimize(sel)
            if len(opt_sel) < len(best_blocks):
                best_blocks = opt_sel
                
    return [list(all_kblocks[i]) for i in best_blocks]
