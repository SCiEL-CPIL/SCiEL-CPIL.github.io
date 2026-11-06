import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin for safety

    # Precompute all 6-subsets and their 5-subset members
    all_blocks = list(combinations(range(12), 6))
    block_5subsets = [list(combinations(b, 5)) for b in all_blocks]
    
    # 1. Fast greedy with bitmask-based 5-subset tracking
    # Map each 5-subset to an index for fast lookup
    all_5subsets = list(combinations(range(12), 5))
    subset_to_idx = {s: i for i, s in enumerate(all_5subsets)}
    n_5subsets = len(all_5subsets)  # 792
    
    # Precompute bitmask representation for each block's 5-subsets
    block_masks = []
    for b in all_blocks:
        mask = 0
        for s in combinations(b, 5):
            mask |= (1 << subset_to_idx[s])
        block_masks.append(mask)
    
    # Greedy: pick blocks in lexicographic order, check via bitmask
    covered_mask = 0
    blocks = []
    
    for i, b in enumerate(all_blocks):
        if time.time() > deadline:
            break
        if (covered_mask & block_masks[i]) == 0:
            blocks.append(list(b))
            covered_mask |= block_masks[i]
            if len(blocks) == 132:
                return blocks
    
    # 2. If greedy didn't reach 132, try randomized greedy with different orderings
    if len(blocks) < 132:
        import random
        best_count = len(blocks)
        
        for trial in range(500):
            if time.time() > deadline:
                break
            random.seed(trial)
            covered_f = 0
            blocks_f = []
            
            indices = list(range(len(all_blocks)))
            random.shuffle(indices)
            
            for idx in indices:
                if (covered_f & block_masks[idx]) == 0:
                    blocks_f.append(list(all_blocks[idx]))
                    covered_f |= block_masks[idx]
                    if len(blocks_f) == 132:
                        return blocks_f
            
            if len(blocks_f) > best_count:
                best_count = len(blocks_f)
                blocks = blocks_f
                if best_count == 132:
                    return blocks
    
    # 3. Backtracking search with pruning for remaining slots
    if len(blocks) < 132:
        # Use the best found so far and try to complete it
        best_blocks = blocks
        best_count = len(blocks)
        
        # Try completing from partial solutions
        for trial in range(100):
            if time.time() > deadline:
                break
            random.seed(10000 + trial)
            
            # Start with a random subset of the best solution and extend
            if best_blocks:
                k = min(len(best_blocks), random.randint(5, 20))
                start_blocks = random.sample(best_blocks, k)
            else:
                start_blocks = []
            
            covered_f = 0
            for b in start_blocks:
                bi = all_blocks.index(tuple(b))
                covered_f |= block_masks[bi]
            
            blocks_f = list(start_blocks)
            
            # Complete greedily
            indices = list(range(len(all_blocks)))
            random.shuffle(indices)
            
            for idx in indices:
                if (covered_f & block_masks[idx]) == 0:
                    blocks_f.append(list(all_blocks[idx]))
                    covered_f |= block_masks[idx]
                    if len(blocks_f) == 132:
                        return blocks_f
            
            if len(blocks_f) > best_count:
                best_count = len(blocks_f)
                best_blocks = blocks_f
                if best_count == 132:
                    return blocks
    
    return best_blocks if 'best_blocks' in dir() else blocks
