import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using global greedy set cover
    with bitmask-based gain tracking and local search optimization.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    # Precompute all t-subsets
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    subset_to_id = {s: i for i, s in enumerate(all_subsets)}
    id_to_subset = [None] * n_subsets
    for i, s in enumerate(all_subsets):
        id_to_subset[i] = s
    
    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Precompute masks for all blocks
    # Each mask is a bitset of the t-subsets covered by that block
    def get_block_mask(block):
        mask = 0
        for subset in itertools.combinations(block, t):
            sid = subset_to_id[subset]
            mask |= (1 << sid)
        return mask
    
    # Precompute all block masks - this is the expensive part
    # For v=24, k=7, t=3: C(24,7) = 346104 blocks, each has C(7,3)=35 subsets
    # That's about 12 million operations, should be manageable
    block_masks = [get_block_mask(b) for b in all_blocks]
    
    # Convert to numpy array for vectorized operations
    # Since n_subsets can be up to C(24,3) = 2024, we need big integers
    # Python's int handles this, but numpy doesn't directly for >64 bits
    # So we'll use Python ints for the masks
    
    def greedy_global(seed):
        """
        Global greedy set cover: at each step, pick the block with maximum
        gain (number of newly covered t-subsets), with random tie-breaking.
        """
        random.seed(seed)
        uncovered = (1 << n_subsets) - 1
        selected_indices = []
        
        while uncovered:
            best_gain = -1
            best_indices = []
            
            for i in range(n_blocks):
                mask = block_masks[i]
                gain = (mask & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_indices = [i]
                elif gain == best_gain:
                    best_indices.append(i)
            
            if not best_indices:
                break
            
            idx = random.choice(best_indices)
            selected_indices.append(idx)
            uncovered &= ~block_masks[idx]
        
        return [all_blocks[i] for i in selected_indices]
    
    def optimize(blocks, seed):
        """
        Local search optimization:
        1. Remove redundant blocks
        2. Try 2-for-1 replacements
        3. Try 1-for-1 swaps
        """
        random.seed(seed)
        n = len(blocks)
        
        # Count coverage for each t-subset
        counts = [0] * n_subsets
        for b in blocks:
            for subset in itertools.combinations(b, t):
                counts[subset_to_id[subset]] += 1
        
        improved = True
        max_iters = 2000
        iters = 0
        
        while improved and iters < max_iters and time.time() < deadline - 10:
            improved = False
            iters += 1
            
            # 1. Remove redundant blocks
            for i in range(len(blocks)):
                if time.time() > deadline - 15:
                    return blocks
                b = blocks[i]
                redundant = True
                for subset in itertools.combinations(b, t):
                    if counts[subset_to_id[subset]] == 1:
                        redundant = False
                        break
                if redundant:
                    blocks.pop(i)
                    for subset in itertools.combinations(b, t):
                        counts[subset_to_id[subset]] -= 1
                    improved = True
                    break
            
            if improved:
                continue
            
            # 2. Try 2-for-1: replace two blocks with one
            n = len(blocks)
            if n >= 2:
                # Try a random sample of pairs to avoid O(n^2) in worst case
                max_pairs_to_try = min(500, n * (n - 1) // 2)
                tried = set()
                attempts = 0
                while attempts < max_pairs_to_try:
                    i = random.randint(0, n - 1)
                    j = random.randint(0, n - 1)
                    if i == j:
                        continue
                    key = (min(i, j), max(i, j))
                    if key in tried:
                        continue
                    tried.add(key)
                    attempts += 1
                    
                    if i > j:
                        i, j = j, i
                    
                    b1 = blocks[i]
                    b2 = blocks[j]
                    
                    # Find subsets that are only covered by b1 or b2
                    needed = []
                    for subset in itertools.combinations(b1, t):
                        sid = subset_to_id[subset]
                        if counts[sid] == 1:
                            needed.append(sid)
                    for subset in itertools.combinations(b2, t):
                        sid = subset_to_id[subset]
                        if counts[sid] == 1:
                            needed.append(sid)
                    
                    if not needed:
                        # Both are redundant
                        b_remove = blocks.pop(j)
                        for subset in itertools.combinations(b_remove, t):
                            counts[subset_to_id[subset]] -= 1
                        b_remove = blocks.pop(i)
                        for subset in itertools.combinations(b_remove, t):
                            counts[subset_to_id[subset]] -= 1
                        improved = True
                        break
                    
                    # Find points that must be in the new block
                    points_needed = set()
                    for sid in needed:
                        points_needed.update(id_to_subset[sid])
                    
                    if len(points_needed) > k:
                        continue
                    
                    # Try to find a block containing all needed points
                    remaining = [p for p in range(v) if p not in points_needed]
                    need = k - len(points_needed)
                    
                    found = False
                    for comb in itertools.combinations(remaining, need):
                        new_block = tuple(sorted(points_needed.union(comb)))
                        
                        # Check if this new block covers all needed subsets
                        new_mask = 0
                        for subset in itertools.combinations(new_block, t):
                            new_mask |= (1 << subset_to_id[subset])
                        
                        covers_all = True
                        for sid in needed:
                            if not (new_mask & (1 << sid)):
                                covers_all = False
                                break
                        
                        if covers_all and new_block not in (b1, b2):
                            # Perform the swap
                            b_remove = blocks.pop(j)
                            for subset in itertools.combinations(b_remove, t):
                                counts[subset_to_id[subset]] -= 1
                            b_remove = blocks.pop(i)
                            for subset in itertools.combinations(b_remove, t):
                                counts[subset_to_id[subset]] -= 1
                            
                            blocks.append(list(new_block))
                            for subset in itertools.combinations(new_block, t):
                                counts[subset_to_id[subset]] += 1
                            
                            improved = True
                            found = True
                            break
                    
                    if found:
                        break
            
            # 3. Try 1-for-1 swaps
            n = len(blocks)
            if n >= 1:
                max_swaps_to_try = min(200, n * 10)
                tried = set()
                attempts = 0
                while attempts < max_swaps_to_try:
                    i = random.randint(0, n - 1)
                    if i in tried:
                        continue
                    tried.add(i)
                    attempts += 1
                    
                    b_old = blocks[i]
                    
                    # Find subsets only covered by b_old
                    needed = []
                    for subset in itertools.combinations(b_old, t):
                        sid = subset_to_id[subset]
                        if counts[sid] == 1:
                            needed.append(sid)
                    
                    if not needed:
                        # b_old is redundant
                        blocks.pop(i)
                        for subset in itertools.combinations(b_old, t):
                            counts[subset_to_id[subset]] -= 1
                        improved = True
                        break
                    
                    points_needed = set()
                    for sid in needed:
                        points_needed.update(id_to_subset[sid])
                    
                    if len(points_needed) > k:
                        continue
                    
                    remaining = [p for p in range(v) if p not in points_needed]
                    need = k - len(points_needed)
                    
                    found = False
                    for comb in itertools.combinations(remaining, need):
                        new_block = tuple(sorted(points_needed.union(comb)))
                        
                        if new_block == b_old:
                            continue
                        
                        new_mask = 0
                        for subset in itertools.combinations(new_block, t):
                            new_mask |= (1 << subset_to_id[subset])
                        
                        covers_all = True
                        for sid in needed:
                            if not (new_mask & (1 << sid)):
                                covers_all = False
                                break
                        
                        if covers_all:
                            # Perform the swap
                            for subset in itertools.combinations(b_old, t):
                                counts[subset_to_id[subset]] -= 1
                            
                            blocks[i] = list(new_block)
                            for subset in itertools.combinations(new_block, t):
                                counts[subset_to_id[subset]] += 1
                            
                            improved = True
                            found = True
                            break
                    
                    if found:
                        break
        
        return blocks
    
    # Main search loop
    best_blocks = None
    best_count = float('inf')
    
    # Try multiple seeds with global greedy
    seed = 42
    while time.time() < deadline - 20:
        blocks = greedy_global(seed)
        if blocks:
            blocks = optimize(blocks, seed)
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
                if best_count <= 76:
                    break
        seed += 1
    
    if best_blocks is None:
        # Fallback: simple greedy
        uncovered = set(all_subsets)
        blocks = []
        for i in range(n_blocks):
            if not uncovered:
                break
            b = all_blocks[i]
            covered = False
            for subset in itertools.combinations(b, t):
                if subset in uncovered:
                    covered = True
                    break
            if covered:
                blocks.append(list(b))
                for subset in itertools.combinations(b, t):
                    uncovered.discard(subset)
        best_blocks = blocks
    
    return best_blocks
