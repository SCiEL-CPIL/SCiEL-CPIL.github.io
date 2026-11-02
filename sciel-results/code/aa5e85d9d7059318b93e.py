"""Randomized local search (simulated annealing) for covering design C(24,7,3)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time


def _precompute_subsets(v, k, t):
    """
    Precompute mapping from t-subsets to indices and point to subset indices.
    """
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    
    # Map subset tuple to index
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    # For each point, list of subset indices containing it
    point_to_subsets = [[] for _ in range(v)]
    for i, s in enumerate(all_subsets):
        for p in s:
            point_to_subsets[p].append(i)
            
    # Precompute bitmask for each point (integer with bits set for subsets containing p)
    # This allows fast OR operation for a block.
    point_masks = [0] * v
    for p in range(v):
        mask = 0
        for idx in point_to_subsets[p]:
            mask |= (1 << idx)
        point_masks[p] = mask
            
    return all_subsets, subset_to_idx, point_to_subsets, point_masks

def _greedy_cover(v, k, t, rng, point_masks, all_subsets, subset_to_idx):
    """
    Fast greedy construction.
    """
    uncovered_mask = (1 << len(all_subsets)) - 1
    blocks = []
    
    while uncovered_mask:
        # Pick a random uncovered subset
        while True:
            idx = rng.randint(0, len(all_subsets) - 1)
            if (uncovered_mask >> idx) & 1:
                break
        
        seed_subset = all_subsets[idx]
        block = list(seed_subset)
        rng.shuffle(block)
        
        # Greedily add points to maximize coverage of uncovered_mask
        while len(block) < k:
            best_x = None
            best_gain = -1
            cands = [x for x in range(v) if x not in block]
            rng.shuffle(cands)
            
            for x in cands:
                gain = (uncovered_mask & point_masks[x]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                    
            if best_x is None or best_gain == 0:
                if best_x is None:
                    break
                # If best_gain == 0, pick random to fill
                best_x = rng.choice(cands)
                
            block.append(best_x)
            
        block.sort()
        b_tuple = tuple(block)
        
        if b_tuple not in blocks:
            blocks.append(b_tuple)
            # Update uncovered mask
            b_mask = 0
            for p in b_tuple:
                b_mask |= point_masks[p]
            uncovered_mask &= ~b_mask
            
    return blocks

def construct(v, k, t):
    rng = random.Random(12345)
    start_time = time.time()
    time_limit = 115.0

    # Precompute
    all_subsets, subset_to_idx, point_to_subsets, point_masks = _precompute_subsets(v, k, t)
    n_subsets = len(all_subsets)
    
    # Initial Greedy
    blocks = _greedy_cover(v, k, t, rng, point_masks, all_subsets, subset_to_idx)
    
    # Compute counts
    counts = [0] * n_subsets
    for b in blocks:
        for combo in itertools.combinations(b, t):
            s_idx = subset_to_idx[combo]
            counts[s_idx] += 1
            
    best_blocks = list(blocks)
    best_count = len(blocks)
    
    def get_block_mask(b):
        mask = 0
        for p in b:
            mask |= point_masks[p]
        return mask

    def is_redundant(idx):
        """Check if block at idx is redundant."""
        b = blocks[idx]
        for combo in itertools.combinations(b, t):
            s_idx = subset_to_idx[combo]
            if counts[s_idx] == 1:
                return False
        return True

    def remove_block(idx):
        nonlocal blocks, counts
        b = blocks[idx]
        for combo in itertools.combinations(b, t):
            s_idx = subset_to_idx[combo]
            counts[s_idx] -= 1
        blocks.pop(idx)

    def add_block(b):
        nonlocal blocks, counts
        blocks.append(b)
        for combo in itertools.combinations(b, t):
            s_idx = subset_to_idx[combo]
            counts[s_idx] += 1

    # Local Search
    iteration = 0
    while time.time() - start_time < time_limit:
        iteration += 1
        progress = False
        
        # 1. Try to remove any redundant block
        indices = list(range(len(blocks)))
        rng.shuffle(indices)
        for i in indices:
            if is_redundant(i):
                remove_block(i)
                progress = True
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = list(blocks)
                break # Restart loop to check for more removals
                
        if progress:
            continue
            
        # 2. Try 2-Block Merge: Replace blocks A and B with a single block C
        n_blocks = len(blocks)
        if n_blocks >= 2:
            # Random sample of pairs
            pairs = list(itertools.combinations(range(n_blocks), 2))
            rng.shuffle(pairs)
            
            found_merge = False
            for i, j in pairs:
                b_i = blocks[i]
                b_j = blocks[j]
                
                # Get unique subsets for i and j
                unique_points = set()
                
                for combo in itertools.combinations(b_i, t):
                    s_idx = subset_to_idx[combo]
                    if counts[s_idx] == 1:
                        unique_points.update(combo)
                        
                for combo in itertools.combinations(b_j, t):
                    s_idx = subset_to_idx[combo]
                    if counts[s_idx] == 1:
                        unique_points.update(combo)
                        
                if len(unique_points) > k:
                    continue
                    
                # Construct C
                c_set = set(unique_points)
                cands = [x for x in range(v) if x not in c_set]
                rng.shuffle(cands)
                
                while len(c_set) < k:
                    c_set.add(cands.pop())
                    
                c_tuple = tuple(sorted(c_set))
                
                if c_tuple in blocks or c_tuple == b_i or c_tuple == b_j:
                    continue
                    
                # Perform Merge: Remove i, j, Add C
                # Remove j first (higher index)
                remove_block(j)
                remove_block(i)
                add_block(c_tuple)
                
                progress = True
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = list(blocks)
                
                found_merge = True
                break
                
            if found_merge:
                continue

        # 3. Perturbation: Swap a block with a random one to escape local minima
        if len(blocks) > 0:
            i = rng.randrange(len(blocks))
            b_i = blocks[i]
            
            unique_points = set()
            for combo in itertools.combinations(b_i, t):
                s_idx = subset_to_idx[combo]
                if counts[s_idx] == 1:
                    unique_points.update(combo)
                    
            if len(unique_points) <= k:
                c_set = set(unique_points)
                cands = [x for x in range(v) if x not in c_set]
                rng.shuffle(cands)
                while len(c_set) < k:
                    c_set.add(cands.pop())
                c_tuple = tuple(sorted(c_set))
                
                if c_tuple != b_i and c_tuple not in blocks:
                    remove_block(i)
                    add_block(c_tuple)
                    # Continue to next iteration to try removals/merges
                    continue

    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
