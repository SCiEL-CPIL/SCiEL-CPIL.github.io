import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 150s limit

    points = list(range(v))
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_index = {sub: i for i, sub in enumerate(all_t_subsets)}

    # Precompute for each point, the list of t-subsets containing that point
    # This helps with local search later
    point_to_tsubs = [[] for _ in range(v)]
    for sub in all_t_subsets:
        for p in sub:
            point_to_tsubs[p].append(sub)

    # Precompute all t-subsets for each possible block (k-subset)
    # Too many blocks: C(19,8) = 75582, each has C(8,5)=56 t-subsets = 4.2M entries. 
    # Let's precompute lazily or use a different approach.

    def greedy_once(random_seed=None):
        rng = random.Random(random_seed)
        
        # Use a list of uncovered t-subset indices for faster random selection
        uncovered_indices = list(range(n_t))
        # Map from index to whether it's still uncovered
        is_uncovered = [True] * n_t
        
        blocks = []
        
        while uncovered_indices:
            # Pick a random uncovered t-subset
            pos = rng.randrange(len(uncovered_indices))
            idx = uncovered_indices[pos]
            s = all_t_subsets[idx]
            
            # Find all blocks of size k that contain s
            s_set = set(s)
            remaining_points = [p for p in points if p not in s_set]
            
            best_block = None
            best_gain = -1
            
            # Generate all combinations of k-t points from remaining_points
            for combo in itertools.combinations(remaining_points, k - t):
                block = tuple(sorted(s + combo))
                
                # Calculate gain: how many uncovered t-subsets does this block cover?
                gain = 0
                for sub in itertools.combinations(block, t):
                    sub_idx = t_index[sub]
                    if is_uncovered[sub_idx]:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
                elif gain == best_gain and gain > 0:
                    # Random tie break to diversify
                    if rng.random() < 0.3:
                        best_block = block
            
            if best_block is None:
                break
                
            blocks.append(best_block)
            
            # Remove covered t-subsets from uncovered
            for sub in itertools.combinations(best_block, t):
                sub_idx = t_index[sub]
                if is_uncovered[sub_idx]:
                    is_uncovered[sub_idx] = False
                    # Remove from uncovered_indices
                    # This is O(n) but we do it once per newly covered subset
                    try:
                        ui_pos = uncovered_indices.index(sub_idx)
                        uncovered_indices[ui_pos] = uncovered_indices[-1]
                        uncovered_indices.pop()
                    except ValueError:
                        pass
            
            # Time check
            if time.time() - start_time > time_limit - 5.0:
                return None
        
        return blocks

    def local_search_remove(blocks, random_seed=None):
        """Try to remove redundant blocks. A block is redundant if every t-subset
        it covers is also covered by another block."""
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
        
        n_blocks = len(blocks)
        # For each t-subset, count how many blocks cover it
        t_cover_count = [0] * n_t
        block_tsubs = []  # For each block, list of t-subset indices it covers
        
        for block in blocks:
            tsubs = []
            for sub in itertools.combinations(block, t):
                idx = t_index[sub]
                tsubs.append(idx)
                t_cover_count[idx] += 1
            block_tsubs.append(tsubs)
        
        # Try to remove blocks in random order
        kept = list(range(n_blocks))
        
        # Sort blocks by number of uniquely covered t-subsets (ascending) - remove those first
        # A block is "removable" if all its t-subsets have count > 1
        # We'll iterate and try to remove
        
        changed = True
        max_passes = 5
        for _pass in range(max_passes):
            if time.time() - start_time > time_limit - 3.0:
                break
            changed = False
            # Try each block in random order
            order = list(range(n_blocks))
            rng.shuffle(order)
            for bi in order:
                if bi not in kept:
                    continue
                # Check if all t-subsets of this block are covered by others
                removable = True
                for idx in block_tsubs[bi]:
                    if t_cover_count[idx] <= 1:
                        removable = False
                        break
                if removable:
                    # Remove this block
                    kept.remove(bi)
                    for idx in block_tsubs[bi]:
                        t_cover_count[idx] -= 1
                    changed = True
            
            if not changed:
                break
        
        return [blocks[i] for i in kept]

    best_blocks = None
    best_len = float('inf')
    
    num_restarts = 0
    while time.time() - start_time < time_limit - 5.0:
        seed = num_restarts
        res = greedy_once(seed)
        num_restarts += 1
        
        if res is not None:
            # Apply local search
            if time.time() - start_time < time_limit - 3.0:
                res = local_search_remove(res, seed)
            
            if len(res) < best_len:
                best_len = len(res)
                best_blocks = res
        
        if best_len <= 264:
            break

    if best_blocks is None:
        if time.time() - start_time < time_limit - 5.0:
            best_blocks = greedy_once(0)
            if best_blocks is not None:
                best_blocks = local_search_remove(best_blocks, 0)
    
    return best_blocks
