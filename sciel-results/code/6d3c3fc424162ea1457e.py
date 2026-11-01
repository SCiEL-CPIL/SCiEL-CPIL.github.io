import itertools
import time
import random

# Global caches
_subset_cache = {}
_cover_cache = {}

def _get_subset_data(v, t):
    key = (v, t)
    if key in _subset_cache:
        return _subset_cache[key]
        
    subset_to_idx = {}
    idx_to_subset = []
    idx = 0
    for s in itertools.combinations(range(v), t):
        mask = 0
        for p in s:
            mask |= (1 << p)
        subset_to_idx[mask] = idx
        idx_to_subset.append(mask)
        idx += 1
        
    _subset_cache[key] = (subset_to_idx, idx_to_subset, idx)
    return _subset_cache[key]

def _get_block_cover(block_mask, v, t):
    if block_mask in _cover_cache:
        return _cover_cache[block_mask]
        
    subset_to_idx, _, _ = _get_subset_data(v, t)
    
    cover_mask = 0
    points = []
    m = block_mask
    while m:
        lsb = m & -m
        p = lsb.bit_length() - 1
        points.append(p)
        m ^= lsb
        
    for s in itertools.combinations(points, t):
        smask = 0
        for p in s:
            smask |= (1 << p)
        if smask in subset_to_idx:
            cover_mask |= (1 << subset_to_idx[smask])
            
    _cover_cache[block_mask] = cover_mask
    return cover_mask

def _select_kth_set_bit(mask, k):
    count = 0
    m = mask
    idx = 0
    while m:
        if m & 1:
            if count == k:
                return idx
            count += 1
        m >>= 1
        idx += 1
    return -1

def _greedy_build(v, k, t, seed):
    rng = random.Random(seed)
    all_points = list(range(v))
    subset_to_idx, idx_to_subset, num_subsets = _get_subset_data(v, t)
    
    uncovered_mask = (1 << num_subsets) - 1
    blocks = []
    blocks_set = set()
    
    while uncovered_mask:
        num_uncovered = uncovered_mask.bit_count()
        target_rank = rng.randint(0, num_uncovered - 1)
        idx = _select_kth_set_bit(uncovered_mask, target_rank)
        target_subset_mask = idx_to_subset[idx]
        
        target_points = []
        tm = target_subset_mask
        while tm:
            lsb = tm & -tm
            p = lsb.bit_length() - 1
            target_points.append(p)
            tm ^= lsb
            
        remaining_points = [p for p in all_points if p not in target_points]
        needed = k - t
        
        best_block = None
        best_gain = -1
        best_candidates = []
        
        for combo in itertools.combinations(remaining_points, needed):
            block_points = target_points + list(combo)
            block_mask = 0
            for p in block_points:
                block_mask |= (1 << p)
                
            if block_mask in blocks_set:
                continue
                
            block_cover = _get_block_cover(block_mask, v, t)
            gain = (uncovered_mask & block_cover).bit_count()
            
            if gain > best_gain:
                best_gain = gain
                best_candidates = [block_mask]
            elif gain == best_gain:
                best_candidates.append(block_mask)
        
        if best_candidates:
            next_block = rng.choice(best_candidates)
            blocks.append(next_block)
            blocks_set.add(next_block)
            block_cover = _get_block_cover(next_block, v, t)
            uncovered_mask &= ~block_cover
            
    return blocks

def _local_search(blocks, v, k, t, deadline):
    if not blocks:
        return blocks
        
    subset_to_idx, _, num_subsets = _get_subset_data(v, t)
    
    block_covers = []
    for b in blocks:
        block_covers.append(_get_block_cover(b, v, t))
        
    cover_count = [0] * num_subsets
    for c in block_covers:
        m = c
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            cover_count[idx] += 1
            m ^= lsb
            
    changed = True
    while changed and time.time() < deadline:
        changed = False
        n = len(blocks)
        
        for i in range(n):
            c = block_covers[i]
            redundant = True
            m = c
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if cover_count[idx] == 1:
                    redundant = False
                    break
                m ^= lsb
                
            if redundant:
                m = c
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    cover_count[idx] -= 1
                    m ^= lsb
                    
                blocks.pop(i)
                block_covers.pop(i)
                changed = True
                break
                
    return blocks

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_blocks = None
    best_len = float('inf')
    
    seed = 0
    while time.time() < deadline:
        if time.time() > deadline - 5.0:
            break
            
        blocks = _greedy_build(v, k, t, seed)
        blocks = _local_search(blocks, v, k, t, deadline)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
        if best_len == 69:
            break
            
        seed += 1
        
    if best_blocks is None:
        return []
        
    result = []
    for b in best_blocks:
        points = []
        m = b
        while m:
            lsb = m & -m
            p = lsb.bit_length() - 1
            points.append(p)
            m ^= lsb
        result.append(points)
        
    return result
