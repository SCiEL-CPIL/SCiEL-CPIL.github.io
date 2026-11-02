import itertools
import time
import random
import numpy as np

def _precompute(v, k, t):
    """Precompute all t-subsets, all k-blocks, and coverage masks."""
    all_tsets = list(itertools.combinations(range(v), t))
    num_tsets = len(all_tsets)
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}
    
    all_kblocks = list(itertools.combinations(range(v), k))
    num_kblocks = len(all_kblocks)
    block_to_idx = {b: i for i, b in enumerate(all_kblocks)}
    
    # Precompute coverage masks for each block
    block_masks = [0] * num_kblocks
    
    for b_idx, block in enumerate(all_kblocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            ts_idx = tset_to_idx[ts]
            mask |= (1 << ts_idx)
        block_masks[b_idx] = mask
        
    full_mask = (1 << num_tsets) - 1
    
    return all_tsets, all_kblocks, block_masks, tset_to_idx, block_to_idx, num_tsets, num_kblocks, full_mask

def _greedy_cover_numpy(v, k, t, precomp, seed=0, deadline=None):
    """Vectorized greedy using numpy."""
    if deadline is None:
        deadline = time.time() + 100
        
    all_tsets, all_kblocks, block_masks, tset_to_idx, block_to_idx, num_tsets, num_kblocks, full_mask = precomp
    
    # Build sparse incidence: for each tset, which blocks cover it
    # and for each block, which tsets it covers
    block_to_tsets = [[] for _ in range(num_kblocks)]
    tset_to_blocks = [[] for _ in range(num_tsets)]
    
    for b_idx, mask in enumerate(block_masks):
        m = mask
        while m:
            lsb = m & (-m)
            ts_idx = lsb.bit_length() - 1
            block_to_tsets[b_idx].append(ts_idx)
            tset_to_blocks[ts_idx].append(b_idx)
            m ^= lsb
    
    block_to_tsets_np = [np.array(x, dtype=np.int32) for x in block_to_tsets]
    tset_to_blocks_np = [np.array(x, dtype=np.int32) for x in tset_to_blocks]
    
    rng = random.Random(seed)
    
    gain = np.array([len(x) for x in block_to_tsets], dtype=np.int32)
    is_uncovered = np.full(num_tsets, True, dtype=bool)
    is_selected = np.full(num_kblocks, False, dtype=bool)
    sol = []
    
    while True:
        if time.time() > deadline:
            return None
            
        max_gain = int(np.max(gain))
        if max_gain <= 0:
            break
            
        candidates = np.where((gain == max_gain) & (~is_selected))[0]
        if len(candidates) == 0:
            break
            
        b_idx = int(candidates[rng.randint(0, len(candidates) - 1)])
        
        sol.append(list(all_kblocks[b_idx]))
        is_selected[b_idx] = True
        gain[b_idx] = -1
        
        bts = block_to_tsets_np[b_idx]
        newly_covered_mask = is_uncovered[bts]
        newly_covered = bts[newly_covered_mask]
        
        if len(newly_covered) == 0:
            continue
            
        is_uncovered[newly_covered] = False
        
        blocks_to_decrement = []
        for ts_idx in newly_covered:
            blocks_to_decrement.extend(tset_to_blocks[ts_idx])
            
        if blocks_to_decrement:
            counts = np.bincount(blocks_to_decrement, minlength=num_kblocks)
            gain -= counts
            gain[is_selected] = -1000
            
    return sol

def _reduce_blocks(blocks, precomp, deadline):
    """Remove redundant blocks from a valid solution."""
    all_tsets, all_kblocks, block_masks, tset_to_idx, block_to_idx, num_tsets, num_kblocks, full_mask = precomp
    
    blocks = list(blocks)
    
    block_indices = []
    for b in blocks:
        idx = block_to_idx.get(tuple(b), -1)
        if idx == -1:
            return blocks
        block_indices.append(idx)
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        n = len(block_indices)
        if n == 0:
            break
        for i in range(n):
            if time.time() > deadline:
                return blocks
            
            mask_without = 0
            for j in range(n):
                if j != i:
                    mask_without |= block_masks[block_indices[j]]
                    
            if mask_without == full_mask:
                block_indices.pop(i)
                changed = True
                break
                
    return [list(all_kblocks[idx]) for idx in block_indices]

def _repair_solution(blocks, precomp, deadline):
    """Make a solution valid by adding blocks if necessary."""
    all_tsets, all_kblocks, block_masks, tset_to_idx, block_to_idx, num_tsets, num_kblocks, full_mask = precomp
    
    blocks = list(blocks)
    block_indices = []
    for b in blocks:
        idx = block_to_idx.get(tuple(b), -1)
        if idx >= 0:
            block_indices.append(idx)
            
    covered_mask = 0
    for idx in block_indices:
        covered_mask |= block_masks[idx]
        
    if covered_mask == full_mask:
        return blocks
        
    remaining_mask = full_mask & ~covered_mask
    is_selected = [False] * num_kblocks
    for idx in block_indices:
        is_selected[idx] = True
        
    while remaining_mask != 0 and time.time() < deadline:
        best_idx = -1
        max_gain = 0
        for i in range(num_kblocks):
            if is_selected[i]:
                continue
            new_bits = block_masks[i] & remaining_mask
            g = new_bits.bit_count()
            if g > max_gain:
                max_gain = g
                best_idx = i
                
        if best_idx == -1 or max_gain == 0:
            break
            
        block_indices.append(best_idx)
        is_selected[best_idx] = True
        covered_mask |= block_masks[best_idx]
        remaining_mask = full_mask & ~covered_mask
        
    return [list(all_kblocks[idx]) for idx in block_indices]

def _local_search(blocks, precomp, deadline):
    """Local search: try to remove blocks, repair, and reduce."""
    all_tsets, all_kblocks, block_masks, tset_to_idx, block_to_idx, num_tsets, num_kblocks, full_mask = precomp
    
    rng = random.Random(int(time.time() * 1000) % 10000)
    best = list(blocks)
    best_len = len(best)
    
    while time.time() < deadline:
        # Try removing a random block and repairing
        if len(best) <= 1:
            break
        candidate = list(best)
        idx_to_remove = rng.randint(0, len(candidate) - 1)
        candidate.pop(idx_to_remove)
        
        # Repair
        candidate = _repair_solution(candidate, precomp, deadline)
        
        if len(candidate) < best_len:
            best = candidate
            best_len = len(candidate)
            
        # Reduce
        candidate = _reduce_blocks(best, precomp, deadline)
        if len(candidate) < best_len:
            best = candidate
            best_len = len(candidate)
            
    return best

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    precomp = _precompute(v, k, t)
    
    # Phase 1: Greedy Initialization
    best_sol = _greedy_cover_numpy(v, k, t, precomp, seed=0, deadline=start_time + 30.0)
    if best_sol is None:
        # Fallback to a simple greedy
        all_tsets, all_kblocks, block_masks, tset_to_idx, block_to_idx, num_tsets, num_kblocks, full_mask = precomp
        covered_mask = 0
        sol = []
        while covered_mask != full_mask:
            best_idx = -1
            best_gain = -1
            for i in range(num_kblocks):
                if any(all_kblocks[i] == s for s in sol):
                    continue
                gain = (block_masks[i] & ~covered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1:
                break
            sol.append(list(all_kblocks[best_idx]))
            covered_mask |= block_masks[best_idx]
        best_sol = sol
        
    best_len = len(best_sol)
    
    # Try a few more greedy seeds
    for i in range(1, 5):
        if time.time() > start_time + 50.0:
            break
        sol = _greedy_cover_numpy(v, k, t, precomp, seed=i, deadline=start_time + 50.0)
        if sol is not None and len(sol) < best_len:
            best_len = len(sol)
            best_sol = sol
            
    # Reduce initial solution
    best_sol = _reduce_blocks(best_sol, precomp, start_time + 60.0)
    best_len = len(best_sol)
    
    # Phase 2: Local Search
    if time.time() < start_time + 110.0:
        best_sol = _local_search(best_sol, precomp, start_time + 110.0)
        best_len = len(best_sol)
        
    # Final reduce
    best_sol = _reduce_blocks(best_sol, precomp, start_time + 115.0)
    
    return best_sol
