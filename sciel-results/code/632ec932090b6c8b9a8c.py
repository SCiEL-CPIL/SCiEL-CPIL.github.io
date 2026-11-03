import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = 118.0  # Safety margin (evaluator kills at 120s, we need margin for overhead)

    def elapsed():
        return time.time() - start_time

    # 1. Precomputation
    # Generate all t-sets and assign indices
    all_tsets = list(itertools.combinations(range(v), t))
    num_tsets = len(all_tsets)
    tset_index = {ts: i for i, ts in enumerate(all_tsets)}
    
    # Generate all blocks and their bitmasks
    # v=19, k=8, t=5 -> C(19,8) = 75582 blocks, C(19,5) = 11628 t-sets
    all_blocks = []
    block_masks = []
    for block in itertools.combinations(range(v), k):
        mask = 0
        for ts in itertools.combinations(block, t):
            mask |= (1 << tset_index[ts])
        all_blocks.append(block)
        block_masks.append(mask)
        
    num_blocks = len(all_blocks)
    
    # Initial full mask
    full_mask = (1 << num_tsets) - 1

    # 2. Greedy Set Cover with randomized tie-breaking
    def greedy_cover(seed):
        rng = random.Random(seed)
        uncovered = full_mask
        selected = []
        
        # Pre-compute block sizes (all same) but we need dynamic counts.
        # To speed up, we can use a heap or just scan. Scanning 75k is fast.
        # We'll keep a list of active blocks.
        
        while uncovered:
            if elapsed() > deadline - 5:
                break
            best_idx = -1
            best_count = -1
            # Random tie-breaking: iterate all, keep max, on tie randomize?
            # Or just shuffle the iteration order?
            # Shuffling 75k items every step is expensive.
            # Instead, we can add a small random noise to the count?
            # Or just pick the first max. To diversify, we can shuffle the *blocks* list once at start?
            # No, we need to re-evaluate counts.
            # Let's just iterate all.
            
            # Optimization: If we have a list of blocks, we can filter out blocks that are subsets of others? No, all same size.
            
            for i in range(num_blocks):
                # If already selected, skip? 
                # We can't easily track "selected" in a set for O(1) lookup inside the loop if we don't maintain a boolean array.
                # Let's maintain a boolean array `used`.
                pass
            
            # Actually, let's just iterate all and check a `used` array.
            # But `used` check is O(1).
            # The cost is `popcount`.
            
            # To avoid checking used blocks, we can maintain a list of `active` blocks.
            # But removing from a list is O(N).
            # Let's just iterate all and skip used.
            
            # Wait, the previous implementation had a bug or was inefficient?
            # Let's use a standard approach:
            # Maintain `uncovered` mask.
            # For each block, compute `count = (block_masks[i] & uncovered).bit_count()`.
            # Pick max.
            
            # To make it faster, we can pre-compute `block_masks` as integers.
            # `bit_count()` is very fast in Python 3.10+.
            
            # Let's optimize the inner loop by not using a `used` array if we remove blocks from a list?
            # Removing from a list is O(N).
            # Let's just use a `used` boolean array.
            
            # Re-initialize used array? No, it's persistent.
            # But we are in a fresh greedy_cover call.
            
            used = [False] * num_blocks
            active_indices = list(range(num_blocks))
            
            while uncovered:
                if elapsed() > deadline - 5:
                    break
                best_idx = -1
                best_count = -1
                
                for i in active_indices:
                    if used[i]:
                        continue
                    count = (block_masks[i] & uncovered).bit_count()
                    if count > best_count:
                        best_count = count
                        best_idx = i
                        if count == num_tsets: # Perfect cover (unlikely)
                            break
                
                if best_idx == -1 or best_count == 0:
                    break
                    
                used[best_idx] = True
                selected.append(best_idx)
                uncovered &= ~block_masks[best_idx]
                
            return selected

    # 3. Local Search (1-for-1 swap and deletion)
    # Precompute tset_to_blocks
    tset_to_blocks = [[] for _ in range(num_tsets)]
    for b_idx, mask in enumerate(block_masks):
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            tset_to_blocks[idx].append(b_idx)
            m ^= lsb
            
    def local_search_v2(initial_sol, seed):
        rng = random.Random(seed)
        sol = list(initial_sol)
        in_sol = [False] * num_blocks
        for b in sol:
            in_sol[b] = True
            
        cover_counts = [0] * num_tsets
        
        def update_counts(block_idx, delta):
            m = block_masks[block_idx]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cover_counts[idx] += delta
                m ^= lsb

        for b in sol:
            update_counts(b, 1)
            
        best_sol = list(sol)
        best_size = len(sol)
        no_improve = 0
        max_no_improve = 50
        
        while elapsed() < deadline - 10 and no_improve < max_no_improve:
            improved = False
            
            # Phase 1: Remove redundant
            order = list(range(len(sol)))
            rng.shuffle(order)
            for pos in order:
                if pos >= len(sol):
                    continue
                b = sol[pos]
                m = block_masks[b]
                redundant = True
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cover_counts[idx] == 1:
                        redundant = False
                        break
                    m ^= lsb
                    
                if redundant:
                    update_counts(b, -1)
                    in_sol[b] = False
                    sol.pop(pos)
                    improved = True
                    if len(sol) < best_size:
                        best_size = len(sol)
                        best_sol = list(sol)
                    break
                    
            if improved:
                no_improve = 0
                continue
                
            # Phase 2: 1-for-1 Swap
            if sol:
                found_swap = False
                for _ in range(20): # Try more swaps
                    if not sol:
                        break
                    pos = rng.randint(0, len(sol) - 1)
                    b_out = sol[pos]
                    
                    critical_mask = 0
                    m = block_masks[b_out]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        if cover_counts[idx] == 1:
                            critical_mask |= lsb
                        m ^= lsb
                        
                    if critical_mask == 0:
                        # Redundant
                        update_counts(b_out, -1)
                        in_sol[b_out] = False
                        sol.pop(pos)
                        improved = True
                        found_swap = True
                        break
                        
                    # Find candidate B
                    # Must cover critical_mask
                    # Must not be in_sol
                    # To find candidates, intersect tset_to_blocks for all tsets in critical_mask
                    # This is expensive if critical_mask has many bits.
                    
                    # Get list of tsets in critical_mask
                    crit_ts_list = []
                    m = critical_mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        crit_ts_list.append(idx)
                        m ^= lsb
                        
                    if not crit_ts_list:
                        continue
                        
                    # Start with candidates of first tset
                    candidates = set(tset_to_blocks[crit_ts_list[0]])
                    # Intersect with others
                    for ts_idx in crit_ts_list[1:]:
                        if not candidates:
                            break
                        candidates &= set(tset_to_blocks[ts_idx])
                        
                    # Remove blocks already in sol
                    candidates -= set(sol)
                    
                    if candidates:
                        b_in = rng.choice(list(candidates))
                        # Perform swap
                        update_counts(b_out, -1)
                        in_sol[b_out] = False
                        update_counts(b_in, 1)
                        in_sol[b_in] = True
                        sol[pos] = b_in
                        found_swap = True
                        improved = True
                        break
                        
                if found_swap:
                    no_improve = 0
                    if len(sol) < best_size:
                        best_size = len(sol)
                        best_sol = list(sol)
                    continue
            else:
                break
                
            no_improve += 1
            
        return best_sol

    # Main Loop
    best_solution = None
    best_size = float('inf')
    
    # Run a few greedy restarts
    for seed in range(5):
        if elapsed() > deadline - 30:
            break
        sol = greedy_cover(seed)
        if sol:
            ls_sol = local_search_v2(sol, seed)
            if len(ls_sol) < best_size:
                best_size = len(ls_sol)
                best_solution = ls_sol
                
    if best_solution is None:
        # Fallback: return empty or raise
        return []
        
    return [list(all_blocks[i]) for i in best_solution]
