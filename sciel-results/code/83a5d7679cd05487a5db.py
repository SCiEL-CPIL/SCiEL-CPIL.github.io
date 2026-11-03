import itertools
import time
import random
import sys
import numpy as np

def _deadline_ok(deadline):
    return time.time() < deadline

def construct(v, k, t):
    start = time.time()
    limit = 120.0
    deadline = start + limit - 5.0  # Safety margin
    
    # 1. Generate all t-subsets and assign indices
    t_subs = list(itertools.combinations(range(v), t))
    n_t = len(t_subs)
    sub_index = {sub: i for i, sub in enumerate(t_subs)}
    
    # 2. Generate all k-blocks and their coverage masks
    # Using bitmasks for fast intersection and gain calculation
    blocks = []
    block_masks = []
    
    for block in itertools.combinations(range(v), k):
        blocks.append(block)
        mask = 0
        for sub in itertools.combinations(block, t):
            mask |= (1 << sub_index[sub])
        block_masks.append(mask)
        
    if not _deadline_ok(deadline):
        return []

    n_blocks = len(blocks)
    
    # Build sub_to_blocks: for each t-subset index, list of block indices that cover it
    sub_to_blocks = [[] for _ in range(n_t)]
    for i, mask in enumerate(block_masks):
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            sub_to_blocks[idx].append(i)
            m ^= lsb
            
    if not _deadline_ok(deadline):
        return []
    
    # Helper: Greedy Set Cover using bitmasks
    def _greedy_cover(rng=None):
        if rng is None:
            rng = random.Random()
            
        uncovered_mask = (1 << n_t) - 1
        selected = []
        
        while uncovered_mask:
            if not _deadline_ok(deadline):
                break
                
            best_block_idx = -1
            best_gain = 0
            
            # Scan all blocks for max gain
            # Optimization: If we have many blocks, this is the bottleneck.
            # 75k blocks * ~300 iterations = 22.5M bit_count ops.
            # bit_count is very fast in CPython 3.10+.
            
            for i in range(n_blocks):
                gain = (uncovered_mask & block_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = i
                elif gain == best_gain and gain > 0 and rng.random() < 0.1:
                    # Random tie breaking to allow restarts to differ
                    best_block_idx = i
                    
            if best_block_idx == -1:
                break
                
            selected.append(best_block_idx)
            uncovered_mask &= ~block_masks[best_block_idx]
            
        return selected

    # Helper: Prune redundant blocks
    def _prune_redundant(selected):
        if not selected:
            return selected
            
        # Calculate coverage counts
        counts = [0] * n_t
        for i in selected:
            m = block_masks[i]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                m ^= lsb
                
        changed = True
        while changed and _deadline_ok(deadline):
            changed = False
            new_selected = []
            for i in selected:
                # Check if block i is redundant
                redundant = True
                m = block_masks[i]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if counts[idx] <= 1:
                        redundant = False
                        break
                    m ^= lsb
                    
                if redundant:
                    # Remove it
                    m = block_masks[i]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        m ^= lsb
                    changed = True
                else:
                    new_selected.append(i)
            selected = new_selected
            
        return selected

    # Helper: Local Search (2-for-1 swap)
    def _local_search(selected):
        if len(selected) < 2:
            return selected
            
        selected_set = set(selected)
        
        # Calculate counts
        counts = [0] * n_t
        for i in selected:
            m = block_masks[i]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                m ^= lsb
                
        improved = True
        while improved and _deadline_ok(deadline):
            improved = False
            # Try to find a 2-for-1 swap
            # Iterate over pairs in selected
            # To avoid O(N^2) explosion if N is large, we can limit attempts or shuffle
            # N is ~300, N^2 is 90k. This is okay for a few iterations.
            
            # We need to check pairs. If we find one, apply it and restart counts.
            # To be efficient, we can just do one pass and pick the first valid swap?
            # Or collect all valid swaps and pick the best?
            # Let's just find one valid swap to reduce complexity.
            
            found_swap = False
            # Shuffle to avoid bias?
            pairs = list(itertools.combinations(selected, 2))
            random.shuffle(pairs)
            
            for a, b in pairs:
                if not _deadline_ok(deadline):
                    break
                    
                # Calculate subsets that are ONLY covered by a or b (i.e., count in S \ {a,b} is 0)
                # s is critical if (s in Cover(a) or s in Cover(b)) and counts[s] - (1 if s in Cover(a) else 0) - (1 if s in Cover(b) else 0) == 0
                # Actually, counts[s] includes a and b.
                # Let count_rest[s] = counts[s] - (1 if s in Cover(a) else 0) - (1 if s in Cover(b) else 0)
                # We need count_rest[s] == 0 for all s in R, where R = Cover(a) U Cover(b)
                
                # R is the union of covered subsets of a and b.
                # We need to find a block c such that R is subset of Cover(c).
                
                # First, determine R.
                # R is small? No, R can be up to 56+56=112 subsets.
                # But most subsets in a and b are covered by other blocks too.
                # We only care about subsets s where count_rest[s] == 0.
                
                critical_subs = []
                mask_a = block_masks[a]
                mask_b = block_masks[b]
                mask_ab = mask_a | mask_b
                
                # Iterate through subsets in mask_ab
                m = mask_ab
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    # Check if this subset is covered by a or b
                    # It is, by definition of mask_ab.
                    # Check count_rest
                    in_a = 1 if (mask_a & lsb) else 0
                    in_b = 1 if (mask_b & lsb) else 0
                    count_rest = counts[idx] - in_a - in_b
                    if count_rest == 0:
                        critical_subs.append(idx)
                    m ^= lsb
                    
                if not critical_subs:
                    # This means a and b are redundant together? 
                    # If no subset is uniquely covered by a or b, then removing one might make the other redundant?
                    # Actually, if critical_subs is empty, it means every subset in A U B is covered by S \ {A,B}.
                    # Then we can remove A (and then B might become redundant).
                    # Pruning should have caught this if A was redundant on its own.
                    # But maybe A is not redundant alone, but A and B together are?
                    # If A has a unique subset s, then s is in critical_subs (since count_rest[s] would be 0).
                    # So if critical_subs is empty, A has no unique subsets? Then A is redundant.
                    # Pruning should have removed it.
                    # So this case likely doesn't occur after pruning.
                    continue
                
                # Now we need a block c that covers ALL subsets in critical_subs.
                # c must not be a or b.
                # c must not be in selected_set (otherwise we are just swapping with existing block, which doesn't reduce size).
                
                # Find intersection of sub_to_blocks for all critical_subs
                # Start with the one with fewest blocks
                min_len = float('inf')
                target_sub = -1
                for s in critical_subs:
                    if len(sub_to_blocks[s]) < min_len:
                        min_len = len(sub_to_blocks[s])
                        target_sub = s
                        
                candidates = set(sub_to_blocks[target_sub])
                
                for s in critical_subs:
                    if s != target_sub:
                        candidates &= set(sub_to_blocks[s])
                        
                # Filter out a, b and any block already in selected
                candidates.discard(a)
                candidates.discard(b)
                candidates -= selected_set
                
                if candidates:
                    # Pick one candidate (random or first)
                    c = next(iter(candidates))
                    
                    # Perform swap
                    selected_set.discard(a)
                    selected_set.discard(b)
                    selected_set.add(c)
                    
                    # Update counts
                    # Remove a and b
                    for idx in critical_subs:
                        # We know count_rest was 0, so counts[idx] was in_a + in_b.
                        # We need to update counts for ALL subsets in a and b, not just critical ones.
                        pass # This is inefficient. Let's just recalculate counts.
                    
                    # Recalculate counts is O(N * 56). N=300. 16k ops. Fast enough.
                    counts = [0] * n_t
                    for i in selected_set:
                        m = block_masks[i]
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            counts[idx] += 1
                            m ^= lsb
                            
                    improved = True
                    found_swap = True
                    break # Restart pair search with new counts
                    
        return list(selected_set)

    # Main loop: Multiple restarts
    best_result = None
    best_size = float('inf')
    
    rng = random.Random(42)
    
    # Estimate time for one full cycle (greedy + prune + local)
    # Greedy: ~1-2s
    # Prune: ~0.1s
    # Local: ~1-5s depending on swaps
    # Total: ~5s per restart?
    # We have 115s. So ~20 restarts possible.
    
    n_restarts = 15
    
    for restart in range(n_restarts):
        if not _deadline_ok(deadline):
            break
            
        # 1. Greedy
        selected = _greedy_cover(rng)
        
        if not selected:
            continue
            
        # 2. Prune
        selected = _prune_redundant(selected)
        
        # 3. Local Search
        if _deadline_ok(deadline):
            selected = _local_search(selected)
            
        if len(selected) < best_size:
            best_size = len(selected)
            best_result = selected
            
        # Optional: If we hit the theoretical lower bound or a very good size, stop?
        # Lower bound is 264. If we get < 270, it's great.
        if best_size <= 265:
            break

    if best_result is None:
        # Fallback
        selected = _greedy_cover()
        selected = _prune_redundant(selected)
        best_result = selected
        
    result = [list(blocks[i]) for i in best_result]
    
    # Final Verification
    if _deadline_ok(deadline - 2.0):
        check_counts = [0] * n_t
        for i in best_result:
            m = block_masks[i]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                check_counts[idx] += 1
                m ^= lsb
        if any(c == 0 for c in check_counts):
            # Invalid, try a fresh simple greedy
            selected = _greedy_cover()
            selected = _prune_redundant(selected)
            result = [list(blocks[i]) for i in selected]
            
    return result
