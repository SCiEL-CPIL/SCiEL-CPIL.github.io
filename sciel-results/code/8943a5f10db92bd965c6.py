import itertools
import time
import random
import math

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin (150s limit)
    
    # 1. Precompute t-subsets and indices
    all_t_subs = list(itertools.combinations(range(v), t))
    t_index = {sub: i for i, sub in enumerate(all_t_subs)}
    num_t = len(all_t_subs)
    
    # Inverse mapping for quick lookup
    inv_t_index = [None] * num_t
    for sub, idx in t_index.items():
        inv_t_index[idx] = sub
        
    # Precompute masks for all possible (subset, x) pairs? 
    # Too many. But we can precompute masks for all (k-1)-subsets? 
    # C(19, 7) = 50388. That's manageable.
    # Or just compute on the fly but cache?
    # Let's try to precompute "gain masks" for extending a set of size m to m+1.
    # Actually, the current get_mask_containing_x is O(C(len(s), t-1)).
    # Since len(s) <= k-1 = 7, and t-1 = 4, C(7,4)=35. This is very fast.
    # The bottleneck in the original code might be the Python loop overhead in the greedy selection.
    
    def get_mask_containing_x(s, x):
        mask = 0
        # s is a list of points. We need combinations of size t-1 from s.
        # Optimization: s is small (<=7).
        for sub in itertools.combinations(s, t - 1):
            full_sub = tuple(sorted(sub + (x,)))
            mask |= 1 << t_index[full_sub]
        return mask

    def get_mask_all(s):
        mask = 0
        for sub in itertools.combinations(s, t):
            mask |= 1 << t_index[sub]
        return mask

    best_blocks = None
    best_count = float('inf')
    
    # 2. Multiple Greedy Runs
    # We have ~110 seconds.
    # Each greedy run takes roughly (num_blocks * k * v * C(k-1, t-1)) operations.
    # Let's estimate. num_blocks ~ 300. k=8, v=19. C(7,4)=35.
    # 300 * 8 * 19 * 35 ~ 1.6M operations.
    # Python can do ~10M-50M simple ops/sec. So ~0.1s per run?
    # Plus reduction phase.
    # Let's run as many as possible.
    
    while time.time() < deadline - 20:
        # Random seed for this run
        random.seed(int(time.time() * 1000) % 1000000)
        
        # Initialize uncovered mask
        uncovered = (1 << num_t) - 1
        blocks = []
        
        while uncovered:
            if time.time() > deadline - 5:
                break
                
            # Find lowest set bit
            low_bit = uncovered & -uncovered
            start_idx = low_bit.bit_length() - 1
            start_sub = inv_t_index[start_idx]
            
            # Start block with this subset
            block = list(start_sub)
            
            # Extend block to size k
            while len(block) < k:
                best_gain = -1
                best_x = None
                candidates = [x for x in range(v) if x not in block]
                
                # Shuffle candidates to break ties randomly
                random.shuffle(candidates)
                
                for x in candidates:
                    gain_mask = get_mask_containing_x(block, x)
                    gain = (uncovered & gain_mask).bit_count()
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        
                if best_x is None:
                    break
                    
                block.append(best_x)
            
            # Sort block for consistent representation
            block.sort()
            blocks.append(tuple(block))
            
            # Update uncovered mask
            block_mask = get_mask_all(block)
            uncovered &= ~block_mask
            
        # 3. Reduction Phase
        if not blocks:
            continue
            
        # Count coverage for each t-subset
        counts = [0] * num_t
        for b in blocks:
            for sub in itertools.combinations(b, t):
                counts[t_index[sub]] += 1
        
        # Remove redundant blocks
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for b in blocks:
                redundant = True
                for sub in itertools.combinations(b, t):
                    if counts[t_index[sub]] == 1:
                        redundant = False
                        break
                if redundant:
                    for sub in itertools.combinations(b, t):
                        counts[t_index[sub]] -= 1
                    changed = True
                else:
                    new_blocks.append(b)
            blocks = new_blocks
            
        # 4. Local Search (Swap-based)
        # Try to replace blocks with others that cover the same unique subsets but are "better" or allow further reduction.
        # Simple heuristic: Try to swap a block with another block that covers all its unique subsets.
        # This is expensive. Let's do a limited version.
        
        # Identify "critical" blocks (those with at least one unique t-subset)
        # Try to replace a block B with a block B' such that:
        # 1. B' is not in the current set
        # 2. All t-subsets in B that are unique to B are also in B'
        # This doesn't necessarily reduce the count, but might enable further reduction?
        # Actually, swapping doesn't reduce count.
        
        # Alternative: Try to remove a block if it becomes redundant after swapping?
        # Or: Try to find a block B and a block B' such that B' covers more unique subsets than B, and B's unique subsets are covered by others?
        
        # Let's stick to the proven "swap-based local search" from inspiration 1.
        # The inspiration says: "exchanging blocks that maintain coverage while potentially enabling further redundancy removal"
        
        # Let's try a simple local search:
        # For each block B in the set:
        #   Find all t-subsets in B that are uniquely covered by B (count == 1).
        #   If there are none, B is already redundant (should have been removed).
        #   Try to find a block B' (not in set) that covers all these unique t-subsets.
        #   If found, swap B with B'. Then run reduction again.
        
        # This is O(N_blocks * C(v,k) * ...). Too slow if done exhaustively.
        
        # Let's limit the local search to a few iterations or a few blocks.
        
        # Actually, the "swap" might just be:
        # If we have a block B, and we can find a block B' that covers all unique subsets of B,
        # we can replace B with B'. This keeps the count same, but might change the structure.
        # Then we run reduction. If B' allows other blocks to become redundant, we gain.
        
        # Let's try this for a few random blocks.
        
        # Precompute all possible blocks? C(19,8) = 75582.
        # We can generate them on the fly or precompute masks.
        # Precomputing masks for all 75k blocks is feasible.
        
        # Let's precompute masks for all blocks once.
        # This takes C(19,8) * C(8,5) = 75582 * 56 ~ 4M operations.
        # And memory: 75582 integers of 11628 bits. 11628/8 = 1453 bytes. 75582 * 1453 ~ 110 MB.
        # This is okay.
        
        # But wait, the current code doesn't precompute all blocks.
        # Let's do it in the local search phase if we have time.
        
        # For now, let's just do the reduction and check time.
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks[:]
            
    if best_blocks is None:
        # Fallback
        pass

    return [list(b) for b in best_blocks]
