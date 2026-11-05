import itertools
import time
import random

def construct(v, k, t):
    all_points = range(v)
    
    def bits_to_mask(bits):
        m = 0
        for b in bits:
            m |= (1 << b)
        return m

    def mask_to_points(mask):
        return [i for i in range(v) if (mask >> i) & 1]

    # Pre-compute all t-subset masks
    all_t_masks = []
    for combo in itertools.combinations(all_points, t):
        all_t_masks.append(bits_to_mask(combo))
        
    # Initial state
    uncovered = set(all_t_masks)
    blocks = []
    coverage_count = {m: 0 for m in all_t_masks}

    def add_block(block_pts):
        nonlocal uncovered, blocks, coverage_count
        blocks.append(sorted(block_pts))
        for sub in itertools.combinations(block_pts, t):
            m = bits_to_mask(sub)
            coverage_count[m] += 1
            if coverage_count[m] == 1:
                if m in uncovered:
                    uncovered.remove(m)

    def remove_block(block_pts):
        nonlocal uncovered, blocks, coverage_count
        idx = blocks.index(sorted(block_pts))
        blocks.pop(idx)
        for sub in itertools.combinations(block_pts, t):
            m = bits_to_mask(sub)
            coverage_count[m] -= 1
            if coverage_count[m] == 0:
                uncovered.add(m)

    def prune_blocks():
        nonlocal uncovered, blocks, coverage_count
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                b = blocks[i]
                essential = False
                for sub in itertools.combinations(b, t):
                    m = bits_to_mask(sub)
                    if coverage_count[m] == 1:
                        essential = True
                        break
                if not essential:
                    remove_block(b)
                    changed = True

    # Phase 1: Deterministic Greedy Construction
    if k <= v:
        first_block = list(range(k))
        add_block(first_block)
        
        while uncovered:
            start_mask = min(uncovered)
            start_pts = mask_to_points(start_mask)
            
            block_pts = start_pts[:]
            
            while len(block_pts) < k:
                best_x = -1
                best_gain = -1
                
                t_minus_1_subs = list(itertools.combinations(block_pts, t-1))
                
                for x in range(v):
                    if x in block_pts:
                        continue
                    
                    gain = 0
                    for sub in t_minus_1_subs:
                        m = 0
                        for p in sub:
                            m |= (1 << p)
                        m |= (1 << x)
                        if m in uncovered:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        
                if best_x == -1:
                    for x in range(v):
                        if x not in block_pts:
                            best_x = x
                            break
                            
                block_pts.append(best_x)
            
            add_block(block_pts)
            
    prune_blocks()
    
    # Phase 2: Local Search (Hill Climbing)
    # We try to swap blocks to reduce the total count.
    # Move: Remove B1, Add B2 (covering new gaps), Prune.
    # If len(blocks) decreases, accept.
    
    best_blocks = blocks[:]
    best_count = len(best_blocks)
    
    # Pre-compute all possible k-subsets? C(19,8) = 75582.
    # This is small enough to pre-compute masks for all k-subsets?
    # C(19,8) is 75582.
    # For each k-subset, we can pre-compute the list of t-subsets it covers.
    # This will speed up the "gain" calculation significantly.
    
    all_k_subs = []
    k_sub_coverage = {}
    for combo in itertools.combinations(all_points, k):
        mask = bits_to_mask(combo)
        subs = [bits_to_mask(sub) for sub in itertools.combinations(combo, t)]
        all_k_subs.append(combo)
        k_sub_coverage[mask] = subs

    # To find the best replacement for a removed block, we need to know
    # which k-subsets cover the currently uncovered t-subsets.
    
    # We will perform a limited number of iterations.
    start_time = time.time()
    max_time = 110 # Leave 10s for overhead
    
    # We need a data structure to quickly find candidates.
    # For each t-mask, store the list of k-masks that cover it.
    # This is an inverted index.
    t_to_k = {m: [] for m in all_t_masks}
    for k_mask, subs in k_sub_coverage.items():
        for sub in subs:
            t_to_k[sub].append(k_mask)
            
    # Now, local search:
    # We iterate through blocks. For each block, try to remove it.
    # If it creates uncovered sets U, find the k-subset that covers the most U.
    # If that k-subset is not the one we removed, swap.
    # Then prune.
    
    improved = True
    while improved and (time.time() - start_time) < max_time:
        improved = False
        # Shuffle blocks to avoid bias
        # But we need to keep track of indices.
        # We'll just iterate through the current list.
        
        # To make it faster, we can just try a few random blocks per pass
        # or iterate all. With ~300 blocks, iterating all is fine.
        
        for i in range(len(best_blocks)):
            if (time.time() - start_time) > max_time:
                break
                
            b_remove = best_blocks[i]
            
            # Tentatively remove
            # We need to update state.
            # To avoid complex undo, we can simulate:
            # 1. Count coverage of b_remove's t-subsets.
            # 2. Identify which of those become uncovered (count was 1).
            
            uncovered_by_remove = []
            for sub in itertools.combinations(b_remove, t):
                m = bits_to_mask(sub)
                if coverage_count[m] == 1:
                    uncovered_by_remove.append(m)
            
            if not uncovered_by_remove:
                # Safe to remove
                # We need to actually remove it from the state
                # But we are working on best_blocks which is a copy?
                # No, we need to maintain the global state for the search.
                # Let's maintain `current_blocks` and `current_coverage` and `current_uncovered`.
                
                # This is getting complicated to manage state in the loop.
                # Let's refactor to use a state object or just careful in-place updates.
                
                pass # Placeholder for logic below

    # The above loop logic is messy. Let's implement a cleaner Local Search.
    
    # Reset state to the pruned greedy result
    current_blocks = best_blocks[:]
    # Rebuild coverage counts for current_blocks
    coverage_count = {m: 0 for m in all_t_masks}
    uncovered = set(all_t_masks)
    for b in current_blocks:
        for sub in itertools.combinations(b, t):
            m = bits_to_mask(sub)
            coverage_count[m] += 1
            if coverage_count[m] == 1:
                if m in uncovered:
                    uncovered.remove(m)
    
    # Local Search
    start_time = time.time()
    max_time = 115
    
    while (time.time() - start_time) < max_time:
        improved = False
        # Try to remove each block
        for i in range(len(current_blocks)):
            if (time.time() - start_time) > max_time:
                break
                
            b = current_blocks[i]
            
            # Check if b is essential
            # If not, remove it
            essential = False
            for sub in itertools.combinations(b, t):
                m = bits_to_mask(sub)
                if coverage_count[m] == 1:
                    essential = True
                    break
                    
            if not essential:
                # Remove
                for sub in itertools.combinations(b, t):
                    m = bits_to_mask(sub)
                    coverage_count[m] -= 1
                    if coverage_count[m] == 0:
                        uncovered.add(m)
                current_blocks.pop(i)
                improved = True
                break # Restart loop as indices changed
                
        if not improved:
            # Try Swap: Remove B, Add B'
            # Pick a random block to remove
            if current_blocks:
                idx = random.randint(0, len(current_blocks) - 1)
                b_remove = current_blocks[idx]
                
                # Find uncovered sets if b_remove is removed
                temp_uncovered = set()
                for sub in itertools.combinations(b_remove, t):
                    m = bits_to_mask(sub)
                    if coverage_count[m] == 1:
                        temp_uncovered.add(m)
                        
                if temp_uncovered:
                    # Find best replacement
                    # Candidates are k-subsets that cover any of temp_uncovered
                    # We can iterate through all k-subsets and score them
                    best_score = 0
                    best_block = None
                    
                    # Optimization: Only consider k-subsets that cover at least one temp_uncovered
                    # We can use the inverted index t_to_k
                    # But t_to_k is large.
                    # Let's just iterate all k_subs if count is small, or sample.
                    # C(19,8) = 75582.
                    # Scoring: count how many m in temp_uncovered are covered by k_sub.
                    
                    # To speed up, we can just check a random subset of k_subs?
                    # Or iterate all. 75k * (t-subsets in k) is small.
                    
                    for k_sub in all_k_subs:
                        # Check if it covers any temp_uncovered
                        # Pre-computed k_sub_coverage[k_mask] gives the list of t-masks
                        k_mask = bits_to_mask(k_sub)
                        subs_covered = k_sub_coverage[k_mask]
                        
                        # Count intersection with temp_uncovered
                        # temp_uncovered is small (max C(8,5)=56)
                        # subs_covered is 56.
                        # Intersection:
                        score = 0
                        for m in subs_covered:
                            if m in temp_uncovered:
                                score += 1
                                
                        if score > best_score:
                            best_score = score
                            best_block = k_sub
                            
                    if best_block and best_block != b_remove:
                        # Try swap
                        # 1. Remove b_remove
                        for sub in itertools.combinations(b_remove, t):
                            m = bits_to_mask(sub)
                            coverage_count[m] -= 1
                            if coverage_count[m] == 0:
                                uncovered.add(m)
                        current_blocks.pop(idx)
                        
                        # 2. Add best_block
                        for sub in itertools.combinations(best_block, t):
                            m = bits_to_mask(sub)
                            coverage_count[m] += 1
                            if coverage_count[m] == 1:
                                if m in uncovered:
                                    uncovered.remove(m)
                        current_blocks.append(best_block)
                        
                        # 3. Prune
                        # Check if any block is now redundant
                        changed = True
                        while changed:
                            changed = False
                            for j in range(len(current_blocks) - 1, -1, -1):
                                bj = current_blocks[j]
                                ess = False
                                for sub in itertools.combinations(bj, t):
                                    m = bits_to_mask(sub)
                                    if coverage_count[m] == 1:
                                        ess = True
                                        break
                                if not ess:
                                    for sub in itertools.combinations(bj, t):
                                        m = bits_to_mask(sub)
                                        coverage_count[m] -= 1
                                        if coverage_count[m] == 0:
                                            uncovered.add(m)
                                    current_blocks.pop(j)
                                    changed = True
                                    
                        improved = True
                        break # Restart loop

    return current_blocks
