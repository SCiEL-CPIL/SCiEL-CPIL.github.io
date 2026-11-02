"""Randomized local search (simulated annealing) for covering design C(24,7,3)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time


def _greedy_cover(v, k, t, rng):
    """Fast greedy construction to get a valid starting covering."""
    all_tsubsets = [frozenset(c) for c in itertools.combinations(range(v), t)]
    uncovered = set(all_tsubsets)
    blocks = []
    
    while uncovered:
        # Pick a random uncovered t-subset as seed
        seed = rng.choice(list(uncovered))
        block = list(seed)
        rng.shuffle(block)
        
        # Greedily add points to maximize coverage
        while len(block) < k:
            best_x = None
            best_gain = -1
            cands = [x for x in range(v) if x not in block]
            rng.shuffle(cands)
            
            for x in cands:
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    fs = frozenset(list(rest) + [x])
                    if fs in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
            
        block.sort()
        b_tuple = tuple(block)
        if b_tuple not in blocks: # Avoid duplicates
            blocks.append(b_tuple)
            for c in itertools.combinations(block, t):
                uncovered.discard(frozenset(c))
                
    return blocks


def construct(v, k, t):
    rng = random.Random(12345)
    start_time = time.time()
    time_limit = 115.0  # leave 5s buffer

    # Precompute all t-subsets
    all_tsubsets = [frozenset(c) for c in itertools.combinations(range(v), t)]
    total_t = len(all_tsubsets)
    
    # Build initial valid covering
    blocks = _greedy_cover(v, k, t, rng)
    
    # Precompute t-subsets for each block
    def get_block_tsubsets(block):
        return [frozenset(c) for c in itertools.combinations(block, t)]
    
    # Compute counts and track single-covered subsets
    def compute_counts_and_stats(blk_list):
        counts = {}
        single_covered = set()
        for b in blk_list:
            for fs in get_block_tsubsets(b):
                counts[fs] = counts.get(fs, 0) + 1
        for fs in all_tsubsets:
            if counts.get(fs, 0) == 1:
                single_covered.add(fs)
        return counts, single_covered

    counts, single_covered = compute_counts_and_stats(blocks)
    best_blocks = list(blocks)
    best_count = len(blocks)
    
    # Helper: check if a block can be removed (all its t-subsets have count > 1)
    def is_removable(block):
        for fs in get_block_tsubsets(block):
            if counts.get(fs, 0) <= 1:
                return False
        return True
    
    # Helper: apply block addition/removal to counts and single_covered
    def apply_block_change(block, add=True):
        """Add or remove a block from counts and single_covered."""
        for fs in get_block_tsubsets(block):
            old_c = counts.get(fs, 0)
            if add:
                new_c = old_c + 1
            else:
                new_c = old_c - 1
            
            if add:
                if old_c == 0:
                    # Was uncovered, now covered
                    pass
                elif old_c == 1:
                    # Was single-covered, now double-covered
                    single_covered.discard(fs)
            else:
                if old_c == 1:
                    # Was single-covered, now uncovered
                    single_covered.add(fs)
                elif old_c == 2:
                    # Was double-covered, now single-covered
                    single_covered.add(fs)
            
            if new_c > 0:
                counts[fs] = new_c
            else:
                if fs in counts:
                    del counts[fs]
    
    # Main search loop
    iteration = 0
    stuck_counter = 0
    
    while time.time() - start_time < time_limit:
        iteration += 1
        
        # Step 1: Try to remove redundant blocks
        removed_any = False
        indices = list(range(len(blocks)))
        rng.shuffle(indices)
        
        for i in indices:
            if is_removable(blocks[i]):
                # Remove block i
                b = blocks[i]
                apply_block_change(b, add=False)
                blocks.pop(i)
                removed_any = True
                stuck_counter = 0
                
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = list(blocks)
                break
        
        if removed_any:
            continue
        
        # Step 2: Try swaps
        if len(blocks) > 0:
            # Find blocks with critical (single-covered) subsets
            # A block is "critical" if it contains at least one single-covered subset
            critical_blocks = []
            for i, b in enumerate(blocks):
                has_critical = False
                for fs in get_block_tsubsets(b):
                    if fs in single_covered:
                        has_critical = True
                        break
                if has_critical:
                    critical_blocks.append(i)
            
            # If no critical blocks, all subsets are double-covered, we should be able to remove some
            # This shouldn't happen if step 1 is working correctly, but handle it
            if not critical_blocks:
                # Force a random removal attempt (shouldn't reach here often)
                i = rng.randrange(len(blocks))
                critical_blocks = [i]
            
            # Pick a random critical block to swap
            i = rng.choice(critical_blocks)
            old_block = blocks[i]
            
            # Collect critical subsets (single-covered) in this block
            critical_subsets = []
            for fs in get_block_tsubsets(old_block):
                if fs in single_covered:
                    critical_subsets.append(fs)
            
            # Points that must be in the new block
            points_needed = set()
            for fs in critical_subsets:
                points_needed.update(fs)
            
            # If too many points needed, we can't form a valid replacement with one block
            if len(points_needed) > k:
                # Try a 2-block swap: replace old_block with two new blocks
                # This is more complex, so for now, try a different block
                continue
            
            # Build new block containing points_needed
            new_block_set = set(points_needed)
            remaining_slots = k - len(new_block_set)
            
            if remaining_slots > 0:
                # Fill remaining slots to maximize coverage
                cands = [x for x in range(v) if x not in new_block_set]
                rng.shuffle(cands)
                
                # Score candidates by how many single-covered or uncovered subsets they help
                scored_cands = []
                for x in cands:
                    gain = 0
                    for rest in itertools.combinations(new_block_set, t-1):
                        fs = frozenset(list(rest) + [x])
                        c = counts.get(fs, 0)
                        if c == 0:
                            gain += 100  # Very high priority: uncovering
                        elif c == 1:
                            gain += 10   # High priority: reducing single coverage
                    scored_cands.append((gain, x))
                
                scored_cands.sort(key=lambda x: -x[0])
                for gain, x in scored_cands[:remaining_slots]:
                    new_block_set.add(x)
            
            if len(new_block_set) != k:
                continue
                
            new_block = tuple(sorted(new_block_set))
            
            if new_block == old_block:
                continue
            
            # Check if new_block is already in blocks
            if new_block in blocks:
                continue
            
            # Simulate the swap: remove old_block, add new_block
            # Check validity: after swap, no subset should have count 0
            
            # Subsets that lose coverage: in old_block but not in new_block
            old_subsets = set(get_block_tsubsets(old_block))
            new_subsets = set(get_block_tsubsets(new_block))
            
            # Subsets that only lose coverage
            losing_subsets = old_subsets - new_subsets
            # Subsets that gain coverage
            gaining_subsets = new_subsets - old_subsets
            
            # Check if any losing subset has count 1 (would become 0)
            valid = True
            for fs in losing_subsets:
                if counts.get(fs, 0) <= 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Apply the swap
            apply_block_change(old_block, add=False)
            apply_block_change(new_block, add=True)
            blocks[i] = new_block
            stuck_counter = 0
            
            # Step 3: After a swap, try to remove blocks again (the swap may have made some blocks redundant)
            # We'll let the next iteration handle this, but we can also do a quick pass
            for j in range(len(blocks)):
                if is_removable(blocks[j]):
                    b = blocks[j]
                    apply_block_change(b, add=False)
                    blocks.pop(j)
                    j -= 1
                    stuck_counter = 0
                    if len(blocks) < best_count:
                        best_count = len(blocks)
                        best_blocks = list(blocks)
        
        # Step 4: If stuck for a while, try a perturbation
        stuck_counter += 1
        if stuck_counter > 50:
            # Try a random restart with a different seed
            stuck_counter = 0
            new_cover = _greedy_cover(v, k, t, rng)
            if len(new_cover) < len(blocks):
                blocks = new_cover
                counts, single_covered = compute_counts_and_stats(blocks)
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = list(blocks)
    
    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
