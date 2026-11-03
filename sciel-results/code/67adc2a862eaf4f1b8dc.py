"""Randomized local search (simulated annealing) for covering design C(24,7,3)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time


def _greedy_init(v, k, t, rng):
    """
    Initialize a valid covering using a greedy set cover approach.
    Returns a list of blocks (tuples of sorted ints).
    """
    all_tsubsets = [frozenset(c) for c in itertools.combinations(range(v), t)]
    uncovered = set(all_tsubsets)
    blocks = []
    
    # Precompute all 3-subsets of each potential block for fast lookup
    # For v=24, k=7, we generate blocks on the fly
    
    while uncovered:
        # Pick a random uncovered 3-subset to start a new block
        seed = rng.choice(list(uncovered))
        block_set = set(seed)
        
        # Greedily expand the block to size k to maximize coverage of uncovered subsets
        while len(block_set) < k:
            best_point = None
            max_gain = -1
            candidates = [x for x in range(v) if x not in block_set]
            rng.shuffle(candidates)
            
            for x in candidates:
                # Calculate gain: how many new uncovered 3-subsets are formed
                gain = 0
                # A new 3-subset containing x is formed by x and 2 points from block_set
                for pair in itertools.combinations(sorted(block_set), t - 1):
                    sub = frozenset(list(pair) + [x])
                    if sub in uncovered:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_point = x
            
            if best_point is None:
                break
            block_set.add(best_point)
        
        # Sort and add block
        block_tuple = tuple(sorted(block_set))
        if block_tuple not in blocks:
            blocks.append(block_tuple)
            # Update uncovered set
            for sub in itertools.combinations(block_tuple, t):
                uncovered.discard(frozenset(sub))
                
    return blocks


def construct(v, k, t):
    rng = random.Random(42)
    start_time = time.time()
    time_limit = 145.0  # Leave 5s buffer before 150s kill
    
    # 1. Initialization
    blocks = _greedy_init(v, k, t, rng)
    best_blocks = list(blocks)
    best_size = len(blocks)
    
    # Precompute all 3-subsets as frozensets for quick lookup
    all_tsubs = [frozenset(c) for c in itertools.combinations(range(v), t)]
    total_t = len(all_tsubs)
    
    def get_counts(blk_list):
        counts = {}
        for b in blk_list:
            for sub in itertools.combinations(b, t):
                f = frozenset(sub)
                counts[f] = counts.get(f, 0) + 1
        return counts

    counts = get_counts(blocks)
    
    def apply_change(old_block, new_block):
        """
        Replace old_block with new_block in the current state.
        Updates counts dictionary.
        """
        if old_block is not None:
            for sub in itertools.combinations(old_block, t):
                f = frozenset(sub)
                counts[f] -= 1
                if counts[f] == 0:
                    del counts[f]
        
        if new_block is not None:
            for sub in itertools.combinations(new_block, t):
                f = frozenset(sub)
                counts[f] = counts.get(f, 0) + 1
                
    def is_valid():
        return len(counts) == total_t
    
    # 2. Local Search Loop
    consecutive_no_improve = 0
    max_no_improve = 50
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Phase 1: Remove redundant blocks
        i = 0
        while i < len(blocks):
            b = blocks[i]
            redundant = True
            for sub in itertools.combinations(b, t):
                f = frozenset(sub)
                if counts.get(f, 0) <= 1:
                    redundant = False
                    break
            if redundant:
                # Remove block
                apply_change(b, None)
                blocks.pop(i)
                improved = True
                if len(blocks) < best_size:
                    best_size = len(blocks)
                    best_blocks = list(blocks)
                # Restart removal phase as counts changed
                i = 0
            else:
                i += 1
        
        # Phase 2: Swap blocks - try to replace a block with a better one
        swapped = False
        for i in range(len(blocks)):
            if time.time() - start_time >= time_limit:
                break
            b = blocks[i]
            critical = []
            for sub in itertools.combinations(b, t):
                f = frozenset(sub)
                if counts.get(f, 0) == 1:
                    critical.append(f)
            
            if not critical:
                continue
            
            # Find points that must be in the new block
            points_needed = set()
            for f in critical:
                points_needed.update(f)
            
            if len(points_needed) > k:
                continue
            
            # Try multiple random completions
            remaining_points = [x for x in range(v) if x not in points_needed]
            rng.shuffle(remaining_points)
            
            # Try up to 20 random completions
            found_swap = False
            for attempt in range(20):
                if attempt > 0:
                    rng.shuffle(remaining_points)
                
                new_block_set = set(points_needed)
                for x in remaining_points[:k - len(points_needed)]:
                    new_block_set.add(x)
                
                new_block = tuple(sorted(new_block_set))
                
                if new_block == b:
                    continue
                
                # Check if new_block covers all critical subsets
                # Since points_needed is union of critical, and new_block contains points_needed,
                # all critical subsets are covered by new_block.
                
                # Check if new_block is already in blocks
                if new_block in blocks:
                    # Just remove b
                    apply_change(b, None)
                    blocks.pop(i)
                    improved = True
                    swapped = True
                    found_swap = True
                    if len(blocks) < best_size:
                        best_size = len(blocks)
                        best_blocks = list(blocks)
                    break
                else:
                    # Perform swap
                    apply_change(b, new_block)
                    blocks[i] = new_block
                    improved = True
                    swapped = True
                    found_swap = True
                    break
            
            if found_swap:
                break
        
        if improved:
            consecutive_no_improve = 0
            continue
        
        # Phase 3: Try 2-block replacement (replace 2 blocks with 1)
        # This is more complex but can yield significant improvements
        if len(blocks) > 1 and time.time() - start_time < time_limit - 10:
            two_swap_done = False
            for i in range(len(blocks)):
                if time.time() - start_time >= time_limit:
                    break
                b1 = blocks[i]
                for j in range(i + 1, len(blocks)):
                    b2 = blocks[j]
                    
                    # Find critical subsets for both blocks
                    critical1 = []
                    for sub in itertools.combinations(b1, t):
                        f = frozenset(sub)
                        if counts.get(f, 0) == 1:
                            critical1.append(f)
                    
                    critical2 = []
                    for sub in itertools.combinations(b2, t):
                        f = frozenset(sub)
                        if counts.get(f, 0) == 1:
                            critical2.append(f)
                    
                    if not critical1 and not critical2:
                        # Both blocks are redundant - should have been removed in Phase 1
                        # But just in case, remove both
                        apply_change(b1, None)
                        apply_change(b2, None)
                        blocks.pop(j)
                        blocks.pop(i)
                        improved = True
                        two_swap_done = True
                        if len(blocks) < best_size:
                            best_size = len(blocks)
                            best_blocks = list(blocks)
                        break
                    
                    # Union of all critical points
                    points_needed = set()
                    for f in critical1:
                        points_needed.update(f)
                    for f in critical2:
                        points_needed.update(f)
                    
                    if len(points_needed) > k:
                        continue
                    
                    # Try to form a single block containing all needed points
                    remaining_points = [x for x in range(v) if x not in points_needed]
                    rng.shuffle(remaining_points)
                    
                    # Try a few completions
                    for _ in range(5):
                        new_block_set = set(points_needed)
                        for x in remaining_points[:k - len(points_needed)]:
                            new_block_set.add(x)
                        
                        new_block = tuple(sorted(new_block_set))
                        
                        # Check if new_block covers all critical subsets
                        # All critical points are in points_needed, which is in new_block
                        # So all critical subsets are covered.
                        
                        # Check if new_block is already in blocks (excluding b1 and b2)
                        if new_block in blocks and new_block != b1 and new_block != b2:
                            # Just remove both b1 and b2
                            apply_change(b1, None)
                            apply_change(b2, None)
                            blocks.pop(j)
                            blocks.pop(i)
                            improved = True
                            two_swap_done = True
                            if len(blocks) < best_size:
                                best_size = len(blocks)
                                best_blocks = list(blocks)
                            break
                        elif new_block == b1 or new_block == b2:
                            # Remove the other block
                            if new_block == b1:
                                apply_change(b2, None)
                                blocks.pop(j)
                            else:
                                apply_change(b1, None)
                                blocks.pop(i)
                            improved = True
                            two_swap_done = True
                            if len(blocks) < best_size:
                                best_size = len(blocks)
                                best_blocks = list(blocks)
                            break
                        else:
                            # Replace both with new_block
                            apply_change(b1, None)
                            apply_change(b2, None)
                            blocks.pop(j)
                            blocks.pop(i)
                            blocks.insert(i, new_block)
                            apply_change(None, new_block)
                            improved = True
                            two_swap_done = True
                            if len(blocks) < best_size:
                                best_size = len(blocks)
                                best_blocks = list(blocks)
                            break
                    
                    if two_swap_done:
                        break
                
                if two_swap_done:
                    break
            
            if two_swap_done:
                consecutive_no_improve = 0
                continue
        
        # Phase 4: Perturbation (Kick)
        # If stuck, remove a random block and cover the holes
        if len(blocks) > 0 and time.time() - start_time < time_limit - 5:
            i = rng.randrange(len(blocks))
            b = blocks[i]
            
            # Save state
            old_blocks = list(blocks)
            old_counts = dict(counts)
            
            # Remove block
            apply_change(b, None)
            blocks.pop(i)
            
            # Find uncovered subsets
            uncovered = []
            for f in all_tsubs:
                if counts.get(f, 0) == 0:
                    uncovered.append(f)
            
            # Greedily cover uncovered subsets
            new_blocks_added = 0
            while uncovered and time.time() - start_time < time_limit - 2:
                # Pick a random uncovered subset
                seed = rng.choice(uncovered)
                block_set = set(seed)
                
                while len(block_set) < k:
                    best_point = None
                    max_gain = -1
                    candidates = [x for x in range(v) if x not in block_set]
                    rng.shuffle(candidates)
                    
                    for x in candidates:
                        gain = 0
                        for pair in itertools.combinations(sorted(block_set), t - 1):
                            sub = frozenset(list(pair) + [x])
                            if sub in uncovered:
                                gain += 1
                        if gain > max_gain:
                            max_gain = gain
                            best_point = x
                    
                    if best_point is None:
                        break
                    block_set.add(best_point)
                
                new_block = tuple(sorted(block_set))
                
                # Check if new_block is already in blocks
                if new_block in blocks:
                    # It's already there, so it covers these subsets?
                    for sub in itertools.combinations(new_block, t):
                        uncovered.remove(frozenset(sub))
                    continue
                
                # Add new block
                blocks.append(new_block)
                apply_change(None, new_block)
                new_blocks_added += 1
                
                # Update uncovered list
                new_uncovered = []
                for f in uncovered:
                    if all(p in new_block for p in f):
                        pass
                    else:
                        new_uncovered.append(f)
                uncovered = new_uncovered
            
            # If we added more blocks than we removed, size increased.
            # If the perturbation didn't lead to an improvement immediately,
            # we might be in a worse state.
            
            # Mark as improved to restart loop
            consecutive_no_improve += 1
            improved = True

        if not improved:
            consecutive_no_improve += 1
            if consecutive_no_improve > max_no_improve:
                # Reset and try a fresh perturbation
                consecutive_no_improve = 0
                # Revert to best and try again
                blocks = list(best_blocks)
                counts = get_counts(blocks)

    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
