"""Program for covering design C(19,8,5) using randomized greedy with local search."""
import itertools
import random
import time


def construct(v, k, t):
    """
    Construct a covering design C(v, k, t).
    
    Uses a hybrid approach:
    1. Start with a greedy covering.
    2. Try to reduce the number of blocks via local search (block replacement).
    3. Use randomization to escape local optima.
    
    Returns a list of blocks (each block is a sorted list of k distinct integers in 0..v-1).
    """
    all_points = list(range(v))
    all_t_subs = list(itertools.combinations(all_points, t))
    
    # Precompute for each t-subset, which points can be added to form larger subsets
    # Actually, let's think differently.
    
    # Step 1: Greedy construction
    def greedy_cover(v, k, t, seed=42):
        rng = random.Random(seed)
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        
        while uncovered:
            # Pick a random uncovered t-subset to start the block
            start_t = rng.choice(list(uncovered))
            block = set(start_t)
            
            while len(block) < k:
                # Find the point that covers the most uncovered t-subsets
                best_x = None
                best_gain = -1
                
                candidates = [x for x in range(v) if x not in block]
                # Shuffle candidates for tie-breaking diversity
                rng.shuffle(candidates)
                
                for x in candidates:
                    gain = 0
                    # Count how many uncovered t-subsets would be covered by adding x
                    for rest in itertools.combinations(block, t - 1):
                        new_t = tuple(sorted(rest + (x,)))
                        if new_t in uncovered:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    elif gain == best_gain and best_x is not None:
                        # Tie-break: prefer point that appears in more uncovered sets
                        pass  # Keep first for simplicity, or could add secondary criterion
                
                if best_x is None:
                    break
                block.add(best_x)
            
            if len(block) >= t:
                block_list = sorted(block)
                blocks.append(block_list)
                # Remove all covered t-subsets
                for ts in itertools.combinations(block_list, t):
                    uncovered.discard(ts)
        
        return blocks
    
    # Step 2: Local search to reduce blocks
    def local_search(blocks, v, k, t, max_iterations=5000, timeout=90):
        """
        Try to remove or replace blocks to reduce total count.
        A block can be removed if all its t-subsets are covered by other blocks.
        A block can be replaced by a different k-subset if the new block covers
        at least as many uncovered t-subsets and allows removal of some other block.
        """
        start_time = time.time()
        blocks = [list(b) for b in blocks]
        
        # Build coverage count for each t-subset
        t_sub_count = {}
        for b in blocks:
            for ts in itertools.combinations(b, t):
                t_sub_count[ts] = t_sub_count.get(ts, 0) + 1
        
        # Try removing blocks that are redundant
        changed = True
        while changed and time.time() - start_time < timeout:
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                # Check if block i is redundant
                redundant = True
                for ts in itertools.combinations(blocks[i], t):
                    if t_sub_count.get(ts, 0) <= 1:
                        redundant = False
                        break
                if redundant:
                    # Remove block i
                    for ts in itertools.combinations(blocks[i], t):
                        t_sub_count[ts] -= 1
                    blocks.pop(i)
                    changed = True
                    break
        
        # Try replacing blocks
        # For each block, try to find a replacement that covers more uncovered t-subsets
        for _ in range(max_iterations):
            if time.time() - start_time > timeout:
                break
            
            # Find t-subsets with coverage == 1 (critical)
            critical = [ts for ts, cnt in t_sub_count.items() if cnt == 1]
            if not critical:
                break
            
            # Pick a random critical t-subset
            ts = random.choice(critical)
            
            # Find which block covers this t-subset
            covering_block_idx = None
            for i, b in enumerate(blocks):
                if set(ts).issubset(set(b)):
                    covering_block_idx = i
                    break
            
            if covering_block_idx is None:
                continue
            
            # Try to replace this block with a new one that covers ts and other uncovered sets
            # Generate candidate new blocks containing ts
            base = set(ts)
            remaining_points = [x for x in range(v) if x not in base]
            
            # Try a few random replacements
            improved = False
            for _ in range(5):
                new_block = set(base)
                for _ in range(k - t):
                    x = random.choice(remaining_points)
                    if x not in new_block:
                        new_block.add(x)
                    else:
                        # Try another point
                        other = [y for y in remaining_points if y not in new_block]
                        if other:
                            new_block.add(random.choice(other))
                
                if len(new_block) != k:
                    continue
                
                new_block = sorted(new_block)
                old_block = blocks[covering_block_idx]
                
                if new_block == old_block:
                    continue
                
                # Calculate if this replacement is beneficial
                # New t-subsets covered by new_block but not old_block
                old_ts_set = set(itertools.combinations(old_block, t))
                new_ts_set = set(itertools.combinations(new_block, t))
                
                gained = new_ts_set - old_ts_set
                lost = old_ts_set - new_ts_set
                
                # Check if any lost t-subset becomes uncovered
                can_replace = True
                for ls in lost:
                    if t_sub_count.get(ls, 0) <= 1:
                        can_replace = False
                        break
                
                if not can_replace:
                    continue
                
                # Count how many new t-subsets we cover that were previously uncovered
                new_covered = sum(1 for gs in gained if t_sub_count.get(gs, 0) == 0)
                
                # If we cover new t-subsets, this might allow further reductions
                # For now, just accept if it doesn't break coverage
                if new_covered > 0 or len(gained) > len(lost):
                    # Apply replacement
                    for ls in lost:
                        t_sub_count[ls] -= 1
                    for gs in gained:
                        t_sub_count[gs] = t_sub_count.get(gs, 0) + 1
                    
                    blocks[covering_block_idx] = new_block
                    improved = True
                    break
            
            if not improved:
                # Try removing a random block if possible
                if blocks:
                    idx = random.randrange(len(blocks))
                    b = blocks[idx]
                    redundant = True
                    for ts in itertools.combinations(b, t):
                        if t_sub_count.get(ts, 0) <= 1:
                            redundant = False
                            break
                    if redundant:
                        for ts in itertools.combinations(b, t):
                            t_sub_count[ts] -= 1
                        blocks.pop(idx)
        
        return blocks
    
    # Main algorithm
    best_blocks = None
    best_len = float('inf')
    
    # Try multiple greedy seeds
    for seed in [42, 123, 456, 789, 1000]:
        blocks = greedy_cover(v, k, t, seed=seed)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [list(b) for b in blocks]
    
    # Apply local search
    if best_blocks:
        optimized = local_search(best_blocks, v, k, t, max_iterations=3000, timeout=90)
        if len(optimized) < best_len:
            best_blocks = optimized
            best_len = len(optimized)
    
    return best_blocks
