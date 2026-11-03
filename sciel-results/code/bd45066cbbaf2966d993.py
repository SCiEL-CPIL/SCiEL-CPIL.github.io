"""Randomized local search (simulated annealing) for covering design C(24,7,3)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time


def _subsets_to_ints(v, k, t):
    """Precompute all t-subsets as frozensets and map them to indices."""
    all_subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {frozenset(s): i for i, s in enumerate(all_subsets)}
    return all_subsets, subset_to_idx


def _block_mask(block, subset_to_idx, v, k, t):
    """Return the set of t-subset indices covered by a block."""
    covered = set()
    for c in itertools.combinations(block, t):
        idx = subset_to_idx[frozenset(c)]
        covered.add(idx)
    return covered


def _greedy_cover(v, k, t, rng, all_subsets, subset_to_idx):
    """Fast greedy construction to get a valid starting covering."""
    uncovered = set(range(len(all_subsets)))
    blocks = []
    block_set = set()
    
    while uncovered:
        # Pick a random uncovered t-subset as seed
        seed_idx = rng.choice(list(uncovered))
        seed = all_subsets[seed_idx]
        block = list(seed)
        rng.shuffle(block)
        block_set_local = set(block)
        
        # Greedily add points to maximize coverage
        while len(block) < k:
            best_x = None
            best_gain = -1
            cands = [x for x in range(v) if x not in block_set_local]
            rng.shuffle(cands)
            
            for x in cands:
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    fs = frozenset(list(rest) + [x])
                    if fs in subset_to_idx:
                        idx = subset_to_idx[fs]
                        if idx in uncovered:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
            block_set_local.add(best_x)
            
        block.sort()
        b_tuple = tuple(block)
        if b_tuple not in block_set:
            block_set.add(b_tuple)
            blocks.append(b_tuple)
            for c in itertools.combinations(block, t):
                idx = subset_to_idx[frozenset(c)]
                uncovered.discard(idx)
                
    return blocks


def _polish(blocks, subset_to_idx, v, k, t, all_subsets, counts, block_covered, rng, time_limit):
    """Final high-precision polish: iteratively remove redundant blocks and try swaps."""
    improved = True
    while improved:
        if time.time() > time_limit:
            break
        improved = False
        
        # Remove all redundant blocks
        removed = True
        while removed:
            removed = False
            for i in range(len(blocks)):
                redundant = all(counts[idx] > 1 for idx in block_covered[i])
                if redundant:
                    for idx in block_covered[i]:
                        counts[idx] -= 1
                    blocks.pop(i)
                    block_covered.pop(i)
                    removed = True
                    break
        
        # Try single-block swaps
        for i in range(len(blocks)):
            if time.time() > time_limit:
                break
            old_block = blocks[i]
            old_covered = block_covered[i]
            old_block_set = set(old_block)
            
            critical = [idx for idx in old_covered if counts[idx] == 1]
            points_needed = set()
            for idx in critical:
                points_needed.update(all_subsets[idx])
            
            if len(points_needed) > k:
                continue
            
            new_block_set = set(points_needed)
            cands = [x for x in range(v) if x not in new_block_set]
            rng.shuffle(cands)
            
            for x in cands:
                if len(new_block_set) >= k:
                    break
                gain = 0
                for rest in itertools.combinations(new_block_set, t-1):
                    fs = frozenset(list(rest) + [x])
                    if fs in subset_to_idx:
                        idx = subset_to_idx[fs]
                        if counts[idx] == 0:
                            gain += 10
                        elif counts[idx] == 1:
                            gain += 1
                if gain > 0:
                    new_block_set.add(x)
                elif len(new_block_set) < k:
                    new_block_set.add(x)
                    
            if len(new_block_set) == k:
                new_block = tuple(sorted(new_block_set))
                if new_block == old_block:
                    continue
                
                new_covered = _block_mask(new_block, subset_to_idx, v, k, t)
                
                valid = True
                for idx in old_covered:
                    if idx not in new_covered and counts[idx] <= 1:
                        valid = False
                        break
                
                if valid:
                    for idx in old_covered:
                        counts[idx] -= 1
                    for idx in new_covered:
                        counts[idx] += 1
                    
                    blocks[i] = new_block
                    block_covered[i] = new_covered
                    improved = True
                    
    return blocks, block_covered, counts


def construct(v, k, t):
    rng = random.Random(12345)
    start_time = time.time()
    time_limit = 145.0  # leave 5s buffer

    # Precompute
    all_subsets, subset_to_idx = _subsets_to_ints(v, k, t)
    n_subsets = len(all_subsets)
    
    # Precompute for each point, the list of t-subsets containing it
    point_to_subsets = [[] for _ in range(v)]
    for idx, s in enumerate(all_subsets):
        for p in s:
            point_to_subsets[p].append(idx)
    
    # Build initial valid covering
    blocks = _greedy_cover(v, k, t, rng, all_subsets, subset_to_idx)
    
    # Precompute covered subsets for each block
    block_covered = [_block_mask(b, subset_to_idx, v, k, t) for b in blocks]
    
    # counts[i] = number of blocks covering subset i
    counts = [0] * n_subsets
    for covered in block_covered:
        for idx in covered:
            counts[idx] += 1
    
    best_blocks = [list(b) for b in blocks]
    best_count = len(blocks)
    
    # Simulated Annealing parameters
    T = 50.0
    T_min = 0.1
    cooling = 0.999
    
    iteration = 0
    no_improve = 0
    max_no_improve = 300
    
    while time.time() - start_time < time_limit:
        iteration += 1
        
        # 1. Try to remove a redundant block (try multiple times per iteration)
        removed = False
        indices = list(range(len(blocks)))
        rng.shuffle(indices)
        for i in indices:
            # Check if all subsets in this block have count > 1
            redundant = True
            for idx in block_covered[i]:
                if counts[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                # Remove it
                for idx in block_covered[i]:
                    counts[idx] -= 1
                blocks.pop(i)
                block_covered.pop(i)
                
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = [list(b) for b in blocks]
                    no_improve = 0
                removed = True
                break
        
        if removed:
            continue
        
        # 2. Try to swap a block with better gain computation
        if len(blocks) > 0:
            i = rng.randrange(len(blocks))
            old_block = blocks[i]
            old_covered = block_covered[i]
            old_block_set = set(old_block)
            
            # Identify critical subsets (count == 1) in old_block
            critical = [idx for idx in old_covered if counts[idx] == 1]
            
            # Collect points needed for critical subsets
            points_needed = set()
            for idx in critical:
                points_needed.update(all_subsets[idx])
            
            # If we can't fit all critical points in one block, skip
            if len(points_needed) > k:
                continue
            
            # Build new block containing points_needed
            new_block_set = set(points_needed)
            cands = [x for x in range(v) if x not in new_block_set]
            rng.shuffle(cands)
            
            # Greedy fill to maximize coverage of low-count subsets
            for x in cands:
                if len(new_block_set) >= k:
                    break
                gain = 0
                for rest in itertools.combinations(new_block_set, t-1):
                    fs = frozenset(list(rest) + [x])
                    if fs in subset_to_idx:
                        idx = subset_to_idx[fs]
                        if counts[idx] == 0:
                            gain += 10
                        elif counts[idx] == 1:
                            gain += 1
                if gain > 0:
                    new_block_set.add(x)
                elif len(new_block_set) < k:
                    new_block_set.add(x)
                    
            if len(new_block_set) == k:
                new_block = tuple(sorted(new_block_set))
                
                if new_block == old_block:
                    continue
                
                new_covered = _block_mask(new_block, subset_to_idx, v, k, t)
                
                # Check if swap is valid: no subset loses its last cover
                valid = True
                for idx in old_covered:
                    if idx not in new_covered and counts[idx] <= 1:
                        valid = False
                        break
                
                if valid:
                    # Apply swap
                    for idx in old_covered:
                        counts[idx] -= 1
                    for idx in new_covered:
                        counts[idx] += 1
                    
                    blocks[i] = new_block
                    block_covered[i] = new_covered
        
        # 3. Try to merge two blocks into one (replace 2 blocks with 1)
        if len(blocks) >= 2 and rng.random() < 0.1:
            i, j = rng.sample(range(len(blocks)), 2)
            b1 = set(blocks[i])
            b2 = set(blocks[j])
            # Try to find a block that covers all critical subsets from both
            crit1 = [idx for idx in block_covered[i] if counts[idx] == 1]
            crit2 = [idx for idx in block_covered[j] if counts[idx] == 1]
            
            points_needed = set()
            for idx in crit1 + crit2:
                points_needed.update(all_subsets[idx])
            
            if len(points_needed) <= k:
                new_block_set = set(points_needed)
                cands = [x for x in range(v) if x not in new_block_set]
                rng.shuffle(cands)
                
                for x in cands:
                    if len(new_block_set) >= k:
                        break
                    gain = 0
                    for rest in itertools.combinations(new_block_set, t-1):
                        fs = frozenset(list(rest) + [x])
                        if fs in subset_to_idx:
                            idx = subset_to_idx[fs]
                            if counts[idx] == 0:
                                gain += 10
                            elif counts[idx] == 1:
                                gain += 1
                    if gain > 0:
                        new_block_set.add(x)
                    elif len(new_block_set) < k:
                        new_block_set.add(x)
                
                if len(new_block_set) == k:
                    new_block = tuple(sorted(new_block_set))
                    new_covered = _block_mask(new_block, subset_to_idx, v, k, t)
                    
                    # Check if this single block covers all critical subsets from both
                    valid = True
                    for idx in crit1 + crit2:
                        if idx not in new_covered:
                            valid = False
                            break
                    
                    if valid:
                        # Remove both blocks and add the new one
                        for idx in block_covered[i]:
                            counts[idx] -= 1
                        for idx in block_covered[j]:
                            counts[idx] -= 1
                        for idx in new_covered:
                            counts[idx] += 1
                        
                        # Remove blocks i and j (careful with indices)
                        if i < j:
                            blocks.pop(j)
                            block_covered.pop(j)
                            blocks.pop(i)
                            block_covered.pop(i)
                        else:
                            blocks.pop(i)
                            block_covered.pop(i)
                            blocks.pop(j)
                            block_covered.pop(j)
                        
                        blocks.append(new_block)
                        block_covered.append(new_covered)
                        
                        if len(blocks) < best_count:
                            best_count = len(blocks)
                            best_blocks = [list(b) for b in blocks]
                            no_improve = 0
        
        # 4. Random perturbation if stuck
        if no_improve > max_no_improve:
            # Try a new greedy cover with different seed
            new_rng = random.Random(rng.randint(0, 10**9))
            new_cover = _greedy_cover(v, k, t, new_rng, all_subsets, subset_to_idx)
            if len(new_cover) < len(blocks):
                blocks = new_cover
                block_covered = [_block_mask(b, subset_to_idx, v, k, t) for b in blocks]
                counts = [0] * n_subsets
                for covered in block_covered:
                    for idx in covered:
                        counts[idx] += 1
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = [list(b) for b in blocks]
                    no_improve = 0
            no_improve = 0
            T = max(T_min, T * 0.5)
        
        if len(blocks) < best_count:
            no_improve = 0
        else:
            no_improve += 1
        
        T = max(T_min, T * cooling)

    # Final polish pass
    blocks, block_covered, counts = _polish(
        blocks, subset_to_idx, v, k, t, all_subsets, counts, block_covered, rng, time_limit
    )
    
    if len(blocks) < best_count:
        best_count = len(blocks)
        best_blocks = [list(b) for b in blocks]

    return best_blocks
# EVOLVE-BLOCK-END
