<<<<<<< SEARCH
"""Randomized local search (simulated annealing) for covering design C(24,7,3)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time


def _precompute(v, k, t):
    """Precompute subset mappings and adjacency structures for fast gain calculation."""
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    
    # Map subset (as frozenset) to index
    subset_to_idx = {frozenset(s): i for i, s in enumerate(all_subsets)}
    
    # For t=3, precompute for each pair (a,b) the list of subset indices containing {a,b}
    # This allows O(1) lookup of which triples are formed by adding x to a pair {a,b}
    pair_to_subsets = {}
    for i, s in enumerate(all_subsets):
        # s is a tuple of 3 elements
        a, b, c = s
        # Add to pairs (a,b), (a,c), (b,c)
        pair_to_subsets.setdefault((a, b), []).append(i)
        pair_to_subsets.setdefault((a, c), []).append(i)
        pair_to_subsets.setdefault((b, c), []).append(i)
        
    return all_subsets, subset_to_idx, pair_to_subsets, n_subsets


def _greedy_cover(v, k, t, rng, all_subsets, subset_to_idx, pair_to_subsets, n_subsets):
    """Fast greedy construction using precomputed pair-to-subset mappings."""
    uncovered = set(range(n_subsets))
    blocks = []
    block_set = set()
    
    while uncovered:
        # Pick a random uncovered t-subset as seed
        seed_idx = rng.choice(list(uncovered))
        seed = all_subsets[seed_idx]
        block = list(seed)
        block_set_local = set(block)
        
        # Greedily add points to maximize coverage of uncovered subsets
        while len(block) < k:
            best_x = None
            best_gain = -1
            
            cands = [x for x in range(v) if x not in block_set_local]
            rng.shuffle(cands)
            
            for x in cands:
                gain = 0
                # For each pair in current block, check triples formed with x
                for a, b in itertools.combinations(block, 2):
                    # Look up triples containing a, b
                    # The triple must contain x, so it is {a, b, x}
                    # We can just check if {a, b, x} is in subset_to_idx and in uncovered
                    # But using pair_to_subsets is faster if we precomputed it for pairs?
                    # Actually, for t=3, the triple is uniquely determined by {a,b,x}.
                    # So we just need to check if frozenset({a,b,x}) is in subset_to_idx.
                    # Since v is small, direct check is fast enough, but let's use the map.
                    fs = frozenset((a, b, x))
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
            # Update uncovered
            for c in itertools.combinations(block, t):
                idx = subset_to_idx[frozenset(c)]
                uncovered.discard(idx)
                
    return blocks


def _block_covered_ids(block, subset_to_idx, t):
    """Get list of subset indices covered by a block."""
    covered = []
    for c in itertools.combinations(block, t):
        covered.append(subset_to_idx[frozenset(c)])
    return covered


def construct(v, k, t):
    rng = random.Random(12345)
    start_time = time.time()
    time_limit = 110.0  # Safety margin for 120s limit

    # Precompute
    all_subsets, subset_to_idx, pair_to_subsets, n_subsets = _precompute(v, k, t)
    
    # Build initial valid covering
    blocks = _greedy_cover(v, k, t, rng, all_subsets, subset_to_idx, pair_to_subsets, n_subsets)
    
    # Precompute covered subset IDs for each block
    block_covered = [_block_covered_ids(b, subset_to_idx, t) for b in blocks]
    
    # counts[i] = number of blocks covering subset i
    counts = [0] * n_subsets
    for covered in block_covered:
        for idx in covered:
            counts[idx] += 1
    
    best_blocks = [list(b) for b in blocks]
    best_count = len(blocks)
    
    # Local Search
    while time.time() - start_time < time_limit:
        # 1. Try to remove a redundant block
        removed = False
        indices = list(range(len(blocks)))
        rng.shuffle(indices)
        for i in indices:
            redundant = True
            for idx in block_covered[i]:
                if counts[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                for idx in block_covered[i]:
                    counts[idx] -= 1
                blocks.pop(i)
                block_covered.pop(i)
                
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = [list(b) for b in blocks]
                removed = True
                break
        
        if removed:
            continue
        
        # 2. Try to swap a block
        if len(blocks) > 0:
            i = rng.randrange(len(blocks))
            old_block = blocks[i]
            old_covered = block_covered[i]
            
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
            # We want to cover as many subsets with count 0 or 1 as possible
            # But we must keep the block size k
            for x in cands:
                if len(new_block_set) >= k:
                    break
                gain = 0
                # Calculate gain: how many subsets with count 0 or 1 does adding x cover?
                # Subsets formed by x and pairs in new_block_set
                for a, b in itertools.combinations(new_block_set, 2):
                    fs = frozenset((a, b, x))
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
                
                new_covered = _block_covered_ids(new_block, subset_to_idx, t)
                
                # Check if swap is valid: no subset loses its last cover
                valid = True
                old_covered_set = set(old_covered)
                new_covered_set = set(new_covered)
                
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
                    
                    if len(blocks) < best_count:
                        best_count = len(blocks)
                        best_blocks = [list(b) for b in blocks]

    return best_blocks
# EVOLVE-BLOCK-END
=======
"""Randomized local search with VNS for covering design C(24,7,3)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time


def _precompute(v, k, t):
    """Precompute subset mappings and adjacency structures for fast gain calculation."""
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    
    # Map subset (as frozenset) to index
    subset_to_idx = {frozenset(s): i for i, s in enumerate(all_subsets)}
    
    # For t=3, precompute for each pair (a,b) the list of subset indices containing {a,b}
    pair_to_subsets = {}
    for i, s in enumerate(all_subsets):
        a, b, c = s
        pair_to_subsets.setdefault((a, b), []).append(i)
        pair_to_subsets.setdefault((a, c), []).append(i)
        pair_to_subsets.setdefault((b, c), []).append(i)
        
    return all_subsets, subset_to_idx, pair_to_subsets, n_subsets


def _greedy_cover(v, k, t, rng, all_subsets, subset_to_idx, pair_to_subsets, n_subsets):
    """Fast greedy construction using precomputed pair-to-subset mappings."""
    uncovered = set(range(n_subsets))
    blocks = []
    block_set = set()
    
    while uncovered:
        seed_idx = rng.choice(list(uncovered))
        seed = all_subsets[seed_idx]
        block = list(seed)
        block_set_local = set(block)
        
        while len(block) < k:
            best_x = None
            best_gain = -1
            
            cands = [x for x in range(v) if x not in block_set_local]
            rng.shuffle(cands)
            
            for x in cands:
                gain = 0
                for a, b in itertools.combinations(block, 2):
                    fs = frozenset((a, b, x))
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


def _block_covered_ids(block, subset_to_idx, t):
    """Get list of subset indices covered by a block."""
    covered = []
    for c in itertools.combinations(block, t):
        covered.append(subset_to_idx[frozenset(c)])
    return covered


def _compute_block_deficit(block, counts, subset_to_idx, t, all_subsets):
    """Compute how many critical subsets this block uniquely covers."""
    critical_count = 0
    for c in itertools.combinations(block, t):
        idx = subset_to_idx[frozenset(c)]
        if counts[idx] == 1:
            critical_count += 1
    return critical_count


def _try_multi_block_swap(blocks, block_covered, counts, subset_to_idx, all_subsets, v, k, t, rng, m, time_limit, start_time):
    """
    Try to swap m blocks simultaneously. Returns True if a valid swap was applied.
    Uses a greedy approach: identify m blocks with high deficit, then try to replace them
    with m new blocks that cover all critical subsets while maximizing coverage.
    """
    if time.time() - start_time > time_limit:
        return False
    
    n_blocks = len(blocks)
    if n_blocks < m:
        return False
    
    # Score each block by its critical coverage (subsets it uniquely covers)
    block_scores = []
    for i in range(n_blocks):
        score = 0
        for idx in block_covered[i]:
            if counts[idx] == 1:
                score += 1
        block_scores.append(score)
    
    # Sort by score descending - we want to replace high-score blocks
    sorted_indices = sorted(range(n_blocks), key=lambda i: -block_scores[i])
    
    # Try several random sets of m blocks to replace
    attempts = 5
    for attempt in range(attempts):
        if time.time() - start_time > time_limit:
            return False
        
        # Select m blocks to replace
        if attempt == 0:
            # Use the top m scoring blocks
            replace_idx = sorted_indices[:m]
        else:
            # Random selection weighted by score
            weights = [block_scores[i] + 1 for i in range(n_blocks)]
            replace_idx = []
            available = list(range(n_blocks))
            for _ in range(m):
                if not available:
                    break
                total_w = sum(weights[j] for j in available)
                r = rng.random() * total_w
                cum = 0
                chosen = available[0]
                for j in available:
                    cum += weights[j]
                    if r <= cum:
                        chosen = j
                        break
                replace_idx.append(chosen)
                available.remove(chosen)
        
        if len(replace_idx) < m:
            continue
        
        # Collect all critical subsets from the blocks we're replacing
        critical_subsets = set()
        for i in replace_idx:
            for idx in block_covered[i]:
                if counts[idx] == 1:
                    critical_subsets.add(idx)
        
        # If no critical subsets, any replacement works - just try random blocks
        if not critical_subsets:
            # Generate m random blocks and check if valid
            new_blocks = []
            for _ in range(m):
                new_block = tuple(sorted(rng.sample(range(v), k)))
                new_blocks.append(new_block)
            
            # Check validity: no subset loses its last cover
            old_covered_set = set()
            for i in replace_idx:
                old_covered_set.update(block_covered[i])
            
            new_covered_all = set()
            for nb in new_blocks:
                for c in itertools.combinations(nb, t):
                    idx = subset_to_idx[frozenset(c)]
                    new_covered_all.add(idx)
            
            valid = True
            for idx in old_covered_set:
                if idx not in new_covered_all and counts[idx] <= 1:
                    valid = False
                    break
            
            if valid:
                # Apply swap
                for i in replace_idx:
                    for idx in block_covered[i]:
                        counts[idx] -= 1
                for nb in new_blocks:
                    nc = _block_covered_ids(nb, subset_to_idx, t)
                    for idx in nc:
                        counts[idx] += 1
                    # Find position to replace
                    # We'll just replace in order
                # Rebuild blocks list
                new_blocks_list = []
                new_covered_list = []
                replaced = set()
                for i in range(n_blocks):
                    if i in replace_idx:
                        # Use corresponding new block
                        pos = list(replace_idx).index(i)
                        nb = new_blocks[pos]
                        new_blocks_list.append(nb)
                        new_covered_list.append(_block_covered_ids(nb, subset_to_idx, t))
                    else:
                        new_blocks_list.append(blocks[i])
                        new_covered_list.append(block_covered[i])
                
                blocks[:] = new_blocks_list
                block_covered[:] = new_covered_list
                return True
            continue
        
        # We have critical subsets to cover. Try to build m blocks that cover them.
        # First, collect all points in critical subsets
        points_in_critical = set()
        for idx in critical_subsets:
            points_in_critical.update(all_subsets[idx])
        
        # If too many points, skip
        if len(points_in_critical) > m * k:
            continue
        
        # Greedily assign critical subsets to new blocks
        # Each new block can contain at most k points
        new_block_sets = [set() for _ in range(m)]
        
        # Sort critical subsets by size (all same size t=3) but prioritize those
        # whose points appear in fewer other critical subsets
        crit_list = list(critical_subsets)
        rng.shuffle(crit_list)
        
        # Greedily pack critical subsets into blocks
        for idx in crit_list:
            pts = set(all_subsets[idx])
            # Find the block with most room that already contains some of these points
            best_block = -1
            best_overlap = -1
            for bi in range(m):
                if len(new_block_sets[bi]) + len(pts - new_block_sets[bi]) <= k:
                    overlap = len(pts & new_block_sets[bi])
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_block = bi
            
            if best_block == -1:
                # Try to put in any block with room
                for bi in range(m):
                    if len(new_block_sets[bi]) + len(pts - new_block_sets[bi]) <= k:
                        best_block = bi
                        break
            
            if best_block == -1:
                # Can't fit all critical subsets, skip this attempt
                break
            
            new_block_sets[best_block].update(pts)
        
        # Check if all critical subsets were placed
        all_placed = True
        for idx in critical_subsets:
            pts = set(all_subsets[idx])
            placed = False
            for bi in range(m):
                if pts.issubset(new_block_sets[bi]):
                    placed = True
                    break
            if not placed:
                all_placed = False
                break
        
        if not all_placed:
            continue
        
        # Fill remaining capacity in each new block to maximize coverage
        for bi in range(m):
            while len(new_block_sets[bi]) < k:
                cands = [x for x in range(v) if x not in new_block_sets[bi]]
                if not cands:
                    break
                rng.shuffle(cands)
                best_x = None
                best_gain = -1
                for x in cands:
                    gain = 0
                    for a, b in itertools.combinations(new_block_sets[bi], 2):
                        fs = frozenset((a, b, x))
                        if fs in subset_to_idx:
                            idx = subset_to_idx[fs]
                            if counts[idx] <= 1:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    break
                new_block_sets[bi].add(best_x)
        
        # Verify all new blocks have exactly k elements
        valid_size = all(len(s) == k for s in new_block_sets)
        if not valid_size:
            continue
        
        # Convert to sorted tuples
        new_blocks = [tuple(sorted(s)) for s in new_block_sets]
        
        # Check for duplicates among new blocks
        if len(set(new_blocks)) != len(new_blocks):
            continue
        
        # Check validity: no subset loses its last cover
        old_covered_set = set()
        for i in replace_idx:
            old_covered_set.update(block_covered[i])
        
        new_covered_all = set()
        for nb in new_blocks:
            for c in itertools.combinations(nb, t):
                idx = subset_to_idx[frozenset(c)]
                new_covered_all.add(idx)
        
        valid = True
        for idx in old_covered_set:
            if idx not in new_covered_all and counts[idx] <= 1:
                valid = False
                break
        
        if not valid:
            continue
        
        # Apply swap
        for i in replace_idx:
            for idx in block_covered[i]:
                counts[idx] -= 1
        for nb in new_blocks:
            nc = _block_covered_ids(nb, subset_to_idx, t)
            for idx in nc:
                counts[idx] += 1
        
        # Rebuild blocks and block_covered lists
        new_blocks_list = []
        new_covered_list = []
        replace_set = set(replace_idx)
        new_block_iter = iter(new_blocks)
        for i in range(n_blocks):
            if i in replace_set:
                nb = next(new_block_iter)
                new_blocks_list.append(nb)
                new_covered_list.append(_block_covered_ids(nb, subset_to_idx, t))
            else:
                new_blocks_list.append(blocks[i])
                new_covered_list.append(block_covered[i])
        
        blocks[:] = new_blocks_list
        block_covered[:] = new_covered_list
        return True
    
    return False


def construct(v, k, t):
    rng = random.Random(12345)
    start_time = time.time()
    time_limit = 110.0  # Safety margin for 120s limit

    # Precompute
    all_subsets, subset_to_idx, pair_to_subsets, n_subsets = _precompute(v, k, t)
    
    # Build initial valid covering
    blocks = _greedy_cover(v, k, t, rng, all_subsets, subset_to_idx, pair_to_subsets, n_subsets)
    
    # Precompute covered subset IDs for each block
    block_covered = [_block_covered_ids(b, subset_to_idx, t) for b in blocks]
    
    # counts[i] = number of blocks covering subset i
    counts = [0] * n_subsets
    for covered in block_covered:
        for idx in covered:
            counts[idx] += 1
    
    best_blocks = [list(b) for b in blocks]
    best_count = len(blocks)
    
    # VNS neighborhood sizes
    vns_cycle = 0
    
    # Local Search with VNS
    while time.time() - start_time < time_limit:
        # 1. Try to remove a redundant block
        removed = False
        indices = list(range(len(blocks)))
        rng.shuffle(indices)
        for i in indices:
            redundant = True
            for idx in block_covered[i]:
                if counts[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                for idx in block_covered[i]:
                    counts[idx] -= 1
                blocks.pop(i)
                block_covered.pop(i)
                
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = [list(b) for b in blocks]
                removed = True
                break
        
        if removed:
            vns_cycle = 0
            continue
        
        # 2. Determine VNS neighborhood size
        # Cycle through m=1, 2, 3
        m = 1 + (vns_cycle % 3)
        
        if m == 1:
            # Single block swap (original logic)
            if len(blocks) > 0:
                i = rng.randrange(len(blocks))
                old_block = blocks[i]
                old_covered = block_covered[i]
                
                # Identify critical subsets (count == 1) in old_block
                critical = [idx for idx in old_covered if counts[idx] == 1]
                
                # Collect points needed for critical subsets
                points_needed = set()
                for idx in critical:
                    points_needed.update(all_subsets[idx])
                
                # If we can't fit all critical points in one block, skip
                if len(points_needed) > k:
                    vns_cycle += 1
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
                    for a, b in itertools.combinations(new_block_set, 2):
                        fs = frozenset((a, b, x))
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
                        vns_cycle += 1
                        continue
                    
                    new_covered = _block_covered_ids(new_block, subset_to_idx, t)
                    
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
                        
                        if len(blocks) < best_count:
                            best_count = len(blocks)
                            best_blocks = [list(b) for b in blocks]
                        vns_cycle = 0
                        continue
        
        elif m == 2:
            # Two-block swap
            if _try_multi_block_swap(blocks, block_covered, counts, subset_to_idx, all_subsets, v, k, t, rng, 2, time_limit, start_time):
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = [list(b) for b in blocks]
                vns_cycle = 0
                continue
        
        elif m == 3:
            # Three-block swap
            if _try_multi_block_swap(blocks, block_covered, counts, subset_to_idx, all_subsets, v, k, t, rng, 3, time_limit, start_time):
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = [list(b) for b in blocks]
                vns_cycle = 0
                continue
        
        # No improvement in this neighborhood, move to next
        vns_cycle += 1

    return best_blocks
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
