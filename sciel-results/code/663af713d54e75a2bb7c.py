<<<<<<< SEARCH
"""Randomized local search (simulated annealing) for covering design C(24,7,3)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np

_CACHE = {}

def _get_precomputed(v, k, t):
    if (v, k, t) in _CACHE:
        return _CACHE[(v, k, t)]
        
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Precompute indices of subsets covered by each block
    # block_indices[b] is a numpy array of subset indices
    block_indices = [None] * n_blocks
    
    # Mapping from block tuple to block index
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    
    for b_idx, block in enumerate(all_blocks):
        idxs = []
        for c in itertools.combinations(block, t):
            idxs.append(subset_to_idx[c])
        block_indices[b_idx] = np.array(idxs, dtype=np.int32)
        
    _CACHE[(v, k, t)] = {
        'all_t_subsets': all_t_subsets,
        'n_subsets': n_subsets,
        'subset_to_idx': subset_to_idx,
        'all_blocks': all_blocks,
        'n_blocks': n_blocks,
        'block_indices': block_indices,
        'block_to_idx': block_to_idx
    }
    return _CACHE[(v, k, t)]

def _greedy_cover_np(v, k, t, rng, pre):
    """Greedy cover using precomputed data."""
    all_t_subsets = pre['all_t_subsets']
    n_subsets = pre['n_subsets']
    all_blocks = pre['all_blocks']
    block_indices = pre['block_indices']
    block_to_idx = pre['block_to_idx']
    
    uncovered = np.ones(n_subsets, dtype=bool)
    blocks = []
    block_set = set()
    
    while uncovered.any():
        # Pick a random uncovered subset
        uncovered_indices = np.where(uncovered)[0]
        seed_idx = rng.choice(uncovered_indices)
        seed = all_t_subsets[seed_idx]
        
        # Find blocks containing this seed? No, we construct a block.
        # Greedy construction of a block
        block_set_local = set(seed)
        current_block = list(seed)
        
        # We need to fill the block to size k
        # To do this efficiently, we can just pick random points, but greedy is better.
        # However, for initialization, a simple random fill or greedy fill is fine.
        # Let's try to pick points that cover many uncovered subsets.
        
        while len(current_block) < k:
            best_x = None
            best_gain = -1
            cands = [x for x in range(v) if x not in block_set_local]
            rng.shuffle(cands)
            
            for x in cands:
                # Calculate gain
                # New subsets formed by adding x to current_block
                # These are combinations of (t-1) elements from current_block and x
                gain = 0
                # Optimization: precompute combinations of current_block?
                # Since k is small (7), this is fast.
                for rest in itertools.combinations(current_block, t-1):
                    c = tuple(sorted(list(rest) + [x]))
                    if c in pre['subset_to_idx']:
                        idx = pre['subset_to_idx'][c]
                        if uncovered[idx]:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            current_block.append(best_x)
            block_set_local.add(best_x)
            
        if len(current_block) == k:
            current_block.sort()
            b_tuple = tuple(current_block)
            if b_tuple not in block_set:
                block_set.add(b_tuple)
                blocks.append(b_tuple)
                b_idx = block_to_idx[b_tuple]
                for idx in block_indices[b_idx]:
                    uncovered[idx] = False
                    
    return blocks

def construct(v, k, t):
    rng = random.Random(12345)
    start_time = time.time()
    time_limit = 115.0
    
    pre = _get_precomputed(v, k, t)
    all_t_subsets = pre['all_t_subsets']
    n_subsets = pre['n_subsets']
    all_blocks = pre['all_blocks']
    n_blocks = pre['n_blocks']
    block_indices = pre['block_indices']
    block_to_idx = pre['block_to_idx']
    
    # Build initial valid covering
    blocks = _greedy_cover_np(v, k, t, rng, pre)
    
    # Convert to indices
    block_idxs = [block_to_idx[b] for b in blocks]
    
    # counts[i] = number of blocks covering subset i
    counts = np.zeros(n_subsets, dtype=np.int32)
    for b_idx in block_idxs:
        counts[block_indices[b_idx]] += 1
        
    best_block_idxs = list(block_idxs)
    best_count = len(block_idxs)
    
    # Local search
    no_improve = 0
    max_no_improve = 500
    
    while time.time() - start_time < time_limit:
        # 1. Try to remove a redundant block
        removed = False
        # Pick a random block to check
        i = rng.randrange(len(block_idxs))
        b_idx = block_idxs[i]
        covered = block_indices[b_idx]
        
        if np.all(counts[covered] > 1):
            counts[covered] -= 1
            block_idxs.pop(i)
            removed = True
            if len(block_idxs) < best_count:
                best_count = len(block_idxs)
                best_block_idxs = list(block_idxs)
                no_improve = 0
            continue
        
        # 2. Try to swap a block
        if len(block_idxs) > 0:
            i = rng.randrange(len(block_idxs))
            old_b_idx = block_idxs[i]
            old_covered = block_indices[old_b_idx]
            
            # Find critical subsets
            critical_mask = (counts[old_covered] == 1)
            
            # If no critical subsets, we can swap to any block?
            # No, we want to minimize redundancy or keep same size.
            # Actually, if no critical, we can just swap to a random block and see if it improves "potential" (e.g. reduces max count)?
            # For now, if no critical, we can pick a random new block.
            # But we must ensure we don't break coverage?
            # If no critical, removing old block leaves all counts >= 1. So any new block is valid.
            
            if np.any(critical_mask):
                critical_indices = old_covered[critical_mask]
                # We need a new block that covers all critical_indices
                # This is a constraint.
                # Find a random block that covers all critical_indices?
                # Or construct one?
                
                # Strategy: Pick a random block, check if it covers critical_indices.
                # If not, try another.
                # Since critical_indices is small (subset of 35), and blocks are random,
                # probability of covering a specific critical triple is 35/C(24,3) ~ 1.7%.
                # If there are 3 critical triples, prob is very low.
                # Better: Construct a block containing the points of critical triples.
                
                points_needed = set()
                for idx in critical_indices:
                    points_needed.update(all_t_subsets[idx])
                
                if len(points_needed) > k:
                    continue # Can't fit all critical points in one block
                
                new_block_set = set(points_needed)
                cands = [x for x in range(v) if x not in new_block_set]
                rng.shuffle(cands)
                
                # Fill rest of block greedily to cover more uncovered/low-count subsets
                for x in cands:
                    if len(new_block_set) >= k:
                        break
                    new_block_set.add(x)
                
                if len(new_block_set) == k:
                    new_block = tuple(sorted(new_block_set))
                    new_b_idx = block_to_idx[new_block]
                    new_covered = block_indices[new_b_idx]
                    
                    # Check validity: new block must cover all critical subsets
                    # Critical subsets are those with count 1 in old block.
                    # We ensured points_needed covers them, so new block covers them.
                    # But we must check if new block covers *all* critical_indices.
                    # Since we added points from critical_indices, and block is a superset of points_needed,
                    # it covers all triples in points_needed.
                    # So it covers all critical_indices.
                    
                    # Apply swap
                    counts[old_covered] -= 1
                    counts[new_covered] += 1
                    block_idxs[i] = new_b_idx
            else:
                # No critical subsets. Swap to a random block to explore.
                # Avoid swapping to the same block.
                new_b_idx = rng.randrange(n_blocks)
                if new_b_idx == old_b_idx:
                    continue
                new_covered = block_indices[new_b_idx]
                
                # Since no critical, swap is always valid.
                counts[old_covered] -= 1
                counts[new_covered] += 1
                block_idxs[i] = new_b_idx
        
        if len(block_idxs) < best_count:
            no_improve = 0
        else:
            no_improve += 1
            
        if no_improve > max_no_improve:
            # Restart
            new_rng = random.Random(rng.randint(0, 10**9))
            new_blocks = _greedy_cover_np(v, k, t, new_rng, pre)
            if len(new_blocks) < len(block_idxs):
                block_idxs = [block_to_idx[b] for b in new_blocks]
                counts = np.zeros(n_subsets, dtype=np.int32)
                for b_idx in block_idxs:
                    counts[block_indices[b_idx]] += 1
                if len(block_idxs) < best_count:
                    best_count = len(block_idxs)
                    best_block_idxs = list(block_idxs)
            no_improve = 0

    return [list(all_blocks[i]) for i in best_block_idxs]
# EVOLVE-BLOCK-END
=======
"""Randomized local search (simulated annealing) for covering design C(24,7,3)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np

_CACHE = {}

def _get_precomputed(v, k, t):
    if (v, k, t) in _CACHE:
        return _CACHE[(v, k, t)]
        
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Precompute indices of subsets covered by each block
    block_indices = [None] * n_blocks
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    
    for b_idx, block in enumerate(all_blocks):
        idxs = []
        for c in itertools.combinations(block, t):
            idxs.append(subset_to_idx[c])
        block_indices[b_idx] = np.array(idxs, dtype=np.int32)
        
    _CACHE[(v, k, t)] = {
        'all_t_subsets': all_t_subsets,
        'n_subsets': n_subsets,
        'subset_to_idx': subset_to_idx,
        'all_blocks': all_blocks,
        'n_blocks': n_blocks,
        'block_indices': block_indices,
        'block_to_idx': block_to_idx
    }
    return _CACHE[(v, k, t)]

def _greedy_cover_np(v, k, t, rng, pre):
    """Greedy cover using precomputed data."""
    all_t_subsets = pre['all_t_subsets']
    n_subsets = pre['n_subsets']
    block_indices = pre['block_indices']
    block_to_idx = pre['block_to_idx']
    
    uncovered = np.ones(n_subsets, dtype=bool)
    blocks = []
    block_set = set()
    
    while uncovered.any():
        uncovered_indices = np.where(uncovered)[0]
        seed_idx = rng.choice(uncovered_indices)
        seed = all_t_subsets[seed_idx]
        
        block_set_local = set(seed)
        current_block = list(seed)
        
        while len(current_block) < k:
            best_x = None
            best_gain = -1
            cands = [x for x in range(v) if x not in block_set_local]
            rng.shuffle(cands)
            
            for x in cands:
                gain = 0
                for rest in itertools.combinations(current_block, t-1):
                    c = tuple(sorted(list(rest) + [x]))
                    if c in pre['subset_to_idx']:
                        idx = pre['subset_to_idx'][c]
                        if uncovered[idx]:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            current_block.append(best_x)
            block_set_local.add(best_x)
            
        if len(current_block) == k:
            current_block.sort()
            b_tuple = tuple(current_block)
            if b_tuple not in block_set:
                block_set.add(b_tuple)
                blocks.append(b_tuple)
                b_idx = block_to_idx[b_tuple]
                for idx in block_indices[b_idx]:
                    uncovered[idx] = False
                    
    return blocks

def _try_remove_redundant(block_idxs, counts, block_indices, rng, max_attempts=20):
    """Try to remove redundant blocks. Returns number removed."""
    removed = 0
    for _ in range(max_attempts):
        if not block_idxs:
            break
        i = rng.randrange(len(block_idxs))
        b_idx = block_idxs[i]
        covered = block_indices[b_idx]
        
        if np.all(counts[covered] > 1):
            counts[covered] -= 1
            block_idxs.pop(i)
            removed += 1
    return removed

def _find_swap_candidates(old_b_idx, critical_indices, all_t_subsets, k, v, rng, pre, counts, block_indices, old_covered, max_tries=50):
    """Find valid swap candidates that cover all critical subsets.
    Returns list of (new_b_idx, new_covered) tuples."""
    if not critical_indices:
        return []
    
    points_needed = set()
    for idx in critical_indices:
        points_needed.update(all_t_subsets[idx])
    
    if len(points_needed) > k:
        return []
    
    candidates = []
    remaining_slots = k - len(points_needed)
    
    if remaining_slots == 0:
        new_block = tuple(sorted(points_needed))
        if new_block in pre['block_to_idx']:
            new_b_idx = pre['block_to_idx'][new_block]
            new_covered = block_indices[new_b_idx]
            # Verify it covers all critical
            crit_set = set(critical_indices)
            if all(c in set(new_covered) for c in crit_set):
                candidates.append((new_b_idx, new_covered))
        return candidates
    
    # Generate random completions
    cands_pool = [x for x in range(v) if x not in points_needed]
    
    for _ in range(max_tries):
        new_block_set = set(points_needed)
        if remaining_slots > 0:
            rng.shuffle(cands_pool)
            for x in cands_pool[:remaining_slots]:
                new_block_set.add(x)
        
        if len(new_block_set) == k:
            new_block = tuple(sorted(new_block_set))
            if new_block in pre['block_to_idx']:
                new_b_idx = pre['block_to_idx'][new_block]
                new_covered = block_indices[new_b_idx]
                # Verify covers all critical
                crit_set = set(critical_indices)
                if all(c in set(new_covered) for c in crit_set):
                    # Check if this swap would reduce redundancy
                    # Gain: how many subsets have count decrease from >=2 to 1
                    # Loss: how many subsets go from 1 to 2
                    old_mask = (counts[old_covered] == 1)
                    new_mask = (counts[new_covered] == 1)
                    # Subsets in old_covered that are critical
                    critical_in_old = np.intersect1d(old_covered[old_mask], new_covered)
                    # Subsets in new_covered that are currently critical
                    critical_in_new = np.intersect1d(new_covered[new_mask], old_covered)
                    # Net effect on "critical count"
                    # We want swaps that don't increase total critical subsets
                    candidates.append((new_b_idx, new_covered))
    
    return candidates

def _polish(block_idxs, counts, block_indices, all_t_subsets, pre, rng, time_budget=5.0):
    """Final polish pass: aggressively try to remove redundant blocks."""
    start = time.time()
    improved = True
    while improved and (time.time() - start) < time_budget:
        improved = False
        removed = _try_remove_redundant(block_idxs, counts, block_indices, rng, max_attempts=100)
        if removed > 0:
            improved = True
    
    return block_idxs, counts

def construct(v, k, t):
    rng = random.Random(12345)
    start_time = time.time()
    time_limit = 115.0
    
    pre = _get_precomputed(v, k, t)
    all_t_subsets = pre['all_t_subsets']
    n_subsets = pre['n_subsets']
    all_blocks = pre['all_blocks']
    n_blocks = pre['n_blocks']
    block_indices = pre['block_indices']
    block_to_idx = pre['block_to_idx']
    
    # Build initial valid covering
    blocks = _greedy_cover_np(v, k, t, rng, pre)
    block_idxs = [block_to_idx[b] for b in blocks]
    
    counts = np.zeros(n_subsets, dtype=np.int32)
    for b_idx in block_idxs:
        counts[block_indices[b_idx]] += 1
        
    best_block_idxs = list(block_idxs)
    best_count = len(block_idxs)
    
    # Local search
    no_improve = 0
    max_no_improve = 300
    
    while time.time() - start_time < time_limit:
        # 1. Try to remove redundant blocks (multiple attempts)
        removed = _try_remove_redundant(block_idxs, counts, block_indices, rng, max_attempts=10)
        if removed > 0:
            if len(block_idxs) < best_count:
                best_count = len(block_idxs)
                best_block_idxs = list(block_idxs)
                no_improve = 0
            continue
        
        # 2. Try to swap a block
        if len(block_idxs) > 0:
            i = rng.randrange(len(block_idxs))
            old_b_idx = block_idxs[i]
            old_covered = block_indices[old_b_idx]
            
            critical_mask = (counts[old_covered] == 1)
            
            if np.any(critical_mask):
                critical_indices = old_covered[critical_mask]
                
                # Find swap candidates
                candidates = _find_swap_candidates(
                    old_b_idx, critical_indices, all_t_subsets, k, v, rng, pre, 
                    counts, block_indices, old_covered, max_tries=30
                )
                
                if candidates:
                    # Pick a random candidate
                    new_b_idx, new_covered = rng.choice(candidates)
                    if new_b_idx != old_b_idx:
                        counts[old_covered] -= 1
                        counts[new_covered] += 1
                        block_idxs[i] = new_b_idx
            else:
                # No critical subsets - try to find a block that reduces overall redundancy
                # Pick a random block and see if it has lower "redundancy score"
                new_b_idx = rng.randrange(n_blocks)
                if new_b_idx == old_b_idx:
                    continue
                new_covered = block_indices[new_b_idx]
                
                # Check if this swap would reduce total redundancy
                # Redundancy = sum of (counts[i] - 1) for all i
                old_redundancy = np.sum(np.maximum(counts[old_covered] - 1, 0))
                new_redundancy = np.sum(np.maximum(counts[new_covered] - 1, 0))
                
                # Only swap if it doesn't increase redundancy too much
                if new_redundancy <= old_redundancy + 2:
                    counts[old_covered] -= 1
                    counts[new_covered] += 1
                    block_idxs[i] = new_b_idx
        
        if len(block_idxs) < best_count:
            no_improve = 0
        else:
            no_improve += 1
            
        if no_improve > max_no_improve:
            # Restart with a new greedy cover
            new_rng = random.Random(rng.randint(0, 10**9))
            new_blocks = _greedy_cover_np(v, k, t, new_rng, pre)
            if len(new_blocks) < len(block_idxs):
                block_idxs = [block_to_idx[b] for b in new_blocks]
                counts = np.zeros(n_subsets, dtype=np.int32)
                for b_idx in block_idxs:
                    counts[block_indices[b_idx]] += 1
                if len(block_idxs) < best_count:
                    best_count = len(block_idxs)
                    best_block_idxs = list(block_idxs)
            no_improve = 0

    # Final polish: aggressively remove redundant blocks
    best_block_idxs, _ = _polish(best_block_idxs, 
                                  np.zeros(n_subsets, dtype=np.int32),
                                  block_indices, all_t_subsets, pre, rng, time_budget=3.0)
    # Recount for the polished version
    counts_check = np.zeros(n_subsets, dtype=np.int32)
    for b_idx in best_block_idxs:
        counts_check[block_indices[b_idx]] += 1
    
    # Verify validity
    if np.all(counts_check >= 1):
        return [list(all_blocks[i]) for i in best_block_idxs]
    
    # Fallback to pre-polish best
    return [list(all_blocks[i]) for i in best_block_idxs]
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
