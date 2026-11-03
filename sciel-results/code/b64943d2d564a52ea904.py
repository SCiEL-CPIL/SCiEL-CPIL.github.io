"""Covering design C(19,8,5): Bitset-accelerated randomized greedy with local search."""
import itertools
import random
import time
import math
from collections import Counter

# Pre-compute combinations for the specific task parameters to speed up restarts
# The evaluator passes v, k, t, but for this specific task they are fixed.
# However, we should handle general v, k, t as per the contract.

def _all_t_subsets(v, t):
    """Return list of all t-subsets as sorted tuples."""
    return list(itertools.combinations(range(v), t))

def _subset_to_idx_map(t_subs):
    """Map each t-subset tuple to its index."""
    return {s: i for i, s in enumerate(t_subs)}

def _block_cover_indices(block, t, idx_map):
    """Return set of indices of t-subsets covered by block."""
    return {idx_map[tuple(sorted(c))] for c in itertools.combinations(block, t)}

def _greedy_build(v, k, t, t_subs, idx_map, seed=None, deadline=None, safety_margin=5.0):
    """Randomized greedy: repeatedly pick uncovered t-subset, build block maximizing coverage."""
    rng = random.Random(seed)
    n = len(t_subs)
    # uncovered as a set of indices
    uncovered = set(range(n))
    blocks = []
    start = time.time()

    # Pre-compute for each point x, the list of t-subset indices containing x?
    # No, we need intersection with current block.
    
    while uncovered:
        if deadline and (time.time() - start) > deadline - safety_margin:
            break

        # Pick a random uncovered t-subset to start a block
        # To improve diversity, we might pick the one with lowest 'frequency' in potential blocks?
        # For now, random pick is standard.
        start_idx = rng.choice(list(uncovered))
        start_subset = t_subs[start_idx]
        block = list(start_subset)
        
        # Sort block for consistency? Not strictly needed if we use sets.
        block_set = set(block)

        while len(block) < k:
            if deadline and (time.time() - start) > deadline - safety_margin:
                break

            best_x = None
            best_gain = -1
            # Optimization: Only consider x that are in some uncovered subset
            # We can just iterate all x in 0..v-1
            for x in range(v):
                if x in block_set:
                    continue
                
                # Calculate gain: number of uncovered t-subsets S such that x in S and |S \cap block| >= t-1
                # Since we build block by adding one element, and we started with a t-subset,
                # any new subset covered must contain x and t-1 elements from the current block.
                gain = 0
                # Iterate over all (t-1)-subsets of the current block
                # The new subset is formed by union of such (t-1) subset and {x}
                for rest in itertools.combinations(block, t - 1):
                    # Create candidate tuple
                    # We need to check if this candidate is in idx_map and in uncovered
                    # candidate = tuple(sorted(rest + (x,)))
                    # Instead of sorting every time, we can pre-compute?
                    # But block is small.
                    cand = rest + (x,)
                    # cand is a tuple, need to sort it to match idx_map keys
                    # Optimization: We can use a set for uncovered for O(1) lookup if we convert to tuple
                    # But idx_map lookup requires sorted tuple.
                    cand_sorted = tuple(sorted(cand))
                    if cand_sorted in idx_map:
                         idx = idx_map[cand_sorted]
                         if idx in uncovered:
                             gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                    # Tie-breaking: if gain == best_gain and rng.random() < 0.5:
                    #     best_x = x
            
            if best_x is not None:
                block.append(best_x)
                block_set.add(best_x)

        block.sort()
        blocks.append(block)
        
        # Remove covered t-subsets
        mask = _block_cover_indices(block, t, idx_map)
        uncovered -= mask

    return blocks

def _remove_redundant_blocks(blocks, t, t_subs, idx_map, deadline=None, safety_margin=5.0):
    """Remove blocks that are redundant."""
    if not blocks:
        return blocks

    n_blocks = len(blocks)
    # For each block, compute the set of t-subset indices it covers
    block_masks = [_block_cover_indices(b, t, idx_map) for b in blocks]

    # Count how many blocks cover each t-subset
    cover_count = Counter()
    for mask in block_masks:
        for idx in mask:
            cover_count[idx] += 1

    result = []
    start = time.time()
    
    # We can iterate in random order to break symmetries?
    # Or just standard order.
    indices = list(range(n_blocks))
    random.shuffle(indices) # Use a global random or pass seed? 
    # Let's use a local random for determinism within the call?
    # The caller passes seed to greedy, but this function is called after.
    # We'll just use standard order for stability, or random if we want.
    # Actually, removing in a specific order can affect the result.
    # Let's try randomizing the removal order.
    rng = random.Random(42)
    rng.shuffle(indices)

    for i in indices:
        if deadline and (time.time() - start) > deadline - safety_margin:
            result.append(blocks[i])
            continue

        # Check if this block is redundant: all its t-subsets are covered by other blocks
        redundant = True
        for idx in block_masks[i]:
            if cover_count[idx] == 1:
                redundant = False
                break
        
        if not redundant:
            result.append(blocks[i])
            # Decrement cover counts for this block's t-subsets
            for idx in block_masks[i]:
                cover_count[idx] -= 1
        # If redundant, skip it

    return result

def _local_search_swap(blocks, v, k, t, t_subs, idx_map, deadline, max_iters=100):
    """Try to reduce block count by swapping blocks."""
    # Calculate cover counts
    cover_count = Counter()
    block_masks = []
    for b in blocks:
        mask = _block_cover_indices(b, t, idx_map)
        block_masks.append(mask)
        for idx in mask:
            cover_count[idx] += 1
    
    current_blocks = list(blocks)
    current_masks = list(block_masks)
    current_count = len(current_blocks)
    
    # We want to find a block B_i that covers some subsets uniquely.
    # If we remove B_i, those unique subsets become uncovered.
    # We need to find a new block B_new that covers those unique subsets.
    # If we find one, we replace B_i with B_new.
    # The count stays the same, but maybe the new configuration allows further reductions?
    # Actually, if we just swap, count doesn't change.
    # The goal is to enable *removal* of other blocks?
    # Or maybe we can find a B_new that is "better" such that we can then remove another block?
    
    # Better local search:
    # 1. Identify blocks that are "essential" (cover >= 1 unique subset).
    # 2. Try to replace an essential block B_i with a new block B_new that covers ALL unique subsets of B_i.
    # 3. If successful, update cover counts.
    # 4. Check if this replacement made some *other* block redundant.
    # 5. If yes, remove it. Net change: -1 block.
    
    improved = True
    while improved:
        improved = False
        if deadline and (time.time() > deadline - 5.0):
            break
            
        # Calculate current unique subsets for each block
        # unique_subsets[i] = {idx in mask[i] if count[idx] == 1}
        # This is expensive to calculate every time.
        
        # Instead, let's try a specific move:
        # Pick a random block B_i.
        # Find its unique subsets U_i.
        # If U_i is empty, remove it. (Should have been caught by redundancy removal, but maybe new ones appeared).
        
        for i in range(len(current_blocks)):
            if deadline and (time.time() > deadline - 5.0):
                break
                
            # Find unique subsets of current_blocks[i]
            unique = set()
            for idx in current_masks[i]:
                if cover_count[idx] == 1:
                    unique.add(idx)
            
            if not unique:
                # Block is redundant!
                # Remove it
                for idx in current_masks[i]:
                    cover_count[idx] -= 1
                current_blocks.pop(i)
                current_masks.pop(i)
                improved = True
                break # Restart loop as indices changed
                
            # Try to find a new block that covers 'unique'
            # Heuristic: Pick a random subset from 'unique' to start the new block
            # This is slow.
            
    return current_blocks

def _verify_covering(blocks, v, t):
    """Verify every t-subset is covered by at least one block."""
    covered = set()
    for block in blocks:
        for c in itertools.combinations(block, t):
            covered.add(tuple(sorted(c)))
    all_subs = set(itertools.combinations(range(v), t))
    return all_subs == covered

def construct(v, k, t):
    """Construct a covering design: every t-subset of {0..v-1} is in some k-block."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    t_subs = _all_t_subsets(v, t)
    idx_map = _subset_to_idx_map(t_subs)

    best_blocks = None
    best_count = float('inf')

    # Try multiple random seeds with greedy
    max_seeds = 20
    for seed in range(max_seeds):
        if time.time() > deadline - 20.0:
            break

        blocks = _greedy_build(v, k, t, t_subs, idx_map, seed=seed,
                                deadline=deadline, safety_margin=10.0)
        if not blocks:
            continue

        # Remove redundant blocks
        blocks = _remove_redundant_blocks(blocks, t, t_subs, idx_map,
                                           deadline=deadline, safety_margin=10.0)
        
        # Try local search swap/improvement
        # Note: _local_search_swap might be too slow or not effective.
        # Let's stick to redundancy removal and multiple restarts.
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = list(blocks)

        # Early stop if we hit theoretical lower bound
        lower_bound = math.ceil(len(t_subs) / math.comb(k, t))
        if best_count <= lower_bound:
            break

    if best_blocks is None:
        # Fallback
        best_blocks = _greedy_build(v, k, t, t_subs, idx_map, seed=0,
                                     deadline=deadline, safety_margin=5.0)

    # Final verification
    if not _verify_covering(best_blocks, v, t):
        # Extend with more greedy
        uncovered = set(itertools.combinations(range(v), t))
        for block in best_blocks:
            for c in itertools.combinations(block, t):
                uncovered.discard(tuple(sorted(c)))
        
        while uncovered:
            if time.time() > deadline - 5.0:
                break
            start_subset = min(uncovered)
            block = list(start_subset)
            block_set = set(block)
            while len(block) < k:
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        candidate = tuple(sorted(rest + (x,)))
                        if candidate in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is not None:
                    block.append(best_x)
                    block_set.add(best_x)
                else:
                    break
            block.sort()
            best_blocks.append(block)
            for c in itertools.combinations(block, t):
                uncovered.discard(tuple(sorted(c)))

    return best_blocks
