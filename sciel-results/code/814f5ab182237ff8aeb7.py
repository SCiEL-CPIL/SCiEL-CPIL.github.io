import time
import itertools
import random
import sys

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using a multi-start randomized greedy 
    construction followed by aggressive reduction.
    
    Strategy:
    1. Precompute all t-subsets and map them to indices.
    2. Run multiple randomized greedy constructions:
       - At each step, select an uncovered t-subset.
       - Evaluate candidate blocks (randomly sampled or exhaustive if small) to find 
         the one covering the target and maximizing new coverage.
       - Add the best block.
    3. Apply a reduction phase: iterate through blocks and remove any block that 
       does not uniquely cover any t-subset.
    4. Keep the best (smallest) valid result found within the time limit.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Precomputation
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    best_blocks = None
    best_size = float('inf')
    
    # Number of restarts to attempt. We will adjust based on time.
    # A single greedy pass is fast, so we can do many.
    # Let's aim for as many as possible, but ensure at least 10-20.
    max_restarts = 1000 
    restarts_done = 0
    
    while restarts_done < max_restarts:
        current_time = time.time()
        # If we have very little time left, break
        if current_time > deadline - 2.0:
            break
            
        # Randomize the order of points to break symmetry in greedy choice
        # This helps diversify the blocks generated
        randomized_points = points[:]
        random.shuffle(randomized_points)
        # Note: The actual subset indices are based on original points, 
        # but we can use randomized selection of target subsets.
        
        # State
        uncovered = set(range(num_t_subsets))
        blocks = []
        
        # Greedy Construction Loop
        # Optimization: Instead of checking all C(v,k) blocks every step (too slow),
        # we sample a set of candidate blocks. 
        # For v=19, k=8, C(19,8) = 75582. 
        # Checking 75k blocks * 11k subsets is too slow in Python loops.
        # We need a smarter candidate generation.
        
        # Heuristic: 
        # 1. Pick a random uncovered t-subset.
        # 2. Generate a pool of candidate blocks that contain this subset.
        #    Number of such blocks is C(v-t, k-t). For 19,8,5: C(14,3)=364.
        #    This is small enough to enumerate all candidates for the target subset!
        
        while uncovered:
            if time.time() > deadline - 5.0:
                # Time out for this restart, discard
                break
                
            # Pick a target t-subset
            # Strategy: Pick one that is "hard" to cover? 
            # For now, random is good for diversity.
            target_idx = random.choice(list(uncovered))
            target_subset = all_t_subsets[target_idx]
            target_set = set(target_subset)
            
            # Generate all blocks that contain target_subset
            # The remaining k-t points must be chosen from the v-t points not in target_subset
            available_points = [p for p in points if p not in target_set]
            num_remaining = k - t
            
            # Enumerate all combinations of available_points of size num_remaining
            # C(14, 3) = 364. This is very fast.
            candidates = []
            for rem in itertools.combinations(available_points, num_remaining):
                block = tuple(sorted(target_subset + rem))
                candidates.append(block)
            
            # Evaluate each candidate for "gain" (number of new t-subsets covered)
            best_candidate = None
            best_gain = -1
            
            # To speed up gain calculation, we can pre-compute the t-subsets of each candidate
            # But 364 candidates * C(8,5)=56 subsets = ~20k checks. 
            # Each check is a dict lookup. This is fast enough.
            
            for block in candidates:
                gain = 0
                # Check all t-subsets of this block
                for ts in itertools.combinations(block, t):
                    idx = t_subset_to_idx.get(ts)
                    if idx is not None and idx in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = block
            
            if best_candidate is None:
                # Should not happen, but safety
                break
                
            # Add block
            blocks.append(best_candidate)
            
            # Update uncovered set
            for ts in itertools.combinations(best_candidate, t):
                idx = t_subset_to_idx.get(ts)
                if idx is not None:
                    uncovered.discard(idx)
        
        # If the greedy loop finished (uncovered is empty), we have a valid covering
        if not uncovered:
            # Reduction Phase
            reduced_blocks = reduce_blocks(blocks, v, k, t, all_t_subsets, t_subset_to_idx)
            
            # Verify validity (just in case reduction logic has a bug)
            if is_valid_covering(reduced_blocks, v, k, t, all_t_subsets, t_subset_to_idx):
                if len(reduced_blocks) < best_size:
                    best_size = len(reduced_blocks)
                    best_blocks = reduced_blocks
                    # Update deadline if we found a very good solution early? 
                    # No, keep searching for better.
            
            restarts_done += 1
    
    # If we didn't find any valid covering (should not happen with this greedy), 
    # return the best attempt or a trivial one
    if best_blocks is None:
        # Fallback: Return a trivial covering (all k-subsets) - too large, but valid
        # Or just return the last attempt if it was valid? 
        # The greedy above *should* always produce a valid covering if it terminates.
        # If it timed out mid-construction, 'uncovered' is not empty.
        # Let's try one more time with a deterministic greedy if we have time.
        if time.time() < deadline - 10.0:
            # Deterministic greedy: always pick the first uncovered
            uncovered = set(range(num_t_subsets))
            blocks = []
            while uncovered:
                target_idx = min(uncovered)
                target_subset = all_t_subsets[target_idx]
                target_set = set(target_subset)
                available_points = [p for p in points if p not in target_set]
                candidates = []
                for rem in itertools.combinations(available_points, k - t):
                    block = tuple(sorted(target_subset + rem))
                    candidates.append(block)
                
                best_candidate = None
                best_gain = -1
                for block in candidates:
                    gain = 0
                    for ts in itertools.combinations(block, t):
                        idx = t_subset_to_idx.get(ts)
                        if idx is not None and idx in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_candidate = block
                
                if best_candidate:
                    blocks.append(best_candidate)
                    for ts in itertools.combinations(best_candidate, t):
                        idx = t_subset_to_idx.get(ts)
                        if idx is not None:
                            uncovered.discard(idx)
            
            if not uncovered:
                reduced_blocks = reduce_blocks(blocks, v, k, t, all_t_subsets, t_subset_to_idx)
                if is_valid_covering(reduced_blocks, v, k, t, all_t_subsets, t_subset_to_idx):
                    return [list(b) for b in reduced_blocks]
        
        # Last resort: return empty or trivial? 
        # The task requires valid. If we fail, score is 0. 
        # But the greedy is robust. 
        return [list(b) for b in blocks] if blocks else []

    return [list(b) for b in best_blocks]

def reduce_blocks(blocks, v, k, t, all_t_subsets, t_subset_to_idx):
    """
    Remove blocks that do not uniquely cover any t-subset.
    """
    if not blocks:
        return blocks
    
    # Count coverage for each t-subset
    cover_count = [0] * len(all_t_subsets)
    
    # Map block index to its t-subsets indices for faster removal
    block_t_subsets = []
    
    for b in blocks:
        ts_indices = []
        for ts in itertools.combinations(b, t):
            idx = t_subset_to_idx.get(ts)
            if idx is not None:
                cover_count[idx] += 1
                ts_indices.append(idx)
        block_t_subsets.append(ts_indices)
    
    # Keep a copy of blocks to iterate
    # We want to remove blocks if all their t-subsets are covered by other blocks.
    # A block is redundant if for every t-subset it covers, cover_count[idx] > 1 
    # AND that other coverage is provided by blocks that are *not* this one.
    # Simpler: Iteratively remove.
    
    # Sort blocks by number of unique contributions? 
    # Or just iterate.
    
    # To be efficient, we can mark blocks as "removable" if they don't have any 
    # t-subset with cover_count == 1.
    # However, removing a block might make another block removable.
    
    # Let's do a pass.
    # 1. Identify blocks with at least one "private" t-subset (cover_count == 1).
    #    These MUST stay.
    # 2. Blocks without private t-subsets are candidates for removal.
    # 3. Remove candidates one by one, updating counts, and check if new blocks become removable.
    
    # Optimization: 
    # If a block has a private t-subset, it stays.
    # If not, it might be removed.
    
    # Let's just do a simple iterative removal.
    # Since we want to minimize blocks, we should remove as many as possible.
    
    # To speed up, we can process blocks in order of "least useful" first?
    # Or just random order.
    
    # Let's use a set of active block indices.
    active_indices = set(range(len(blocks)))
    
    # We need to re-evaluate cover counts dynamically.
    # This is O(N^2 * C(k,t)) in worst case, but N is usually small (hundreds).
    
    # Efficient approach:
    # 1. Compute initial cover_count.
    # 2. Create a list of blocks that have NO private t-subsets.
    # 3. Try to remove them.
    
    # Actually, a standard "greedy reduction" is:
    # For each block in the list (maybe sorted by size of contribution?):
    #   Tentatively remove it.
    #   Check if any t-subset becomes uncovered.
    #   If yes, put it back.
    #   If no, keep it removed.
    
    # This is O(N * C(k,t) * lookup). 
    # N ~ 300, C(8,5) = 56. 300 * 56 = 16800 ops. Very fast.
    
    # Let's implement this.
    
    # We need to maintain cover_count.
    # Start with the full count.
    
    # To make removal safe, we should process blocks that are "most redundant" first?
    # Or just iterate.
    
    # Let's iterate through indices.
    # We will maintain a list of blocks to keep.
    
    # Re-initialize cover_count for the reduction phase to be safe
    cover_count = [0] * len(all_t_subsets)
    block_t_subsets = []
    for b in blocks:
        ts_indices = []
        for ts in itertools.combinations(b, t):
            idx = t_subset_to_idx.get(ts)
            if idx is not None:
                cover_count[idx] += 1
                ts_indices.append(idx)
        block_t_subsets.append(ts_indices)
        
    # List of indices to try removing
    # We can try removing in random order to avoid bias
    indices_to_try = list(range(len(blocks)))
    random.shuffle(indices_to_try)
    
    kept_indices = set(range(len(blocks)))
    
    for idx in indices_to_try:
        if idx not in kept_indices:
            continue
            
        # Tentatively remove block idx
        # Check if any of its t-subsets have cover_count == 1
        removable = True
        for ts_idx in block_t_subsets[idx]:
            if cover_count[ts_idx] == 1:
                removable = False
                break
        
        if removable:
            # Actually remove it
            kept_indices.discard(idx)
            for ts_idx in block_t_subsets[idx]:
                cover_count[ts_idx] -= 1
            # Note: Removing this block might make other blocks removable 
            # (if they relied on this block's removal to become non-private? 
            # No, removing a block DECREASES cover counts. 
            # A block becomes removable if it has NO private t-subsets.
            # Private means cover_count == 1.
            # If we remove a block, cover_counts decrease.
            # This might turn a cover_count of 2 into 1, making a t-subset 
            # "private" to the remaining blocks that cover it.
            # So removing a block can make OTHER blocks LESS removable (more essential).
            # Therefore, the order matters. 
            # If we remove a redundant block, we might "lock in" other blocks.
            # This is a known issue. 
            # To mitigate, we could do multiple passes or sort by "least essential".
            # For now, single pass is standard and effective.
            pass
    
    reduced_blocks = [blocks[i] for i in sorted(kept_indices)]
    return reduced_blocks

def is_valid_covering(blocks, v, k, t, all_t_subsets, t_subset_to_idx):
    """Check if blocks cover all t-subsets."""
    if not blocks:
        return len(all_t_subsets) == 0
        
    covered = set()
    for b in blocks:
        for ts in itertools.combinations(b, t):
            idx = t_subset_to_idx.get(ts)
            if idx is not None:
                covered.add(idx)
    return len(covered) == len(all_t_subsets)

if __name__ == "__main__":
    # Test
    blocks = construct(19, 8, 5)
    print(f"Number of blocks: {len(blocks)}")
    # Verify
    all_t_subsets = list(itertools.combinations(range(19), 5))
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    valid = is_valid_covering(blocks, 19, 8, 5, all_t_subsets, t_subset_to_idx)
    print(f"Valid: {valid}")

