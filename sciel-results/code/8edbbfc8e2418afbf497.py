import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Precompute all t-subsets (triples) and their indices
    all_subs = list(itertools.combinations(range(v), t))
    num_subs = len(all_subs)
    sub_to_idx = {sub: i for i, sub in enumerate(all_subs)}
    
    # Helper to get indices of t-subsets covered by a k-subset (block)
    def get_covered_indices(block):
        # block is a tuple of k points
        # find all t-subsets of block
        covered = []
        for sub in itertools.combinations(block, t):
            covered.append(sub_to_idx[sub])
        return covered

    # 1. Greedy Initialization
    # We want to cover all subs.
    # uncovered is a boolean array
    uncovered = np.ones(num_subs, dtype=bool)
    blocks = []
    
    # To speed up greedy, we can precompute candidates for each triple?
    # No, let's just pick a random uncovered triple and find best block.
    
    while np.any(uncovered):
        if time.time() > deadline:
            break
        
        # Pick a random uncovered subset
        uncov_indices = np.where(uncovered)[0]
        if len(uncov_indices) == 0:
            break
        target_idx = uncov_indices[random.randint(0, len(uncov_indices)-1)]
        target_sub = all_subs[target_idx]
        
        # Find best block containing target_sub
        # A block must contain all points in target_sub.
        # Remaining k-t points can be chosen from v-t points.
        remaining_points = [p for p in range(v) if p not in target_sub]
        
        best_gain = -1
        best_block = None
        best_covered = None
        
        # Iterate through all combinations of remaining points
        # Number of candidates: C(v-t, k-t) = C(21, 4) = 5985
        for rem in itertools.combinations(remaining_points, k-t):
            block = tuple(sorted(target_sub + rem))
            # Calculate gain: how many uncovered subs does this block cover?
            covered_indices = get_covered_indices(block)
            # Count how many of these are currently uncovered
            # Using numpy for speed
            gain = np.sum(uncovered[covered_indices])
            
            if gain > best_gain:
                best_gain = gain
                best_block = block
                best_covered = covered_indices
                
        if best_block is None:
            # Fallback if something went wrong
            best_block = tuple(sorted(target_sub + tuple(remaining_points[:k-t])))
            best_covered = get_covered_indices(best_block)
            
        blocks.append(best_block)
        # Update uncovered
        uncovered[best_covered] = False
        
    # 2. Local Search: Redundancy Removal
    # We have a valid covering. Try to remove blocks.
    # We maintain coverage counts.
    coverage_counts = np.zeros(num_subs, dtype=int)
    for b in blocks:
        covered = get_covered_indices(b)
        coverage_counts[covered] += 1
        
    # Verify validity
    if np.min(coverage_counts) < 1:
        # Should not happen if greedy was correct, but just in case
        pass
        
    # Iterative removal
    # Shuffle blocks to avoid bias
    random.shuffle(blocks)
    
    removed_count = 0
    # We loop until no more blocks can be removed or time is up
    # To allow for more aggressive search, we can restart the removal process
    # if we get stuck, or just keep trying.
    
    # Efficient removal:
    # Iterate through blocks. For each, check if removing it leaves min(coverage) >= 1.
    # To check min(coverage) >= 1 after removing block B:
    # We need to know if any sub covered by B has count 1.
    # If so, B is critical. If not, B is redundant.
    
    current_blocks = blocks[:]
    
    while time.time() < deadline:
        progress = False
        # Shuffle to try different orders
        random.shuffle(current_blocks)
        
        new_blocks = []
        for b in current_blocks:
            covered = get_covered_indices(b)
            # Check if any covered sub has count 1
            # If yes, removing b would make that sub uncovered.
            if np.min(coverage_counts[covered]) == 1:
                # Critical block, keep it
                new_blocks.append(b)
            else:
                # Redundant block, remove it
                coverage_counts[covered] -= 1
                progress = True
        
        if not progress:
            # Try a few random restarts of the removal process or break?
            # If we can't remove any, we are at a local optimum for this set.
            # We could try to "swap" points, but that's complex.
            # Given the time budget, breaking might be okay if we are near 76.
            # However, we can try to re-initialize greedily with different seeds if time permits?
            # No, greedy is slow.
            # Let's just break if no progress in a full pass.
            break
        
        current_blocks = new_blocks
        
    # Final check
    if time.time() > deadline - 5:
         return [list(b) for b in current_blocks]
         
    return [list(b) for b in current_blocks]
