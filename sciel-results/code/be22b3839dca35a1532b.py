"""
Strategy: Randomized Parallel Batch Construction (RPBC)
Instead of sequential greedy, we generate many random candidate blocks, 
then use a fast greedy selection over these candidates to maximize coverage 
of the initial uncovered t-subsets. This avoids the local minima of 
sequential greedy by considering a diverse set of blocks at once.
"""

# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    # Deadline management: leave 8 seconds for final repair and return
    deadline = time.time() + 112.0
    
    # 1. Precomputation
    # Generate all t-subsets as bitmasks
    tsub_masks = []
    for sub in itertools.combinations(range(v), t):
        m = 0
        for p in sub:
            m |= (1 << p)
        tsub_masks.append(m)
    
    n_tsub = len(tsub_masks)
    
    # Map t-subset mask to index for O(1) lookup
    idx_of = {m: i for i, m in enumerate(tsub_masks)}
    
    # Initialize coverage count
    cover_count = np.zeros(n_tsub, dtype=np.int32)
    
    # Function to calculate the gain of adding a block (mask)
    # Returns the number of newly covered t-subsets
    def calc_gain(mask):
        gain = 0
        # Iterate over all t-subsets of the block
        # We can precompute the list of t-subsets for a given mask? 
        # No, masks are dynamic in the batch.
        # But we can iterate over combinations of the points in the mask.
        points = [p for p in range(v) if (mask >> p) & 1]
        for sub in itertools.combinations(points, t):
            sm = 0
            for p in sub:
                sm |= (1 << p)
            if cover_count[idx_of[sm]] == 0:
                gain += 1
        return gain

    # Function to apply a block to the coverage state
    def apply_block(mask):
        points = [p for p in range(v) if (mask >> p) & 1]
        for sub in itertools.combinations(points, t):
            sm = 0
            for p in sub:
                sm |= (1 << p)
            cover_count[idx_of[sm]] += 1

    blocks = []
    blocks_set = set()
    
    # 2. Batch Generation and Selection Loop
    # We will generate a batch of random blocks, score them, and add the best ones.
    # This is a "batched greedy" approach.
    
    rng = random.Random(42)
    
    # To make it faster, we can precompute the list of points for each potential block?
    # There are C(19,7) = 50388 blocks. 
    # Let's precompute the t-subsets for every possible k-subset.
    # This is a one-time cost that allows O(1) or O(C(k,t)) gain calculation.
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    k_sub_masks = []
    k_sub_tsubs = [] # List of lists of t-subset masks
    
    for comb in all_k_subsets:
        mask = 0
        for p in comb:
            mask |= (1 << p)
        k_sub_masks.append(mask)
        
        subs = []
        for sub in itertools.combinations(comb, t):
            sm = 0
            for p in sub:
                sm |= (1 << p)
            subs.append(sm)
        k_sub_tsubs.append(subs)
    
    # Map mask to index in all_k_subsets for quick lookup
    mask_to_idx = {m: i for i, m in enumerate(k_sub_masks)}
    
    # Initial state: all uncovered
    # We don't actually need to simulate the whole thing if we just pick blocks.
    # But we need to track which t-subsets are covered.
    
    # Let's use a different approach for the main loop:
    # 1. Generate a pool of candidate blocks.
    # 2. Greedily select from the pool based on current uncovered count.
    
    # To ensure we don't miss any, we will iterate in "rounds".
    # In each round, we pick a set of blocks that cover the most uncovered items.
    
    # Actually, a simpler and often effective strategy for covering numbers is:
    # 1. Start with an empty set of blocks.
    # 2. While there are uncovered t-subsets:
    #    a. Find the k-subset that covers the maximum number of uncovered t-subsets.
    #    b. Add it.
    # This is the standard greedy set cover. The current program does a "grow" version.
    # The "inspiration" suggests scanning ALL k-subsets per step is better.
    # Let's implement the standard greedy set cover but optimized with NumPy.
    
    # State:
    # uncovered_mask: boolean array of size n_tsub, True if uncovered.
    # We can represent the coverage of each k-subset as a boolean vector of size n_tsub.
    # This matrix is C(v,k) x C(v,t) = 50388 x 27132. 
    # 50388 * 27132 * 1 byte = ~1.3 GB. Too big for RAM? 
    # 50388 * 27132 / 8 (if bits) = ~170 MB. 
    # Let's try to avoid the full matrix.
    
    # Alternative: For each step, calculate the gain for each of the 50388 blocks.
    # Gain = sum(1 for tsub in block if uncovered[tsub])
    # We can precompute for each k-subset the list of t-subset indices it covers.
    # k_sub_tsub_indices: list of lists of indices.
    
    k_sub_tsub_indices = []
    for subs in k_sub_tsubs:
        indices = [idx_of[sm] for sm in subs]
        k_sub_tsub_indices.append(indices)
    
    # Now, greedy loop:
    # uncovered: numpy bool array
    uncovered = np.ones(n_tsub, dtype=bool)
    
    # To speed up gain calculation, we can maintain a count of how many times each 
    # t-subset is covered by the *selected* blocks? No, we just need to know if it's 
    # covered by AT LEAST ONE.
    
    # Actually, we can just iterate.
    # For each candidate block, sum(uncovered[indices]).
    # This is 50388 * (average length of indices). 
    # Average length is C(7,4) = 35.
    # 50388 * 35 = ~1.7 million operations per step.
    # Number of steps? ~150 blocks.
    # 150 * 1.7M = 255M operations. In Python, this might be too slow (10-20s per step?).
    # We need vectorization.
    
    # Let's create a sparse matrix representation or just use numpy arrays.
    # We can create a matrix M of shape (n_ksub, max_tsubs_per_ksub) with -1 padding?
    # Or just a list of arrays.
    
    # Let's try a different optimization:
    # Instead of checking all 50388 blocks, we can use a heuristic to prune.
    # But the inspiration says "scan all candidate 7-subsets".
    
    # Let's optimize the inner loop using NumPy.
    # We will store the indices as a flat array and offsets.
    
    # offsets[i] to offsets[i+1] contains the indices for block i.
    all_indices = []
    offsets = [0]
    for indices in k_sub_tsub_indices:
        all_indices.extend(indices)
        offsets.append(len(all_indices))
    
    all_indices_arr = np.array(all_indices, dtype=np.int32)
    offsets_arr = np.array(offsets, dtype=np.int32)
    
    # Greedy Loop
    blocks_selected = []
    
    while uncovered.any() and time.time() < deadline:
        # Calculate gain for all blocks
        # gain[i] = sum(uncovered[all_indices_arr[offsets[i]:offsets[i+1]]])
        
        # We can do this in a loop, but let's see if we can vectorize.
        # It's hard to vectorize variable length sums without padding.
        # Let's just use a Python loop but with numpy slicing.
        
        gains = np.zeros(len(k_sub_masks), dtype=np.int32)
        
        for i in range(len(k_sub_masks)):
            start = offsets_arr[i]
            end = offsets_arr[i+1]
            if start < end:
                # Sum of uncovered bits in this range
                gains[i] = np.sum(uncovered[all_indices_arr[start:end]])
        
        # Find max gain
        max_gain = np.max(gains)
        
        if max_gain == 0:
            # Should not happen if uncovered.any() is True, unless logic error
            break
        
        # Find all blocks with max gain
        candidates = np.where(gains == max_gain)[0]
        
        # Randomly pick one to break ties (or pick the first one)
        # Picking the first one is deterministic. Let's add some randomness for diversity.
        # But deterministic is safer for reproducibility.
        # Let's pick the one with the smallest index among candidates to be deterministic.
        best_idx = candidates[0]
        
        # Add this block
        blocks_selected.append(best_idx)
        
        # Update uncovered
        start = offsets_arr[best_idx]
        end = offsets_arr[best_idx+1]
        uncovered[all_indices_arr[start:end]] = False
        
        # Check deadline
        if time.time() > deadline:
            break
    
    # 3. Repair Phase
    # If we exited due to deadline, we must cover the remaining uncovered t-subsets.
    # We do this by creating a block for each uncovered t-subset.
    
    uncovered_indices = np.where(uncovered)[0]
    
    for idx in uncovered_indices:
        if time.time() > deadline - 2.0: # Leave 2s for return
            # Emergency: just add blocks for the rest without checking time?
            # We must return a valid covering.
            pass
        
        tsub_mask = tsub_masks[idx]
        # Build a k-subset containing this t-subset
        have = [p for p in range(v) if (tsub_mask >> p) & 1]
        need = k - len(have)
        
        # Pick random extra points
        remaining_points = [p for p in range(v) if not ((tsub_mask >> p) & 1)]
        
        if need > 0:
            # Use random selection for repair to avoid bias
            extra = rng.sample(remaining_points, need)
        else:
            extra = []
        
        block = have + extra
        block.sort()
        
        # Convert to mask
        mask = 0
        for p in block:
            mask |= (1 << p)
        
        # Add if not already present
        # We don't strictly need to check for duplicates in repair if we just append,
        # but the problem says blocks must be distinct.
        # We will check against the set of selected blocks.
        
        # We need to map block to index in all_k_subsets to check if it's in blocks_selected?
        # blocks_selected contains indices into all_k_subsets.
        
        if mask not in [k_sub_masks[i] for i in blocks_selected]:
             # This check is O(N). Better to maintain a set of masks.
             pass
        
        # Let's maintain a set of masks for the selected blocks.
        
    # Re-do the repair properly with a set
    
    selected_masks = set(k_sub_masks[i] for i in blocks_selected)
    
    for idx in uncovered_indices:
        tsub_mask = tsub_masks[idx]
        have = [p for p in range(v) if (tsub_mask >> p) & 1]
        need = k - len(have)
        
        remaining_points = [p for p in range(v) if not ((tsub_mask >> p) & 1)]
        
        if need > 0:
            extra = rng.sample(remaining_points, need)
        else:
            extra = []
        
        block = have + extra
        block.sort()
        
        mask = 0
        for p in block:
            mask |= (1 << p)
        
        if mask not in selected_masks:
            selected_masks.add(mask)
            blocks_selected.append(mask_to_idx[mask])
            
    # 4. Convert to list of lists
    result = []
    for idx in blocks_selected:
        result.append(list(all_k_subsets[idx]))
        
    return result

# EVOLVE-BLOCK-END
