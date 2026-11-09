"""Bitmask-accelerated greedy with randomized restarts and redundancy removal for C(19,7,4)."""
import itertools
import random
import time
import numpy as np

# Precompute all 4-subsets of 19 points as bitmasks (19 bits)
_ALL_4SUBSETS = list(itertools.combinations(range(19), 4))
_ALL_4SUBSETS_LEN = len(_ALL_4SUBSETS)
_SUBSET_TO_INDEX = {s: i for i, s in enumerate(_ALL_4SUBSETS)}

# For each 7-block, precompute which 4-subsets it covers as a numpy boolean mask
# There are C(19,7) = 50388 possible blocks. Precompute all.
_ALL_BLOCKS_7 = list(itertools.combinations(range(19), 7))
_ALL_BLOCKS_7_LEN = len(_ALL_BLOCKS_7)

# block_mask[i] = numpy array of _ALL_4SUBSETS_LEN booleans, True if block i covers that 4-subset
# This is 50388 * 3876 = ~195M booleans = ~195MB. Might be too much.
# Alternative: store as list of frozensets or use integer bitmasks.
# 3876 bits fits in 61 uint64s. Let's use numpy uint64 arrays of shape (50388, 61).
# Actually, let's just precompute on-the-fly per block using the index mapping, but cache.

# Better approach: represent each 4-subset as an index 0..3875.
# For a given 7-block, the 4-subsets it covers are C(7,4)=35 specific indices.
# We can precompute for each of the 50388 blocks, the list of 35 indices it covers.
_BLOCK_COVERED_INDICES = []
for block in _ALL_BLOCKS_7:
    block_set = set(block)
    covered = []
    for s in _ALL_4SUBSETS:
        if all(x in block_set for x in s):
            covered.append(_SUBSET_TO_INDEX[s])
    _BLOCK_COVERED_INDICES.append(covered)

# Map from block tuple to its index in _ALL_BLOCKS_7
_BLOCK_TO_IDX = {b: i for i, b in enumerate(_ALL_BLOCKS_7)}


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # We'll do multiple randomized greedy restarts, then pick the best, then do redundancy removal.
    # Each greedy run: maintain a numpy boolean array 'covered' of length 3876.
    # At each step, pick the block that maximizes newly covered subsets (greedy set cover).
    # To speed up: precompute for each block the list of 4-subset indices it covers.
    # Gain of a block = number of indices in its list that are not yet covered.
    
    # For efficiency with many blocks, we can use numpy:
    # covered_mask is a numpy array of uint8 (0 or 1) of length 3876.
    # For each candidate block, gain = sum(covered_mask[indices] == 0 for indices in block_indices)
    # But checking all 50388 blocks each step is slow.
    
    # Alternative: only consider blocks that contain the "most urgent" uncovered subset.
    # Or: use a priority queue / lazy approach.
    
    # Simpler fast approach: 
    # 1. Find an uncovered 4-subset S.
    # 2. Among all 7-blocks containing S, pick the one that covers the most other uncovered 4-subsets.
    # 3. Add it, update covered.
    # This is the standard greedy for covering designs and is much faster.
    
    # Precompute: for each 4-subset index, which block indices contain it.
    # There are C(19-4, 7-4) = C(15,3) = 455 blocks containing any given 4-subset.
    SUBSET_TO_BLOCKS = [[] for _ in range(_ALL_4SUBSETS_LEN)]
    for b_idx, block in enumerate(_ALL_BLOCKS_7):
        block_set = set(block)
        for s in _ALL_4SUBSETS:
            if all(x in block_set for x in s):
                SUBSET_TO_BLOCKS[_SUBSET_TO_INDEX[s]].append(b_idx)
    
    # Verify: each 4-subset should be in exactly 455 blocks
    # assert all(len(lst) == 455 for lst in SUBSET_TO_BLOCKS)
    
    best_blocks = None
    best_count = float('inf')
    
    # Number of restarts: tune based on time. Each restart takes ~O(n_blocks * 455 * 35) operations.
    # Let's try 20-50 restarts.
    
    rng = random.Random(42)
    
    for restart in range(100):
        if time.time() > deadline - 20:  # leave 20s for redundancy removal
            break
        
        covered = np.zeros(_ALL_4SUBSETS_LEN, dtype=np.uint8)
        blocks = []
        used_blocks = set()
        
        while covered.sum() < _ALL_4SUBSETS_LEN:
            if time.time() > deadline - 20:
                break
            
            # Find an uncovered 4-subset. Use random selection among uncovered for diversity.
            uncovered_indices = np.where(covered == 0)[0]
            if len(uncovered_indices) == 0:
                break
            
            # Pick a random uncovered subset (or the first one for determinism in some restarts)
            if restart % 3 == 0:
                # Deterministic: pick first uncovered
                target_idx = uncovered_indices[0]
            else:
                # Random
                target_idx = uncovered_indices[rng.randint(0, len(uncovered_indices) - 1)]
            
            # Get all blocks that contain this 4-subset
            candidate_block_indices = SUBSET_TO_BLOCKS[target_idx]
            
            # For each candidate, compute gain = number of newly covered 4-subsets
            best_gain = -1
            best_block_idx = None
            
            for b_idx in candidate_block_indices:
                if b_idx in used_blocks:
                    continue
                # Gain = how many of this block's covered subsets are currently uncovered
                indices = _BLOCK_COVERED_INDICES[b_idx]
                gain = int(np.sum(covered[indices] == 0))
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            
            if best_block_idx is None:
                # All blocks containing target are already used. Pick any unused block containing it.
                for b_idx in candidate_block_indices:
                    if b_idx not in used_blocks:
                        best_block_idx = b_idx
                        break
                if best_block_idx is None:
                    # Should not happen, but safety
                    break
            
            # Add the block
            blocks.append(_ALL_BLOCKS_7[best_block_idx])
            used_blocks.add(best_block_idx)
            covered[_BLOCK_COVERED_INDICES[best_block_idx]] = 1
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks[:]
        
        # Check if we've hit a good number; if below 140, we can stop early
        if best_count <= 140:
            break
    
    if best_blocks is None:
        # Fallback: should not happen
        best_blocks = []
    
    # Redundancy removal (local search): iteratively try to remove each block.
    # If removing it doesn't uncover any 4-subset, keep it removed.
    # Repeat until no more removals possible.
    
    if time.time() < deadline - 10:
        # Build coverage count for each 4-subset: how many blocks cover it.
        # cov_count[i] = number of blocks in best_blocks that cover 4-subset i
        cov_count = np.zeros(_ALL_4SUBSETS_LEN, dtype=np.int32)
        for block in best_blocks:
            b_idx = _BLOCK_TO_IDX[block]
            cov_count[_BLOCK_COVERED_INDICES[b_idx]] += 1
        
        # Try removing blocks one by one
        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            new_blocks = []
            for block in best_blocks:
                b_idx = _BLOCK_TO_IDX[block]
                indices = _BLOCK_COVERED_INDICES[b_idx]
                # If all 4-subsets covered by this block are also covered by other blocks, we can remove it.
                # i.e., cov_count[i] > 1 for all i in indices
                if np.all(cov_count[indices] > 1):
                    # Remove this block
                    cov_count[indices] -= 1
                    changed = True
                else:
                    new_blocks.append(block)
            best_blocks = new_blocks
    
    # Ensure validity: if for some reason coverage is broken (shouldn't be), add missing blocks.
    # Final check
    final_covered = np.zeros(_ALL_4SUBSETS_LEN, dtype=np.uint8)
    for block in best_blocks:
        b_idx = _BLOCK_TO_IDX[block]
        final_covered[_BLOCK_COVERED_INDICES[b_idx]] = 1
    
    if final_covered.sum() < _ALL_4SUBSETS_LEN:
        # Add blocks to cover remaining
        uncovered = np.where(final_covered == 0)[0]
        for idx in uncovered:
            # Find a block covering this subset
            for b_idx in SUBSET_TO_BLOCKS[idx]:
                block = _ALL_BLOCKS_7[b_idx]
                if block not in best_blocks:
                    best_blocks.append(block)
                    final_covered[_BLOCK_COVERED_INDICES[b_idx]] = 1
                    break
    
    # Convert to list of lists for the interface
    return [list(b) for b in best_blocks]
