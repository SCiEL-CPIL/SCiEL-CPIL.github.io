import itertools
import random
import time
import numpy as np
from collections import Counter

# Precomputation for v=24, k=7, t=3
V, K, T = 24, 7, 3
ALL_TSETS = list(itertools.combinations(range(V), T))
NUM_TSETS = len(ALL_TSETS)
TSET_TO_IDX = {ts: i for i, ts in enumerate(ALL_TSETS)}

# For each tset, list of blocks that contain it
# A block is a tuple of 7 points.
# C(24,7) = 346104. C(21,4)=5985 blocks per tset.
# Total entries: 2024 * 5985 = 12,113,640.
# Storing as list of lists of ints (block index) or block tuples?
# Block tuples take 7*8=56 bytes + overhead. 12M * 56 = 672MB. Too big.
# Let's store block index (int). 12M * 4 = 48MB. OK.
# But we need to convert block index back to points?
# We can precompute all blocks as a list of tuples, and have a map from block_tuple to index?
# Or just generate block tuples on the fly from the index?
# C(24,7) is 346k. We can store all blocks as a list of frozensets or tuples.
# Let's store `ALL_BLOCKS` as a list of tuples.
# And `TSET_TO_BLOCKS` as a list of lists of indices into `ALL_BLOCKS`.

# Precomputation time check:
# Iterating 346k blocks, for each 35 tsets. 346k * 35 = 12M operations. Should be fast (<1s).

ALL_BLOCKS = list(itertools.combinations(range(V), K))
BLOCK_TO_IDX = {b: i for i, b in enumerate(ALL_BLOCKS)}
TSET_TO_BLOCKS = [[] for _ in range(NUM_TSETS)]

for b_idx, block in enumerate(ALL_BLOCKS):
    for ts in itertools.combinations(block, T):
        ts_idx = TSET_TO_IDX[ts]
        TSET_TO_BLOCKS[ts_idx].append(b_idx)

# Convert TSET_TO_BLOCKS to numpy arrays for faster iteration?
# Or just keep as lists.

# Precompute for each block, the list of tset indices it contains.
BLOCK_TO_TSETS = []
for block in ALL_BLOCKS:
    block_to_tsets_idx = [TSET_TO_IDX[ts] for ts in itertools.combinations(block, T)]
    BLOCK_TO_TSETS.append(block_to_tsets_idx)

# Precompute intersection of TSET_TO_BLOCKS for pairs?
# Too many pairs. Compute on the fly.

def _greedy_covering(v, k, t, rng, time_limit, tsets, tset_idx):
    """
    Fast greedy using precomputed TSET_TO_BLOCKS.
    """
    deadline = time.time() + time_limit
    uncovered = bytearray([1] * NUM_TSETS)
    uncovered_count = NUM_TSETS
    blocks = []
    
    # To speed up "pick random uncovered", maintain a list of uncovered indices?
    # Or just random sample until hit.
    # With 2024 tsets, random sampling is very fast.
    
    while uncovered_count > 0 and time.time() < deadline:
        # Pick random uncovered tset
        while True:
            idx = rng.randint(0, NUM_TSETS - 1)
            if uncovered[idx]:
                break
        
        # Get candidate blocks
        candidates = TSET_TO_BLOCKS[idx]
        best_block_idx = -1
        best_gain = -1
        
        # Iterate candidates
        # For each candidate block, calculate gain.
        # Gain = number of tsets in block that are uncovered.
        # Block has 35 tsets.
        # 5985 candidates * 35 checks = 209k ops per step.
        # 76 steps * 209k = 16M ops. Fast.
        
        # Optimization: If we find a block with gain == 35 (covers all its tsets), we can't beat it?
        # Actually, we want max gain. Max possible is 35.
        
        for b_idx in candidates:
            gain = 0
            for ts_idx in BLOCK_TO_TSETS[b_idx]:
                if uncovered[ts_idx]:
                    gain += 1
                    if gain > best_gain:
                        # Early break if we hit max possible?
                        # Max gain for a block containing this tset is 35.
                        # If gain == 35, we can break?
                        # Yes, 35 is the max.
                        if gain == 35:
                            break
            
            if gain > best_gain:
                best_gain = gain
                best_block_idx = b_idx
        
        if best_block_idx == -1:
            break
            
        # Add block
        block_tuple = ALL_BLOCKS[best_block_idx]
        blocks.append(list(block_tuple))
        
        # Update uncovered
        for ts_idx in BLOCK_TO_TSETS[best_block_idx]:
            if uncovered[ts_idx]:
                uncovered[ts_idx] = 0
                uncovered_count -= 1
        
        # Check time
        if time.time() > deadline:
            break

    return blocks

def _local_search(blocks, v, k, t, rng, time_limit, tsets, tset_idx):
    """
    Local search: Remove redundant, then Swap.
    """
    deadline = time.time() + time_limit
    blocks = [tuple(b) for b in blocks]
    
    # State
    tset_count = Counter()
    block_tsets = []
    
    def build_state(blocks):
        nonlocal tset_count, block_tsets
        tset_count = Counter()
        block_tsets = []
        for b in blocks:
            b_tsets = [TSET_TO_IDX[ts] for ts in itertools.combinations(b, t)]
            block_tsets.append(b_tsets)
            for ts_idx in b_tsets:
                tset_count[ts_idx] += 1
    
    build_state(blocks)
    
    def remove_redundant():
        nonlocal blocks, tset_count, block_tsets
        changed = True
        while changed and time.time() < deadline:
            changed = False
            new_blocks = []
            new_block_tsets = []
            new_tset_count = Counter()
            
            for i, b in enumerate(blocks):
                b_tsets = block_tsets[i]
                is_critical = False
                for ts_idx in b_tsets:
                    if tset_count[ts_idx] == 1:
                        is_critical = True
                        break
                
                if is_critical:
                    new_blocks.append(b)
                    new_block_tsets.append(b_tsets)
                    for ts_idx in b_tsets:
                        new_tset_count[ts_idx] += 1
                else:
                    changed = True
            
            if changed:
                blocks = new_blocks
                block_tsets = new_block_tsets
                tset_count = new_tset_count

    def try_swap():
        nonlocal blocks, tset_count, block_tsets
        n = len(blocks)
        if n == 0:
            return False
        
        idx = rng.randint(0, n - 1)
        old_block = blocks[idx]
        old_tsets = block_tsets[idx]
        
        # Find newly uncovered
        newly_uncovered = []
        for ts_idx in old_tsets:
            if tset_count[ts_idx] == 1:
                newly_uncovered.append(ts_idx)
        
        if not newly_uncovered:
            # Redundant
            blocks.pop(idx)
            block_tsets.pop(idx)
            for ts_idx in old_tsets:
                tset_count[ts_idx] -= 1
            return True
        
        # Find replacement
        # We need a block that covers ALL of newly_uncovered.
        # If len(newly_uncovered) > 2, unlikely to find a single block (since a block covers 35 tsets, but they must be specific).
        # Actually, if newly_uncovered has > 2 elements, they must be subsets of the new block.
        # So the new block must contain the union of all points in newly_uncovered.
        # If union size > k, impossible.
        
        union_points = set()
        for ts_idx in newly_uncovered:
            union_points.update(tsets[ts_idx])
        
        if len(union_points) > k:
            return False
        
        # Candidate blocks must contain union_points.
        # How to find them?
        # We can pick the first tset in newly_uncovered, get its blocks, and filter.
        # Or if len(newly_uncovered) == 1, just filter TSET_TO_BLOCKS.
        # If len(newly_uncovered) == 2, intersection.
        
        target_tsets = newly_uncovered
        if len(target_tsets) == 1:
            candidates = TSET_TO_BLOCKS[target_tsets[0]]
        elif len(target_tsets) == 2:
            # Intersection
            set1 = set(TSET_TO_BLOCKS[target_tsets[0]])
            candidates = [b_idx for b_idx in TSET_TO_BLOCKS[target_tsets[1]] if b_idx in set1]
        else:
            # > 2, very rare, just pick first and filter
            candidates = TSET_TO_BLOCKS[target_tsets[0]]
        
        # Filter candidates that cover all newly_uncovered
        found = -1
        for b_idx in candidates:
            b_tsets = set(BLOCK_TO_TSETS[b_idx])
            # Check if all newly_uncovered are in b_tsets
            if all(ts_idx in b_tsets for ts_idx in newly_uncovered):
                # Also check if this block is not the old block (or if it is, it's a no-op swap, skip)
                if ALL_BLOCKS[b_idx] != old_block:
                    found = b_idx
                    break
        
        if found == -1:
            return False
        
        # Perform swap
        new_block = ALL_BLOCKS[found]
        
        # Update state
        for ts_idx in old_tsets:
            tset_count[ts_idx] -= 1
        
        new_tsets_list = BLOCK_TO_TSETS[found]
        for ts_idx in new_tsets_list:
            tset_count[ts_idx] += 1
        
        blocks[idx] = new_block
        block_tsets[idx] = new_tsets_list
        
        return True
    
    # Main loop
    while time.time() < deadline:
        remove_redundant()
        
        # Try a few swaps per iteration to avoid stalling
        for _ in range(50):
            if time.time() > deadline:
                break
            if not try_swap():
                # Try a different random block?
                # If one fails, the state hasn't changed, so another random pick might work.
                continue
        
        # If we can't remove anything, we are stuck.
        # To escape local minima, we could add a random block and remove two? (2-opt)
        # But for now, let's just rely on multiple restarts.
        
        # Check if we can do better? No, local search only minimizes.
        
        # If no progress in a while, break?
        # For simplicity, let it run until deadline.
        
        # Optimization: If no swaps succeeded in the last 50 tries, we are likely stuck.
        # We can break and let the outer loop restart with a different seed.
        # How to track?
        # Let's just run until deadline.
        
    return blocks

def construct(v, k, t):
    start_time = time.time()
    total_budget = 145.0 # 150s limit, leave 5s margin
    
    # Precomputation is global, so it happens on import.
    # Check if precomputation is done?
    # It is done at module level.
    
    best_blocks = None
    best_len = float('inf')
    
    # Run multiple greedy + local search instances
    # Greedy takes ~0.5s. Local search takes remaining.
    # Let's do 5 instances.
    num_restarts = 5
    per_restart_time = (total_budget - 2.0) / num_restarts
    
    for i in range(num_restarts):
        remaining = total_budget - (time.time() - start_time) - 2.0
        if remaining < 1.0:
            break
        
        rng = random.Random(1000 + i)
        
        # Greedy
        greedy_time = min(1.0, remaining * 0.2)
        blocks = _greedy_covering(v, k, t, rng, greedy_time, ALL_TSETS, TSET_TO_IDX)
        
        # Local Search
        ls_time = remaining - greedy_time - 0.5
        if ls_time > 0.5:
            refined = _local_search(blocks, v, k, t, rng, ls_time, ALL_TSETS, TSET_TO_IDX)
            blocks = refined
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
    if best_blocks is None:
        # Fallback
        rng = random.Random(0)
        best_blocks = _greedy_covering(v, k, t, rng, 2.0, ALL_TSETS, TSET_TO_IDX)
        
    return best_blocks
