"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using global greedy with precomputed masks and efficient reduction.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    N_t = len(all_t_subsets)
    subset_to_id = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Precompute all k-blocks and their coverage masks
    all_blocks = list(itertools.combinations(points, k))
    N_blocks = len(all_blocks)
    
    # masks[i] is a boolean array of length N_t, True if block i covers t-subset j
    # To save memory and time, we can use a sparse representation or just compute on the fly if N_blocks is small
    # For v=19, k=8, N_blocks=75582, N_t=11628. 
    # A dense boolean matrix would be 75582 * 11628 * 1 byte ~ 880 MB. This is too large for typical memory limits or slow to initialize.
    # Instead, we compute gains on the fly or use a more efficient data structure.
    # However, for v=19, t=5, k=8, we can optimize the gain calculation.
    
    # Alternative: Precompute for each t-subset, the list of blocks that cover it?
    # Or just use the "Local Greedy" but with faster gain calculation using precomputed point contributions?
    
    # Let's stick to a hybrid: 
    # 1. Fast Local Greedy with precomputed "point-pair" or "point-subset" gains?
    # Actually, the previous approach (p8) used numpy bitmasks for the *uncovered* set and precomputed masks for *blocks*?
    # If memory is an issue, we can't precompute all block masks.
    
    # Let's try a different strategy:
    # Use the "Global Greedy" but compute gains efficiently.
    # Gain of block B = number of uncovered t-subsets in B.
    # We can maintain `uncovered` as a numpy array of bools.
    # To find the best block, we iterate through all blocks and compute `np.sum(uncovered & mask_b)`.
    # This is O(N_blocks * N_t) per step. 75k * 11k = 800M ops. Too slow in Python loop.
    
    # We need a smarter greedy.
    # The "Local Greedy" (building block point by point) is O(k * v * C(k-1, t-1)) per point.
    # C(7, 4) = 35. 8 points * 19 points * 35 = 5320 ops per block.
    # Number of blocks ~ 300. Total ops ~ 1.5M. This is very fast.
    # The bottleneck in the current code is the *reduction* and the *random restarts* overhead.
    
    # Improvement:
    # 1. Optimize the Local Greedy to be deterministic or use better tie-breaking.
    # 2. Implement a fast reduction step using coverage counts.
    
    def get_block_mask(block):
        """Returns a set of t-subsets covered by block."""
        return set(itertools.combinations(block, t))

    def greedy_cover(seed=None):
        """
        Runs a local greedy construction.
        """
        if seed is not None:
            random.seed(seed)
        
        uncovered = set(all_t_subsets)
        blocks = []
        
        while uncovered:
            # Pick a random uncovered t-subset to anchor the block
            # This introduces randomness to escape local minima
            start_subset = random.choice(list(uncovered))
            block = list(start_subset)
            
            while len(block) < k:
                best_gain = -1
                best_point = None
                # Precompute combinations of current block of size t-1
                # To speed up, we can cache these? No, block changes.
                current_combos = list(itertools.combinations(block, t - 1))
                
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in current_combos:
                        cand = tuple(sorted(rest + (x,)))
                        if cand in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                
                if best_point is None:
                    # Should not happen if uncovered is not empty and v > k
                    break
                block.append(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for comb in itertools.combinations(block, t):
                uncovered.discard(comb)
        
        return blocks

    def reduce_blocks(blocks):
        """
        Efficient reduction using coverage counts.
        """
        if not blocks:
            return blocks
        
        # Map each t-subset to a list of block indices covering it?
        # Or just maintain a count array.
        # Since N_t is ~11k, a count array is fine.
        
        # Initial counts
        counts = np.zeros(N_t, dtype=np.int32)
        for b in blocks:
            for comb in itertools.combinations(b, t):
                counts[subset_to_id[comb]] += 1
        
        reduced_blocks = list(blocks)
        changed = True
        
        # To make reduction faster, we can try to remove blocks that cover many "non-essential" subsets.
        # A block is removable if for all t-subsets it covers, count > 1.
        
        # We can order blocks by "redundancy score" or just iterate.
        # Iterating and checking is O(N_blocks * C(k, t)).
        
        while changed and time.time() < deadline:
            changed = False
            # Shuffle to break symmetry
            indices = list(range(len(reduced_blocks)))
            random.shuffle(indices)
            
            for idx in indices:
                if time.time() > deadline - 5:
                    break
                
                block = reduced_blocks[idx]
                # Check if block is redundant
                redundant = True
                for comb in itertools.combinations(block, t):
                    sid = subset_to_id[comb]
                    if counts[sid] == 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove block
                    # Decrement counts
                    for comb in itertools.combinations(block, t):
                        sid = subset_to_id[comb]
                        counts[sid] -= 1
                    
                    reduced_blocks.pop(idx)
                    changed = True
        
        return reduced_blocks

    best_blocks = None
    best_count = float('inf')
    
    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        seed = restart_count * 12345 + 999
        blocks = greedy_cover(seed=seed)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
        # Try reduction on the best found so far
        if time.time() < deadline - 10:
            reduced = reduce_blocks(best_blocks)
            if len(reduced) < best_count:
                best_count = len(reduced)
                best_blocks = reduced
    
    if best_blocks is None:
        # Fallback
        best_blocks = greedy_cover(seed=0)
    
    return best_blocks
# EVOLVE-BLOCK-END
