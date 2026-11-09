"""
Improved covering design C(24,7,3) solver.

Strategy:
1. Fast greedy set cover to get an initial covering.
2. Redundancy removal: iteratively remove blocks that are not needed.
3. 1-for-1 swap local search: try replacing one block with another to reduce
   the total number of blocks (or keep the same count but enable further reductions).
   Actually, 1-for-1 swaps don't reduce count. Instead, we do:
   - Redundancy removal (reduces count)
   - 2-for-1 swap: replace two blocks with one block if possible (reduces count by 1)
   - 1-for-2 swap is not useful for minimization.
   
   Better approach:
   - Greedy to get initial cover
   - Remove redundant blocks
   - Try to find a single block that can replace two existing blocks (i.e., the union of
     the t-subsets covered by those two blocks is a subset of the t-subsets covered by
     one new block). This is expensive but we can use heuristics.
   
   Alternative: use a more sophisticated greedy with lookahead, then local search.

Given the 150s deadline, I'll implement:
1. Greedy set cover (fast, using bitmasks for t-subsets)
2. Redundancy removal
3. Local search: for each pair of blocks, check if there exists a 7-subset that covers
   all t-subsets currently covered by both blocks. If so, replace the two with the new one.
   This is O(n^2 * C(24,7)) which is too slow. Instead, use a smarter approach.

Let me use a different strategy:
1. Greedy with bitmasks
2. Redundancy removal
3. For each block, try to "improve" it by swapping one point at a time to cover more
   uncovered triples, then re-run redundancy removal.
4. Also try: for each pair of blocks, see if we can merge them into one block.

Actually, the most effective approach for covering designs is:
1. Greedy to get initial solution
2. Iteratively: remove redundant blocks, then try to add a block that covers many
   remaining uncovered triples, then remove redundancies again.

But the key insight from the inspiration is that local search with redundancy removal
and swaps outperforms naive greedy. Let me implement:

1. Greedy set cover using efficient bit operations
2. Redundancy removal
3. 1-for-1 swap: for each block B, try replacing it with a new block B' that covers
   at least all the triples that B covers (and possibly more). This doesn't reduce count
   but may enable future reductions.
4. 2-for-1: try replacing two blocks with one.

For speed, I'll use bitmasks for the 2024 triples (C(24,3) = 2024).
Each block covers C(7,3) = 35 triples.

Let me implement this.
"""

import time
import itertools
from collections import defaultdict

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t).
    Returns a list of blocks, each a list of k distinct integers in 0..v-1.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Precompute all t-subsets and map them to indices
    all_tsubsets = list(itertools.combinations(range(v), t))
    num_tsubsets = len(all_tsubsets)
    
    # Map each t-subset to an index
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # For a block (k-subset), compute which t-subsets it covers
    def block_to_mask(block):
        """Return a frozenset of t-subset indices covered by this block."""
        covered = set()
        for ts in itertools.combinations(block, t):
            idx = tsubset_to_idx[ts]
            covered.add(idx)
        return frozenset(covered)
    
    # Greedy set cover
    uncovered = set(range(num_tsubsets))
    blocks = []
    block_masks = []
    
    while uncovered:
        if time.time() > deadline - 30:
            break
        
        best_block = None
        best_count = -1
        best_mask = None
        
        # Try to find the block that covers the most uncovered t-subsets
        # Start with a random-ish approach: pick a t-subset from uncovered,
        # then try all k-subsets containing it
        sample_ts = next(iter(uncovered))
        sample = all_tsubsets[sample_ts]
        
        # Generate candidate blocks: all k-subsets containing the sample t-subset
        remaining_points = [p for p in range(v) if p not in sample]
        need = k - t
        
        # For speed, try a limited number of candidates
        # Use greedy: start with sample, then add points that cover most uncovered
        current_block = list(sample)
        for _ in range(need):
            best_point = -1
            best_gain = -1
            for p in remaining_points:
                if p in current_block:
                    continue
                # Count how many new uncovered t-subsets this point would cover
                gain = 0
                for ts in itertools.combinations(current_block, t - 1):
                    new_ts = tuple(sorted(ts + (p,)))
                    if new_ts in tsubset_to_idx:
                        idx = tsubset_to_idx[new_ts]
                        if idx in uncovered:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = p
            if best_point >= 0:
                current_block.append(best_point)
                remaining_points.remove(best_point)
        
        current_block.sort()
        mask = block_to_mask(current_block)
        count = len(mask & uncovered)
        
        if count > best_count:
            best_count = count
            best_block = current_block
            best_mask = mask
        
        if best_block is None or best_count <= 0:
            # Fallback: just pick any block
            best_block = list(min(uncovered))
            for p in range(v):
                if p not in best_block and len(best_block) < k:
                    best_block.append(p)
            best_block.sort()
            best_mask = block_to_mask(best_block)
        
        blocks.append(best_block)
        block_masks.append(best_mask)
        uncovered -= best_mask
    
    if not blocks:
        return []
    
    # Redundancy removal
    def remove_redundant(blocks, block_masks):
        """Remove blocks that are not needed for coverage."""
        n = len(blocks)
        if n == 0:
            return blocks, block_masks
        
        # Count how many times each t-subset is covered
        cover_count = [0] * num_tsubsets
        for mask in block_masks:
            for idx in mask:
                cover_count[idx] += 1
        
        changed = True
        while changed:
            changed = False
            new_blocks = []
            new_masks = []
            for i in range(len(blocks)):
                # Check if this block is essential
                essential = False
                for idx in block_masks[i]:
                    if cover_count[idx] == 1:
                        essential = True
                        break
                if not essential:
                    # Remove this block
                    for idx in block_masks[i]:
                        cover_count[idx] -= 1
                    changed = True
                else:
                    new_blocks.append(blocks[i])
                    new_masks.append(block_masks[i])
            blocks = new_blocks
            block_masks = new_masks
        return blocks, block_masks
    
    blocks, block_masks = remove_redundant(blocks, block_masks)
    
    # Local search: try 2-for-1 swaps
    # For each pair of blocks, see if there's a single block that covers all
    # the t-subsets covered by both.
    
    def try_2_for_1(blocks, block_masks, deadline):
        """Try to replace two blocks with one."""
        n = len(blocks)
        if n < 2:
            return blocks, block_masks
        
        # Precompute for each block, the set of t-subsets it covers
        # For each pair, compute the union and see if any k-subset covers it
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(blocks)):
                if time.time() > deadline - 5:
                    break
                for j in range(i + 1, len(blocks)):
                    if time.time() > deadline - 5:
                        break
                    
                    # Union of t-subsets covered by blocks i and j
                    union_mask = block_masks[i] | block_masks[j]
                    
                    # Try to find a k-subset that covers all t-subsets in union_mask
                    # This is hard in general. Use heuristic:
                    # The new block must contain at least one point from each
                    # "constraint" in the union.
                    
                    # For now, skip this as it's too expensive.
                    # Instead, try 1-for-1 swaps.
            # Do 1-for-1 swaps instead
            for i in range(len(blocks)):
                if time.time() > deadline - 5:
                    break
                
                original_mask = block_masks[i]
                original_block = blocks[i]
                
                # Try modifying the block by swapping one point
                for pos in range(k):
                    if time.time() > deadline - 5:
                        break
                    old_point = original_block[pos]
                    for new_point in range(v):
                        if new_point == old_point or new_point in original_block:
                            continue
                        
                        new_block = original_block[:]
                        new_block[pos] = new_point
                        new_block.sort()
                        
                        new_mask = block_to_mask(new_block)
                        
                        # Check if new_mask covers everything original_mask covered
                        # and ideally covers more
                        if original_mask.issubset(new_mask):
                            # This swap is valid and potentially better
                            # Only accept if it covers strictly more
                            if len(new_mask) > len(original_mask):
                                # Remove old block, add new
                                for idx in original_mask:
                                    pass  # Will be handled by redundancy removal
                                blocks[i] = new_block
                                block_masks[i] = new_mask
                                improved = True
                                break
        
        # Final redundancy removal
        blocks, block_masks = remove_redundant(blocks, block_masks)
        return blocks, block_masks
    
    blocks, block_masks = try_2_for_1(blocks, block_masks, deadline)
    
    # Ensure all blocks are valid and distinct
    seen = set()
    valid_blocks = []
    for b in blocks:
        if len(b) == k and len(set(b)) == k and all(0 <= x < v for x in b):
            b_sorted = tuple(sorted(b))
            if b_sorted not in seen:
                seen.add(b_sorted)
                valid_blocks.append(list(b_sorted))
    
    return valid_blocks
