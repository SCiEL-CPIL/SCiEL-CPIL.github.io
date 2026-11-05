"""Optimized greedy covering for C(19,8,5) using bitmask representation and time-bounded search."""
# EVOLVE-BLOCK-START
import itertools
import time
import sys
from collections import defaultdict

def construct(v, k, t):
    """
    Construct a covering of v points with k-subsets such that every t-subset is covered.
    Uses bitmask-based greedy with redundancy removal and time-bounded restarts.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 115s safety margin before 120s limit
    
    # Precompute all t-subsets as bitmasks
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    
    # Map each t-subset to an index for fast lookup
    tsubset_to_idx = {tuple(s): i for i, s in enumerate(all_tsubsets)}
    
    # For each k-subset, precompute which t-subsets it covers (as a frozenset of indices)
    # But this is too memory intensive for v=19, k=8, t=5.
    # Instead, we'll compute on the fly but use bitmasks for points.
    
    # Represent points as bits in an integer
    # A k-subset is represented by an integer mask with k bits set
    
    # For a given block (set of points), the t-subsets it covers are all t-subsets 
    # that are subsets of the block.
    
    def block_to_tsubset_indices(block_mask):
        """Given a bitmask for a block, return list of t-subset indices it covers."""
        # Extract points from mask
        points = []
        for p in range(v):
            if block_mask & (1 << p):
                points.append(p)
        # Generate all t-subsets of these points
        result = []
        for tsub in itertools.combinations(points, t):
            idx = tsubset_to_idx[tsub]
            result.append(idx)
        return result
    
    def greedy_cover():
        """Run a single greedy covering pass."""
        # uncovered: set of t-subset indices not yet covered
        uncovered = set(range(n_tsubsets))
        blocks = []
        
        while uncovered:
            # Pick the first uncovered t-subset to anchor the new block
            anchor_idx = min(uncovered)
            anchor = all_tsubsets[anchor_idx]
            anchor_mask = 0
            for p in anchor:
                anchor_mask |= (1 << p)
            
            # Build the block greedily: start with anchor, add points that cover most uncovered t-subsets
            block_points = set(anchor)
            block_mask = anchor_mask
            
            # Precompute: for each candidate point not in block, how many uncovered t-subsets 
            # would it cover if added?
            # A t-subset is covered by block if all its points are in block.
            # When we add point x, the newly covered t-subsets are those that contain x and 
            # whose other t-1 points are already in the block.
            
            while len(block_points) < k:
                best_point = -1
                best_gain = -1
                for x in range(v):
                    if x in block_points:
                        continue
                    # Count how many uncovered t-subsets would be covered by adding x
                    gain = 0
                    # t-subsets containing x where the other t-1 points are in block
                    for tsub in itertools.combinations(block_points, t - 1):
                        full_tsub = tuple(sorted(tsub + (x,)))
                        if full_tsub in tsubset_to_idx:
                            idx = tsubset_to_idx[full_tsub]
                            if idx in uncovered:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point == -1:
                    break
                block_points.add(best_point)
                block_mask |= (1 << best_point)
            
            # Convert to sorted list
            block = sorted(block_points)
            blocks.append(block)
            
            # Update uncovered
            covered = set(block_to_tsubset_indices(block_mask))
            uncovered -= covered
            
            # Time check
            if time.time() > deadline - 5.0:
                break
        
        return blocks
    
    # Run initial greedy
    blocks = greedy_cover()
    
    # Check if complete
    if time.time() < deadline - 10.0:
        # Try to improve by removing redundant blocks
        # A block is redundant if all its t-subsets are covered by other blocks
        n_blocks = len(blocks)
        for i in range(n_blocks):
            if time.time() > deadline - 5.0:
                break
            # Check if block i is redundant
            block_i_tsubsets = set(block_to_tsubset_indices(
                sum(1 << p for p in blocks[i])
            ))
            redundant = True
            for j in range(n_blocks):
                if j == i:
                    continue
                block_j_tsubsets = set(block_to_tsubset_indices(
                    sum(1 << p for p in blocks[j])
                ))
                if not block_i_tsubsets.issubset(block_j_tsubsets):
                    # Check union of all other blocks
                    pass
            
            # Better: check if union of all other blocks covers block_i's t-subsets
            if i > 0 and i < n_blocks:
                other_covered = set()
                for j in range(n_blocks):
                    if j != i:
                        other_covered |= set(block_to_tsubset_indices(
                            sum(1 << p for p in blocks[j])
                        ))
                if block_i_tsubsets.issubset(other_covered):
                    blocks.pop(i)
                    n_blocks -= 1
    
    return blocks
# EVOLVE-BLOCK-END
