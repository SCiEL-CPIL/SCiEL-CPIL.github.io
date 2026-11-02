"""Combined program for C(19,7,4) covering design.
Uses efficient greedy construction with precomputed data structures,
followed by local search to reduce block count."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict


def construct(v, k, t):
    deadline = time.time() + 115  # safety margin under 120s limit
    
    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    
    # Map each t-subset to its index for O(1) lookup
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    # Track which t-subsets are covered
    covered = [False] * n_t
    blocks = []
    
    def time_ok():
        return time.time() < deadline
    
    # Precompute for each point, the list of t-subsets containing it (as indices)
    point_tsubs = [[] for _ in range(v)]
    for i, sub in enumerate(all_t_subsets):
        for pt in sub:
            point_tsubs[pt].append(i)
    
    def add_block(block):
        """Cover all t-subsets within block."""
        for sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[sub]
            covered[idx] = True
        blocks.append(tuple(block))
    
    def uncovered_count_for_point_in_block(block_set, x):
        """Count how many uncovered t-subsets would be covered by adding x to block."""
        if x in block_set:
            return 0
        count = 0
        for rest in itertools.combinations(block_set, t - 1):
            sub = tuple(sorted(rest + (x,)))
            idx = t_sub_to_idx[sub]
            if not covered[idx]:
                count += 1
        return count
    
    # Greedy construction
    while True:
        if not time_ok():
            break
        
        # Find the first uncovered t-subset
        uncovered_idx = None
        for i in range(n_t):
            if not covered[i]:
                uncovered_idx = i
                break
        if uncovered_idx is None:
            break
        
        # Start block with the t-subset
        seed_subset = all_t_subsets[uncovered_idx]
        block_set = set(seed_subset)
        
        # Extend block to size k
        while len(block_set) < k:
            if not time_ok():
                break
            
            best_x = None
            best_gain = -1
            for x in all_points:
                if x in block_set:
                    continue
                gain = uncovered_count_for_point_in_block(block_set, x)
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x is None:
                break
            block_set.add(best_x)
        
        if not time_ok():
            break
        
        # Add the block
        block = tuple(sorted(block_set))
        add_block(block)
    
    # Local improvement: try to reduce block count
    if time_ok() and blocks:
        blocks = _local_search(v, k, t, blocks, deadline)
    
    return [list(b) for b in blocks]


def _local_search(v, k, t, blocks, deadline):
    """Try to reduce block count via local search."""
    if time.time() > deadline - 5:
        return blocks
    
    # Build coverage map: for each t-subset, which blocks cover it
    cover_map = defaultdict(list)
    for i, block in enumerate(blocks):
        for tsub in itertools.combinations(block, t):
            cover_map[tsub].append(i)
    
    best_blocks = list(blocks)
    best_count = len(best_blocks)
    
    # Try removing redundant blocks
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        for i in range(len(best_blocks) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            
            # Check if block i is redundant (all its t-subsets covered by other blocks)
            redundant = True
            for tsub in itertools.combinations(best_blocks[i], t):
                if len(cover_map[tsub]) <= 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove block i
                removed_block = best_blocks[i]
                del best_blocks[i]
                # Update cover_map
                for tsub in itertools.combinations(removed_block, t):
                    if i in cover_map[tsub]:
                        cover_map[tsub].remove(i)
                # Reindex
                for tsub in cover_map:
                    cover_map[tsub] = [j if j < i else j - 1 for j in cover_map[tsub]]
                best_count = len(best_blocks)
                changed = True
                break  # Restart loop since indices changed
    
    # Try swapping blocks: replace a block with a better one that allows removing another
    random.seed(42)
    attempts = 0
    max_attempts = 2000
    
    while time.time() < deadline - 5 and attempts < max_attempts:
        attempts += 1
        
        # Pick a random block to try replacing
        idx = random.randint(0, len(best_blocks) - 1)
        old_block = best_blocks[idx]
        
        # Find t-subsets only covered by this block
        unique_tsubs = []
        for tsub in itertools.combinations(old_block, t):
            if len(cover_map[tsub]) == 1:
                unique_tsubs.append(tsub)
        
        if not unique_tsubs:
            # Block is redundant, remove it
            removed_block = best_blocks[idx]
            del best_blocks[idx]
            for tsub in itertools.combinations(removed_block, t):
                if idx in cover_map[tsub]:
                    cover_map[tsub].remove(idx)
            for tsub in cover_map:
                cover_map[tsub] = [j if j < idx else j - 1 for j in cover_map[tsub]]
            best_count = len(best_blocks)
            continue
        
        # Try to find a new block that covers all unique_tsubs
        # The new block must contain all points that are in any unique_tsub
        union_size = len(set(pt for tsub in unique_tsubs for pt in tsub))
        if union_size > k:
            continue
        
        # The new block must contain the union of all unique_tsubs
        candidate = set(pt for tsub in unique_tsubs for pt in tsub)
        
        # Fill up to k elements
        while len(candidate) < k:
            # Add a point that covers the most uncovered t-subsets
            best_point = None
            best_gain = -1
            for x in range(v):
                if x in candidate:
                    continue
                gain = 0
                for tsub in unique_tsubs:
                    if x not in tsub:
                        continue
                    # Check if adding x would help cover this tsub
                    if all(pt in candidate for pt in tsub if pt != x):
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point is None:
                break
            candidate.add(best_point)
        
        if len(candidate) != k:
            continue
        
        new_block = tuple(sorted(candidate))
        
        # Verify new block covers all unique_tsubs
        valid = True
        for tsub in unique_tsubs:
            if not all(pt in new_block for pt in tsub):
                valid = False
                break
        
        if not valid:
            continue
        
        # Check if new block is already in best_blocks
        if new_block in best_blocks:
            continue
        
        # Accept the replacement
        old_block_set = set(old_block)
        new_block_set = set(new_block)
        
        # Update cover_map: remove old block's contributions, add new block's
        for tsub in itertools.combinations(old_block, t):
            if idx in cover_map[tsub]:
                cover_map[tsub].remove(idx)
        
        for tsub in itertools.combinations(new_block, t):
            if tsub not in cover_map:
                cover_map[tsub] = []
            cover_map[tsub].append(idx)
        
        best_blocks[idx] = new_block
    
    return best_blocks
# EVOLVE-BLOCK-END
