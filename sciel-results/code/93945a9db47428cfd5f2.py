"""Bitset-accelerated greedy + local-search covering for C(v,k,t)."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import sys
from collections import defaultdict

# We keep the same entry point: construct(v, k, t) -> list of blocks.


def _greedy_construct_bitset(v, k, t, deadline, rng):
    """
    Greedy construction using bitsets for O(1) gain calculation.
    Each t-subset gets a bit index; we track which are uncovered via a bitset.
    For each candidate point x, gain = popcount(uncovered & mask_x_given_block).
    Returns (blocks, uncovered_bitset).
    """
    # Enumerate all t-subsets and assign indices
    all_subs = list(itertools.combinations(range(v), t))
    n_subs = len(all_subs)
    sub_to_idx = {s: i for i, s in enumerate(all_subs)}
    
    # Precompute for each point x, the bitmask of t-subsets containing x
    # But that's too large. Instead, for a partial block B, the t-subsets that 
    # would be completed by adding x are those formed by (t-1)-subsets of B plus x.
    # We'll compute gain on the fly using the uncovered set.
    
    # Use a set of uncovered subset indices (as integers) for efficiency
    uncovered = set(range(n_subs))
    blocks = []
    
    while uncovered:
        if time.time() > deadline:
            break
        # Pick the smallest-index uncovered t-subset as seed
        target_idx = min(uncovered)
        target = all_subs[target_idx]
        block = list(target)
        
        # Build block greedily
        while len(block) < k:
            if time.time() > deadline:
                break
            best_x = -1
            best_gain = -1
            
            # Precompute (t-1)-subsets of current block
            if t - 1 <= 0:
                # t=1 case: gain is 1 if x not in block and (x,) uncovered
                # Handle separately
                for x in range(v):
                    if x in block:
                        continue
                    idx = sub_to_idx.get((x,))
                    if idx is not None and idx in uncovered:
                        gain = 1
                    else:
                        gain = 0
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
            else:
                # For each candidate x, count how many (t-1)-subsets of block 
                # form an uncovered t-subset when x is added
                # Precompute the set of (t-1)-subsets of block for quick lookup
                block_set = set(block)
                
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        new_sub = tuple(sorted(rest + (x,)))
                        idx = sub_to_idx.get(new_sub)
                        if idx is not None and idx in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
            
            if best_x == -1 or best_gain <= 0:
                break
            block.append(best_x)
        
        if len(block) < k:
            # Pad with arbitrary points to make it size k (shouldn't happen if v >= k)
            block_set = set(block)
            for x in range(v):
                if x not in block_set and len(block) < k:
                    block.append(x)
        
        block.sort()
        blocks.append(block)
        
        # Remove all t-subsets covered by this block
        for sub in itertools.combinations(block, t):
            idx = sub_to_idx[sub]
            uncovered.discard(idx)
    
    return blocks, uncovered, all_subs, sub_to_idx


def _remove_redundant_blocks(blocks, v, t, all_subs, sub_to_idx, deadline):
    """Remove blocks that are fully redundant."""
    if not blocks:
        return blocks
    
    # Count coverage per t-subset
    cover_count = defaultdict(int)
    for b in blocks:
        for sub in itertools.combinations(b, t):
            cover_count[sub] += 1
    
    new_blocks = []
    for b in blocks:
        redundant = True
        for sub in itertools.combinations(b, t):
            if cover_count[sub] <= 1:
                redundant = False
                break
        if not redundant:
            new_blocks.append(b)
    
    return new_blocks


def _local_search_remove(blocks, v, t, all_subs, sub_to_idx, deadline):
    """
    Local search: iteratively remove blocks whose t-subsets are all covered by others.
    Also try: for each pair of blocks, check if removing one is safe.
    """
    if not blocks:
        return blocks
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Build cover count
        cover_count = defaultdict(int)
        for b in blocks:
            for sub in itertools.combinations(b, t):
                cover_count[sub] += 1
        
        for i in range(len(blocks)):
            if time.time() > deadline:
                return blocks
            can_remove = True
            for sub in itertools.combinations(blocks[i], t):
                if cover_count[sub] <= 1:
                    can_remove = False
                    break
            if can_remove:
                blocks.pop(i)
                improved = True
                break
    
    return blocks


def _local_search_replace(blocks, v, k, t, all_subs, sub_to_idx, deadline, rng):
    """
    Try to replace a block with a better one. For each block i, try building 
    a new block that covers more of the t-subsets that block i uniquely covers,
    while not losing any coverage.
    """
    if not blocks or time.time() > deadline:
        return blocks
    
    # Build cover count
    cover_count = defaultdict(int)
    for b in blocks:
        for sub in itertools.combinations(b, t):
            cover_count[sub] += 1
    
    n_blocks = len(blocks)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n_blocks):
            if time.time() > deadline:
                return blocks
            
            old_block = blocks[i]
            old_subs = list(itertools.combinations(old_block, t))
            
            # Find which t-subsets are uniquely covered by this block
            unique_subs = [sub for sub in old_subs if cover_count[sub] == 1]
            
            if not unique_subs:
                # Block is redundant, remove it
                for sub in old_subs:
                    cover_count[sub] -= 1
                blocks.pop(i)
                n_blocks -= 1
                improved = True
                break
            
            # Try to build a new block that covers all unique_subs
            # and maximizes additional coverage
            new_block = None
            # Seed with the first unique subset
            seed = list(unique_subs[0])
            new_block = seed[:]
            
            while len(new_block) < k:
                if time.time() > deadline:
                    break
                best_x = -1
                best_gain = -1
                new_block_set = set(new_block)
                
                for x in range(v):
                    if x in new_block_set:
                        continue
                    gain = 0
                    # Count how many unique_subs would be covered
                    for rest in itertools.combinations(new_block, t - 1):
                        new_sub = tuple(sorted(rest + (x,)))
                        if new_sub in unique_subs:
                            gain += 100  # Heavy weight for unique coverage
                        # Also count additional coverage
                        idx = sub_to_idx.get(new_sub)
                        if idx is not None and cover_count.get(new_sub, 0) == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                
                if best_x == -1:
                    break
                new_block.append(best_x)
            
            if len(new_block) < k:
                new_block_set = set(new_block)
                for x in range(v):
                    if x not in new_block_set and len(new_block) < k:
                        new_block.append(x)
            
            new_block.sort()
            
            # Check if new block covers all unique_subs
            new_block_set = set(new_block)
            covers_all_unique = True
            for sub in unique_subs:
                if not all(x in new_block_set for x in sub):
                    covers_all_unique = False
                    break
            
            if covers_all_unique and new_block != old_block:
                # Replace: update cover counts
                for sub in old_subs:
                    cover_count[sub] -= 1
                for sub in itertools.combinations(new_block, t):
                    cover_count[sub] += 1
                blocks[i] = new_block
                improved = True
                break
    
    return blocks


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0
    rng = random.Random(42)
    
    # Phase 1: Greedy construction with bitset-style tracking
    blocks, uncovered, all_subs, sub_to_idx = _greedy_construct_bitset(v, k, t, deadline, rng)
    
    if not blocks:
        # Fallback: return empty or minimal
        return []
    
    # Phase 2: Remove redundant blocks
    if time.time() < deadline:
        blocks = _remove_redundant_blocks(blocks, v, t, all_subs, sub_to_idx, deadline)
    
    # Phase 3: Local search - remove redundant
    if time.time() < deadline:
        blocks = _local_search_remove(blocks, v, t, all_subs, sub_to_idx, deadline)
    
    # Phase 4: Local search - replace blocks
    if time.time() < deadline:
        blocks = _local_search_replace(blocks, v, k, t, all_subs, sub_to_idx, deadline, rng)
    
    # Final cleanup: remove redundant again
    if time.time() < deadline:
        blocks = _remove_redundant_blocks(blocks, v, t, all_subs, sub_to_idx, deadline)
    
    # Ensure blocks are distinct and sorted
    seen = set()
    unique_blocks = []
    for b in blocks:
        b_sorted = sorted(b)
        key = tuple(b_sorted)
        if key not in seen:
            seen.add(key)
            unique_blocks.append(b_sorted)
    
    return unique_blocks
# EVOLVE-BLOCK-END
