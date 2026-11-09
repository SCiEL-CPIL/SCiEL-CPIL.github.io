"""Covering design C(v,k,t) constructor with greedy + local search."""
import itertools
import time
import random
import sys
from collections import defaultdict


def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t).
    Returns a list of blocks (each a sorted list of k distinct integers in 0..v-1).
    """
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin for 150s limit
    
    # Precompute all t-subsets as tuples
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subsets)
    
    # Map each t-subset to an index for fast lookup
    t_sub_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    def time_ok():
        return time.time() < deadline
    
    # Phase 1: Greedy construction
    # uncovered[i] = True if t-subset i is not yet covered
    uncovered = [True] * num_t
    uncovered_count = num_t
    blocks = []
    
    # For speed, we'll track for each point, which t-subsets it helps cover
    # Actually, let's use a simpler greedy: pick the block that covers the most uncovered t-subsets
    
    # Precompute for each k-subset, the list of t-subset indices it covers
    # This is too memory intensive for large v. Instead, compute on the fly.
    
    def count_covered(block):
        """Count how many uncovered t-subsets are covered by this block."""
        count = 0
        for ts in itertools.combinations(block, t):
            idx = t_sub_to_idx[ts]
            if uncovered[idx]:
                count += 1
        return count
    
    # Greedy: repeatedly pick the k-subset that covers the most uncovered t-subsets
    # To speed this up, we use a heuristic: start from an uncovered t-subset and build a block around it
    
    while uncovered_count > 0 and time_ok():
        # Find an uncovered t-subset to start with
        # Pick the first uncovered one (or a random one for diversity)
        start_idx = -1
        for i in range(num_t):
            if uncovered[i]:
                start_idx = i
                break
        if start_idx < 0:
            break
        
        start_set = all_t_subsets[start_idx]
        
        # Build a block containing start_set that maximizes coverage
        # Start with the t-subset, add points one by one
        block = list(start_set)
        
        # Greedily add points to maximize new coverage
        while len(block) < k and time_ok():
            best_gain = -1
            best_point = -1
            current_block_set = set(block)
            
            for x in range(v):
                if x in current_block_set:
                    continue
                # Count how many uncovered t-subsets would be covered by adding x
                # A t-subset is covered if it's a subset of block + {x}
                # i.e., it's a t-subset of block + {x} that contains x or not
                # More precisely: t-subsets of (block + {x}) that are NOT subsets of block
                # These are t-subsets that contain x and (t-1) elements from block
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    ts = tuple(sorted(rest + (x,)))
                    idx = t_sub_to_idx[ts]
                    if uncovered[idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point < 0 or best_gain <= 0:
                # No improvement, just add a random point to fill up
                remaining = [x for x in range(v) if x not in current_block_set]
                if remaining:
                    best_point = random.choice(remaining)
                else:
                    break
            
            block.append(best_point)
        
        block.sort()
        
        # Add this block
        blocks.append(block)
        
        # Update uncovered
        for ts in itertools.combinations(block, t):
            idx = t_sub_to_idx[ts]
            if uncovered[idx]:
                uncovered[idx] = False
                uncovered_count -= 1
    
    if uncovered_count > 0:
        # Emergency: if we haven't finished, force complete the covering
        # This shouldn't happen with the deadline check, but just in case
        while uncovered_count > 0:
            for i in range(num_t):
                if uncovered[i]:
                    start_set = all_t_subsets[i]
                    block = list(start_set)
                    while len(block) < k:
                        remaining = [x for x in range(v) if x not in block]
                        block.append(random.choice(remaining))
                    block.sort()
                    blocks.append(block)
                    for ts in itertools.combinations(block, t):
                        idx = t_sub_to_idx[ts]
                        if uncovered[idx]:
                            uncovered[idx] = False
                            uncovered_count -= 1
                    break
    
    # Phase 2: Local search - try to remove redundant blocks
    # A block is redundant if all its t-subsets are covered by other blocks
    
    def is_redundant(block_idx, blocks_list):
        """Check if block at block_idx is redundant (all its t-subsets covered by others)."""
        block = blocks_list[block_idx]
        for ts in itertools.combinations(block, t):
            idx = t_sub_to_idx[ts]
            # Check if any OTHER block covers this t-subset
            found = False
            for j, b in enumerate(blocks_list):
                if j == block_idx:
                    continue
                if set(ts).issubset(set(b)):
                    found = True
                    break
            if not found:
                return False
        return True
    
    # Try to remove blocks in random order
    removed = True
    while removed and time_ok():
        removed = False
        random.shuffle(blocks)  # Actually, we should shuffle indices
        for i in range(len(blocks) - 1, -1, -1):
            if not time_ok():
                break
            # Check if block i is redundant
            block = blocks[i]
            redundant = True
            for ts in itertools.combinations(block, t):
                idx = t_sub_to_idx[ts]
                found = False
                for j in range(len(blocks)):
                    if j == i:
                        continue
                    if set(ts).issubset(set(blocks[j])):
                        found = True
                        break
                if not found:
                    redundant = False
                    break
            if redundant:
                del blocks[i]
                removed = True
    
    # Phase 3: Try to merge or replace blocks to reduce count further
    # This is more complex, so we'll do a simple swap-based local search
    
    # Try replacing a block with a better one
    if time_ok():
        for _ in range(50):  # Limited iterations
            if not time_ok():
                break
            if len(blocks) == 0:
                break
            # Pick a random block to try to improve
            idx = random.randrange(len(blocks))
            old_block = blocks[idx]
            
            # Try to find a better block that covers all the t-subsets that old_block uniquely covers
            # For simplicity, try random replacements
            for attempt in range(20):
                if not time_ok():
                    break
                # Generate a random k-subset
                new_block = sorted(random.sample(range(v), k))
                
                # Check if replacing old_block with new_block still covers everything
                # We need: all t-subsets of old_block that are NOT covered by other blocks
                # must be covered by new_block
                unique_ts = []
                for ts in itertools.combinations(old_block, t):
                    idx_ts = t_sub_to_idx[ts]
                    # Check if covered by any block other than idx
                    covered_by_other = False
                    for j in range(len(blocks)):
                        if j == idx:
                            continue
                        if set(ts).issubset(set(blocks[j])):
                            covered_by_other = True
                            break
                    if not covered_by_other:
                        unique_ts.append(ts)
                
                # Check if new_block covers all unique_ts
                valid = True
                for ts in unique_ts:
                    if not set(ts).issubset(set(new_block)):
                        valid = False
                        break
                
                if valid:
                    # Also check if new_block adds coverage (optional, for improvement)
                    # For now, just accept if it's valid
                    blocks[idx] = new_block
                    break
    
    # Final cleanup: remove any duplicate blocks
    seen = set()
    unique_blocks = []
    for b in blocks:
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            unique_blocks.append(b)
    blocks = unique_blocks
    
    return blocks
