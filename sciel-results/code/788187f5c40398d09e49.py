"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import Counter


def construct(v, k, t):
    # Greedy covering with multiple passes and local improvement.
    # We use a deadline guard to ensure we finish within 120s.
    deadline = time.time() + 115.0  # safety margin

    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    
    # Map each t-subset to an index for fast lookup
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # uncovered: set of indices of uncovered t-subsets
    uncovered = set(range(n_tsubsets))
    blocks = []
    
    # Precompute for each block (tuple of sorted points) the list of t-subset indices it covers
    # We'll compute this on the fly but cache it
    
    def block_covers_tsubsets(block):
        """Return list of t-subset indices covered by this block."""
        result = []
        for ts in itertools.combinations(block, t):
            result.append(tsubset_to_idx[ts])
        return result
    
    # Greedy phase: repeatedly add the block that covers the most uncovered t-subsets
    # To speed this up, we use a heuristic to find good blocks
    
    while uncovered and time.time() < deadline:
        # Find the best block to add
        # Strategy: start with a random uncovered t-subset, then greedily extend
        # to maximize coverage of uncovered t-subsets
        
        best_block = None
        best_gain = -1
        
        # Try several candidate starting t-subsets
        sample_size = min(20, len(uncovered))
        sample = random.sample(uncovered, sample_size)
        
        for start_idx in sample:
            start_ts = all_tsubsets[start_idx]
            block = list(start_ts)
            
            # Greedily extend the block
            while len(block) < k:
                # Find the point that covers the most uncovered t-subsets
                best_point = None
                best_point_gain = -1
                
                for x in range(v):
                    if x in block:
                        continue
                    
                    # Calculate gain: how many new uncovered t-subsets would be covered
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        new_ts = tuple(sorted(rest + (x,)))
                        if new_ts in tsubset_to_idx:
                            idx = tsubset_to_idx[new_ts]
                            if idx in uncovered:
                                gain += 1
                    
                    if gain > best_point_gain:
                        best_point_gain = gain
                        best_point = x
                
                if best_point is None:
                    break
                block.append(best_point)
            
            block.sort()
            
            # Calculate total gain for this block
            gain = 0
            for ts in itertools.combinations(block, t):
                idx = tsubset_to_idx[ts]
                if idx in uncovered:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_block = block
        
        if best_block is None or best_gain <= 0:
            # Fallback: just take any uncovered t-subset and extend arbitrarily
            start_idx = next(iter(uncovered))
            start_ts = all_tsubsets[start_idx]
            block = list(start_ts)
            while len(block) < k:
                for x in range(v):
                    if x not in block:
                        block.append(x)
                        break
            block.sort()
            best_block = block
        
        blocks.append(best_block)
        
        # Update uncovered
        for ts in itertools.combinations(best_block, t):
            idx = tsubset_to_idx[ts]
            uncovered.discard(idx)
        
        if time.time() > deadline - 5:
            break
    
    # If we still have uncovered t-subsets, we need to add more blocks
    # But first, let's try to improve the current solution
    
    if uncovered and time.time() < deadline - 10:
        # Try to remove some blocks and re-cover
        # This is a local search: try removing a block if all its t-subsets are covered by others
        
        def is_block_redundant(block_idx):
            """Check if removing block_idx would leave any t-subset uncovered."""
            block = blocks[block_idx]
            for ts in itertools.combinations(block, t):
                idx = tsubset_to_idx[ts]
                # Check if this t-subset is covered by any other block
                covered = False
                for j, other_block in enumerate(blocks):
                    if j == block_idx:
                        continue
                    if all(x in other_block for x in ts):
                        covered = True
                        break
                if not covered:
                    return False
            return True
        
        # Try to remove redundant blocks
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                if is_block_redundant(i):
                    blocks.pop(i)
                    changed = True
    
    # Final verification and completion if needed
    if uncovered:
        # Add blocks for remaining uncovered t-subsets
        while uncovered and time.time() < deadline:
            start_idx = next(iter(uncovered))
            start_ts = all_tsubsets[start_idx]
            block = list(start_ts)
            while len(block) < k:
                # Add points that cover the most uncovered t-subsets
                best_point = None
                best_point_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        new_ts = tuple(sorted(rest + (x,)))
                        if new_ts in tsubset_to_idx:
                            idx = tsubset_to_idx[new_ts]
                            if idx in uncovered:
                                gain += 1
                    if gain > best_point_gain:
                        best_point_gain = gain
                        best_point = x
                if best_point is None:
                    for x in range(v):
                        if x not in block:
                            best_point = x
                            break
                block.append(best_point)
            block.sort()
            blocks.append(block)
            for ts in itertools.combinations(block, t):
                idx = tsubset_to_idx[ts]
                uncovered.discard(idx)
    
    return blocks
# EVOLVE-BLOCK-END
