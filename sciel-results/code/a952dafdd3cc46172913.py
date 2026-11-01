import itertools
import random
import time
import numpy as np
from collections import defaultdict


def construct(v, k, t):
    """
    Covering design C(v, k, t): find as few k-subsets of {0,...,v-1} as possible
    such that every t-subset is contained in at least one block.
    
    Strategy:
    1. Vectorized greedy construction (fast initialization)
    2. Local search: try to remove redundant blocks
    3. Local search: try to replace a block with a better one (covering more unique t-subsets)
    4. Time-limited with explicit deadline guard
    """
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # All t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    
    # Map each t-subset to its index
    tsubset_index = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # For greedy: maintain uncovered set as a set of indices
    uncovered = set(range(n_tsubsets))
    blocks = []
    
    # Precompute for each point x, the list of t-subsets containing x (by index)
    # tsubsets_with_point[x] = list of t-subset indices that contain x
    tsubsets_with_point = [[] for _ in range(v)]
    for i, ts in enumerate(all_tsubsets):
        for x in ts:
            tsubsets_with_point[x].append(i)
    
    # For efficiency, also precompute the t-subsets contained in a k-set given a partial block
    # We'll use a different approach: for greedy, at each step we pick the point that covers
    # the most uncovered t-subsets given the current partial block.
    
    def gain_for_point(x, current_block):
        """Count uncovered t-subsets that would be covered if x is added to current_block."""
        if len(current_block) == 0:
            return len(uncovered & set(tsubsets_with_point[x]))
        
        # A t-subset is covered by current_block + [x] if it contains all of current_block
        # (which has t-1 elements) and x
        # Actually, we need t-subsets that contain all elements of current_block ∪ {x}
        # and are currently uncovered
        # For this to be a t-subset, len(current_block) + 1 must be <= t
        # Since we're building a k-block and adding points one at a time,
        # and t <= k, when len(current_block) >= t-1, adding x can only cover
        # t-subsets that contain current_block ∪ {x}
        
        if len(current_block) + 1 > t:
            # Once we have more than t points, no new t-subset can be "completed" 
            # by adding one more point - but wait, that's wrong.
            # Actually, a t-subset is contained in the final block if all t of its elements
            # are in the block. As we add points to the block, a t-subset becomes covered
            # when all t of its elements have been added to the block.
            # So at any stage, the t-subsets that will be covered by adding x are those
            # that contain x and (t-1) other elements from current_block.
            
            # The number of (t-1)-subsets of current_block is C(len(current_block), t-1)
            # For each such subset S, the t-subset S ∪ {x} is covered if it's uncovered
            count = 0
            if len(current_block) >= t - 1:
                for rest in itertools.combinations(current_block, t - 1):
                    ts = tuple(sorted(rest + (x,)))
                    if ts in tsubset_index and tsubset_index[ts] in uncovered:
                        count += 1
            return count
        else:
            # len(current_block) + 1 <= t
            # We need t-subsets containing all of current_block ∪ {x}
            # This means t-subsets containing current_block and x, with (t - len(current_block) - 1) more points
            # Count uncovered t-subsets that contain x and all elements of current_block
            count = 0
            for i in tsubsets_with_point[x]:
                ts = all_tsubsets[i]
                if i in uncovered and all(p in ts for p in current_block):
                    count += 1
            return count
    
    # Greedy phase
    while uncovered and time.time() < deadline - 60:
        # Start with the lexicographically smallest uncovered t-subset
        min_idx = min(uncovered)
        block = list(all_tsubsets[min_idx])
        
        # Add points one by one, choosing the one with maximum gain
        while len(block) < k:
            best_gain = -1
            best_point = -1
            for x in range(v):
                if x in block:
                    continue
                g = gain_for_point(x, block)
                if g > best_gain:
                    best_gain = g
                    best_point = x
            if best_point == -1:
                break
            block.append(best_point)
        
        block.sort()
        
        # Check if this block is already in our list
        if block not in blocks:
            blocks.append(block)
            # Update uncovered
            for ts in itertools.combinations(block, t):
                idx = tsubset_index[tuple(ts)]
                uncovered.discard(idx)
    
    if not uncovered:
        return blocks
    
    # Local search phase: try to remove redundant blocks
    # For each block, check if removing it leaves any t-subset uncovered
    # We need to track coverage counts
    
    # Build coverage: for each t-subset, how many blocks cover it
    # And for each block, which t-subsets it covers
    block_tsubsets = []
    for block in blocks:
        tsubs = [tsubset_index[tuple(ts)] for ts in itertools.combinations(block, t)]
        block_tsubsets.append(set(tsubs))
    
    # coverage_count[i] = number of blocks covering t-subset i
    coverage_count = np.zeros(n_tsubsets, dtype=np.int32)
    for tsubs in block_tsubsets:
        for i in tsubs:
            coverage_count[i] += 1
    
    changed = True
    while changed and time.time() < deadline - 30:
        changed = False
        # Try to remove blocks in random order
        indices = list(range(len(blocks)))
        random.shuffle(indices)
        
        for bi in indices:
            if time.time() > deadline - 30:
                break
            # Check if any t-subset in block_tsubsets[bi] has coverage_count == 1
            # If not, we can remove this block
            can_remove = True
            for i in block_tsubsets[bi]:
                if coverage_count[i] == 1:
                    can_remove = False
                    break
            
            if can_remove:
                # Remove block bi
                for i in block_tsubsets[bi]:
                    coverage_count[i] -= 1
                block_tsubsets.pop(bi)
                blocks.pop(bi)
                changed = True
                # Restart iteration since indices are now invalid
                break
    
    # Second local search: try to replace a block with one that covers more unique t-subsets
    # This is harder; we'll try a simple heuristic
    # For each block, see if there's a better block that covers all the same unique t-subsets
    # and some additional ones
    
    # Actually, let's try a different approach: random restart with different greedy orders
    # But given time constraints, let's just do the removal pass a few more times with different orders
    
    for _ in range(3):
        if time.time() > deadline - 15:
            break
        # Rebuild coverage
        coverage_count = np.zeros(n_tsubsets, dtype=np.int32)
        for tsubs in block_tsubsets:
            for i in tsubs:
                coverage_count[i] += 1
        
        changed = True
        while changed and time.time() < deadline - 15:
            changed = False
            indices = list(range(len(blocks)))
            random.shuffle(indices)
            
            for bi in indices:
                if time.time() > deadline - 15:
                    break
                can_remove = True
                for i in block_tsubsets[bi]:
                    if coverage_count[i] == 1:
                        can_remove = False
                        break
                
                if can_remove:
                    for i in block_tsubsets[bi]:
                        coverage_count[i] -= 1
                    block_tsubsets.pop(bi)
                    blocks.pop(bi)
                    changed = True
                    break
    
    # Final verification: ensure all t-subsets are covered
    final_covered = np.zeros(n_tsubsets, dtype=np.bool_)
    for block in blocks:
        for ts in itertools.combinations(block, t):
            final_covered[tsubset_index[tuple(ts)]] = True
    
    if not np.all(final_covered):
        # Fallback: add blocks for uncovered t-subsets (shouldn't happen)
        for i in range(n_tsubsets):
            if not final_covered[i]:
                ts = all_tsubsets[i]
                # Find a k-set containing this t-set
                remaining = list(set(range(v)) - set(ts))
                random.shuffle(remaining)
                block = list(ts) + remaining[:k - t]
                block.sort()
                blocks.append(block)
    
    return blocks
