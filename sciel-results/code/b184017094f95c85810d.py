"""Optimized greedy covering for C(v,k,t) with time guard and local search refinement."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

def _all_tsubsets(v, t):
    """Generate all t-subsets of {0,...,v-1} as tuples."""
    return list(itertools.combinations(range(v), t))

def _block_tsubsets(block, t):
    """Generate all t-subsets contained in a block."""
    return set(itertools.combinations(block, t))

def _greedy_construct(v, k, t, deadline):
    """
    Fast greedy covering using bitsets for coverage tracking.
    Returns list of blocks (each a sorted list of ints).
    """
    # Precompute all t-subsets
    all_subs = _all_tsubsets(v, t)
    n_subs = len(all_subs)
    
    # Map each t-subset to an index
    sub_to_idx = {s: i for i, s in enumerate(all_subs)}
    
    # Coverage: for each point x, which t-subsets would be covered if x were added
    # to a set S? We track incrementally.
    
    # For efficiency, precompute for each point x, the list of t-subsets containing x
    point_subs = [[] for _ in range(v)]
    for idx, sub in enumerate(all_subs):
        for x in sub:
            point_subs[x].append(idx)
    
    # Uncovered mask: bit i is 1 if sub i is uncovered
    uncovered = np.ones(n_subs, dtype=np.int32)
    n_uncovered = n_subs
    
    blocks = []
    
    while n_uncovered > 0:
        if time.time() > deadline:
            break
        
        # Find the first uncovered subset to start the block
        # Find index of first uncovered
        unc_idx = np.where(uncovered > 0)[0]
        if len(unc_idx) == 0:
            break
        start_sub = all_subs[unc_idx[0]]
        
        block = list(start_sub)
        
        # Build the block greedily
        while len(block) < k:
            if time.time() > deadline:
                break
            
            # For each candidate point x not in block, compute how many currently
            # uncovered t-subsets would be covered by adding x to the current partial block.
            # A t-subset is covered by adding x if:
            #   - it contains x
            #   - the remaining t-1 elements are a subset of the current block
            #   - it is currently uncovered
            
            best_x = -1
            best_gain = -1
            
            for x in range(v):
                if x in block:
                    continue
                
                # Count uncovered t-subsets containing x where the other t-1 elements
                # are all in the current block
                gain = 0
                # We need: sub contains x, sub\{x} ⊆ block, and sub is uncovered
                # Equivalent: for each (t-1)-subset of block, check if sub = that + x is uncovered
                if len(block) >= t - 1:
                    for rest in itertools.combinations(block, t - 1):
                        sub = tuple(sorted(rest + (x,)))
                        idx = sub_to_idx.get(sub)
                        if idx is not None and uncovered[idx] > 0:
                            gain += 1
                elif len(block) == t - 2:
                    # Need one more element from block... actually t-1 elements from block
                    # but block only has t-2 elements, so no t-subset can be fully covered
                    # unless we consider that the t-subset needs t-1 elements from block
                    # and 1 is x. But block has only t-2 elements, so we can't form t-1
                    # from block. Gain = 0.
                    pass
                else:
                    gain = 0
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x == -1 or best_gain <= 0:
                # No point adds coverage; pick any remaining point
                for x in range(v):
                    if x not in block:
                        best_x = x
                        break
                if best_x == -1:
                    break
            
            block.append(best_x)
        
        if len(block) < k:
            break
        
        block.sort()
        blocks.append(block)
        
        # Update coverage
        for sub in _block_tsubsets(block, t):
            idx = sub_to_idx[sub]
            if uncovered[idx] > 0:
                uncovered[idx] = 0
                n_uncovered -= 1
    
    return blocks

def _check_covered(blocks, v, t):
    """Check if all t-subsets are covered."""
    covered = set()
    for block in blocks:
        for sub in itertools.combinations(block, t):
            covered.add(sub)
    return len(covered) == len(list(itertools.combinations(range(v), t)))

def _local_search(blocks, v, k, t, deadline):
    """
    Try to reduce the number of blocks by removing redundant blocks and
    trying to merge/replace blocks.
    """
    if not blocks:
        return blocks
    
    all_subs = _all_tsubsets(v, t)
    n_subs = len(all_subs)
    sub_to_idx = {s: i for i, s in enumerate(all_subs)}
    
    # Compute coverage count for each t-subset
    cover_count = np.zeros(n_subs, dtype=np.int32)
    for block in blocks:
        for sub in itertools.combinations(block, t):
            idx = sub_to_idx[sub]
            cover_count[idx] += 1
    
    n_uncovered = np.sum(cover_count == 0)
    if n_uncovered > 0:
        return blocks  # Not a valid covering
    
    # Try to remove blocks that don't reduce coverage below 1 for any subset
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n_blocks = len(blocks)
        for i in range(n_blocks):
            if time.time() > deadline:
                break
            # Check if block i is redundant
            # Temporarily remove it
            for sub in itertools.combinations(blocks[i], t):
                idx = sub_to_idx[sub]
                cover_count[idx] -= 1
            if np.all(cover_count >= 1):
                # Block is redundant, remove it
                blocks.pop(i)
                improved = True
                break
            else:
                # Restore
                for sub in itertools.combinations(blocks[i], t):
                    idx = sub_to_idx[sub]
                    cover_count[idx] += 1
    
    # Try to replace pairs of blocks with a single block
    # This is expensive, so only do it if we have time
    if time.time() < deadline - 10:
        for i in range(len(blocks)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(blocks)):
                if time.time() > deadline:
                    break
                # Try to find a k-subset that covers all t-subsets covered by blocks[i] and blocks[j]
                # that are only covered by these two blocks
                # This is complex; skip for now
                pass
    
    return blocks

def construct(v, k, t):
    """
    Construct a covering of {0,...,v-1} with k-subsets such that every t-subset
    is contained in at least one block.
    
    Returns a list of blocks, each a sorted list of k distinct integers.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Quick sanity check
    if t > k:
        return []
    if t == 0:
        return []
    
    # Run greedy
    blocks = _greedy_construct(v, k, t, deadline)
    
    # If we ran out of time, blocks might not be a valid covering
    # Check validity
    if time.time() < deadline - 5:
        # Verify coverage
        all_subs = _all_tsubsets(v, t)
        covered = set()
        for block in blocks:
            for sub in itertools.combinations(block, t):
                covered.add(sub)
        
        if len(covered) == len(all_subs):
            # Valid covering, try local search to improve
            blocks = _local_search(blocks, v, k, t, deadline)
        else:
            # Not valid, need to add more blocks
            # Continue greedy from where we left off
            # Find uncovered subsets
            uncovered = [s for s in all_subs if s not in covered]
            while uncovered and time.time() < deadline:
                sub = uncovered[0]
                block = list(sub)
                while len(block) < k:
                    if time.time() > deadline:
                        break
                    best_x = -1
                    best_gain = -1
                    for x in range(v):
                        if x in block:
                            continue
                        gain = 0
                        if len(block) >= t - 1:
                            for rest in itertools.combinations(block, t - 1):
                                s2 = tuple(sorted(rest + (x,)))
                                if s2 in set(uncovered):
                                    gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                    if best_x == -1:
                        for x in range(v):
                            if x not in block:
                                best_x = x
                                break
                        if best_x == -1:
                            break
                    block.append(best_x)
                if len(block) < k:
                    break
                block.sort()
                blocks.append(block)
                # Update uncovered
                block_subs = set(itertools.combinations(block, t))
                uncovered = [s for s in uncovered if s not in block_subs]
    
    # Final check: if not valid, return what we have (will score 0 but better than timeout)
    # Ensure blocks are distinct
    seen = set()
    unique_blocks = []
    for b in blocks:
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            unique_blocks.append(b)
    
    return unique_blocks
# EVOLVE-BLOCK-END
