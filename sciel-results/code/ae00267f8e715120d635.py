"""Covering design C(24,7,3) via greedy with efficient gain computation and local search."""
import itertools
import time
import random
from collections import defaultdict

def _prune_redundant_blocks(blocks, v, t):
    """
    Removes blocks that do not cover any unique t-subset.
    Returns the pruned list of blocks.
    """
    if not blocks:
        return blocks
    
    # Precompute t-subsets for each block
    block_tsets = []
    for b in blocks:
        tsets = [frozenset(c) for c in itertools.combinations(b, t)]
        block_tsets.append(tsets)
        
    # Count how many blocks cover each t-set
    tset_to_blocks = defaultdict(list)
    for i, tsets in enumerate(block_tsets):
        for ts in tsets:
            tset_to_blocks[ts].append(i)
            
    to_remove = set()
    for i, tsets in enumerate(block_tsets):
        is_redundant = True
        for ts in tsets:
            if len(tset_to_blocks[ts]) == 1:
                # This t-set is ONLY covered by block i
                is_redundant = False
                break
        if is_redundant:
            to_remove.add(i)
            
    if to_remove:
        new_blocks = [b for i, b in enumerate(blocks) if i not in to_remove]
        return new_blocks
    else:
        return blocks

def _local_search(blocks, v, k, t, max_iter=50, deadline=None):
    """
    Try to improve the covering by replacing blocks or removing redundant ones.
    """
    if not blocks:
        return blocks
    
    best_blocks = list(blocks)
    best_count = len(best_blocks)
    
    # Precompute t-subsets for quick coverage checking
    def get_tsets(block):
        return [frozenset(c) for c in itertools.combinations(block, t)]
    
    # Build coverage map: tset -> count of blocks covering it
    def build_coverage(block_list):
        cov = defaultdict(int)
        for b in block_list:
            for ts in get_tsets(b):
                cov[ts] += 1
        return cov
    
    for iteration in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        cov = build_coverage(best_blocks)
        
        # Try removing each block and see if we can find a replacement
        for i in range(len(best_blocks)):
            if deadline and time.time() > deadline:
                break
            
            candidate = best_blocks[i]
            candidate_tsets = get_tsets(candidate)
            
            # Check if removing this block leaves any t-set uncovered
            uncovered_by_removal = []
            for ts in candidate_tsets:
                if cov[ts] == 1:
                    uncovered_by_removal.append(ts)
            
            if not uncovered_by_removal:
                # Block is redundant, remove it
                best_blocks = [b for j, b in enumerate(best_blocks) if j != i]
                best_count = len(best_blocks)
                break  # Restart iteration since blocks changed
            
            # Try to find a replacement block that covers all uncovered t-sets
            # and possibly more
            best_replacement = None
            best_gain = -1
            
            # Sample some random blocks as candidates
            for _ in range(20):
                if deadline and time.time() > deadline:
                    break
                # Generate a random block containing at least one uncovered t-set
                seed_ts = random.choice(uncovered_by_removal)
                block = list(seed_ts)
                while len(block) < k:
                    x = random.randint(0, v - 1)
                    if x not in block:
                        block.append(x)
                block.sort()
                
                # Check if this block covers all uncovered t-sets
                block_tsets = set(get_tsets(block))
                covers_all = all(ts in block_tsets for ts in uncovered_by_removal)
                
                if covers_all:
                    # Count how many additional t-sets this block covers
                    additional = len(block_tsets - set(candidate_tsets))
                    if additional > best_gain:
                        best_gain = additional
                        best_replacement = block
            
            if best_replacement is not None:
                # Replace the block
                best_blocks[i] = best_replacement
                # Don't break, continue trying other blocks
        else:
            # No improvement found in this iteration
            break
    
    return best_blocks

def construct(v, k, t):
    deadline = time.time() + 115  # Safety margin
    
    # Precompute all t-subsets as frozensets for fast lookup
    all_tsets = [frozenset(c) for c in itertools.combinations(range(v), t)]
    all_tsets_set = set(all_tsets)
    
    # Precompute t-1 subsets for each possible t-subset for faster gain computation
    # For each t-subset, store the list of (t-1)-subsets
    tset_to_tminus1 = {}
    for ts in all_tsets:
        ts_list = list(ts)
        tminus1_list = [frozenset(c) for c in itertools.combinations(ts_list, t - 1)]
        tset_to_tminus1[ts] = tminus1_list
    
    best_blocks = None
    best_count = float('inf')
    
    # Multi-start greedy
    while time.time() < deadline:
        start_time = time.time()
        
        # 1. Greedy Construction
        uncovered = set(all_tsets)
        blocks = []
        
        while uncovered and time.time() < deadline - 10:
            # Pick a random uncovered t-set to seed a new block
            seed_tset = random.choice(list(uncovered))
            block = list(seed_tset)
            
            while len(block) < k and time.time() < deadline - 10:
                block_set = set(block)
                best_x = -1
                best_gain = -1
                best_candidates = []
                
                # For each candidate x not in block
                for x in range(v):
                    if x in block_set:
                        continue
                    # gain = number of (t-1)-subsets R of block such that {x}∪R is uncovered
                    gain = 0
                    for R in itertools.combinations(block, t - 1):
                        ts = frozenset(list(R) + [x])
                        if ts in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_candidates = [x]
                    elif gain == best_gain:
                        best_candidates.append(x)
                
                if best_x == -1 and not best_candidates:
                    if best_gain == 0:
                        remaining = [x for x in range(v) if x not in block_set]
                        if remaining:
                            best_x = random.choice(remaining)
                        else:
                            break
                    else:
                        break
                else:
                    best_x = random.choice(best_candidates)
                
                block.append(best_x)
            
            if len(block) == k:
                block.sort()
                block_frozen = frozenset(block)
                
                # Remove all covered t-sets
                covered = set()
                for c in itertools.combinations(block, t):
                    covered.add(frozenset(c))
                uncovered -= covered
                blocks.append(block)
        
        # Check if we have full coverage
        if uncovered:
            # If time ran out during construction, fill remaining
            for ts in list(uncovered):
                if time.time() >= deadline - 5:
                    break
                block = list(ts)
                while len(block) < k:
                    x = random.randint(0, v - 1)
                    if x not in block:
                        block.append(x)
                block.sort()
                if block not in blocks:
                    blocks.append(block)
                    for c in itertools.combinations(block, t):
                        uncovered.discard(frozenset(c))
        
        if uncovered:
            continue
            
        # 2. Local Search / Pruning
        pruned_blocks = _prune_redundant_blocks(blocks, v, t)
        
        prev_len = len(pruned_blocks)
        while len(pruned_blocks) < prev_len and time.time() < deadline - 5:
            prev_len = len(pruned_blocks)
            pruned_blocks = _prune_redundant_blocks(pruned_blocks, v, t)
        
        # 3. Local search refinement
        if time.time() < deadline - 15:
            refined_blocks = _local_search(pruned_blocks, v, k, t, max_iter=30, deadline=deadline - 5)
            if len(refined_blocks) < len(pruned_blocks):
                pruned_blocks = refined_blocks
        
        if len(pruned_blocks) < best_count:
            best_count = len(pruned_blocks)
            best_blocks = pruned_blocks
            
        if time.time() > deadline - 20:
            break

    if best_blocks is None:
        # Fallback: quick deterministic construction
        uncovered = set(all_tsets)
        blocks = []
        while uncovered and time.time() < deadline - 1:
            seed_tset = next(iter(uncovered))
            block = list(seed_tset)
            while len(block) < k:
                block_set = set(block)
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block_set: continue
                    gain = sum(1 for R in itertools.combinations(block, t-1) if frozenset(list(R)+[x]) in uncovered)
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x == -1:
                    rem = [x for x in range(v) if x not in block_set]
                    if rem: best_x = rem[0]
                    else: break
                block.append(best_x)
            if len(block) == k:
                block.sort()
                covered = set(frozenset(c) for c in itertools.combinations(block, t))
                uncovered -= covered
                blocks.append(block)
        
        if not uncovered:
            best_blocks = blocks
        else:
            for ts in list(uncovered):
                if time.time() >= deadline - 0.5: break
                block = list(ts)
                while len(block) < k:
                    x = random.randint(0, v-1)
                    if x not in block: block.append(x)
                block.sort()
                if block not in blocks:
                    blocks.append(block)
                    for c in itertools.combinations(block, t):
                        uncovered.discard(frozenset(c))
            best_blocks = blocks

    return best_blocks
