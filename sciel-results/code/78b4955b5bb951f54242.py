import time
import random
import itertools
import numpy as np

# Global cache for precomputed data
_CACHE = {}

def _prepare(v, k, t):
    key = (v, k, t)
    if key in _CACHE:
        return _CACHE[key]
    
    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    all_subsets = list(itertools.combinations(points, t))
    
    n_blocks = len(all_blocks)
    n_subsets = len(all_subsets)
    
    # Map subset tuple to index
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    # For each block, list of subset indices it covers
    block_subsets = []
    for b in all_blocks:
        subs = []
        for s in itertools.combinations(b, t):
            subs.append(subset_to_idx[s])
        block_subsets.append(subs)
        
    # For each subset, list of block indices that cover it
    # This is crucial for fast swap search
    blocks_covering_subset = [[] for _ in range(n_subsets)]
    for i, subs in enumerate(block_subsets):
        for s_idx in subs:
            blocks_covering_subset[s_idx].append(i)
            
    # Convert to numpy arrays for greedy phase if needed, 
    # but for local search, lists of lists are fine.
    # For greedy, we can use a sparse matrix representation or just iterate.
    # Let's keep it simple with lists for local search.
    
    _CACHE[key] = (all_blocks, block_subsets, blocks_covering_subset, n_blocks, n_subsets)
    return _CACHE[key]

def _greedy_construct(v, k, t, deadline, rng):
    all_blocks, block_subsets, blocks_covering_subset, n_blocks, n_subsets = _prepare(v, k, t)
    
    uncovered = [True] * n_subsets
    counts = [0] * n_subsets
    selected = []
    selected_set = set()
    
    # Precompute a "weight" for each subset? No, just track uncovered.
    
    while any(uncovered) and time.time() < deadline:
        best_block = -1
        best_score = -1
        
        # Iterate through all blocks to find the one covering most uncovered subsets
        # This is O(N_blocks * avg_subsets_per_block)
        # 50k * 35 ~ 1.7M ops per step. If we need 100 steps, 170M ops.
        # Python might be too slow for 100 full scans.
        # Optimization: Maintain a list of "active" blocks? 
        # Or use NumPy?
        
        # Let's try to speed up the scan.
        # We can precompute the list of subsets for each block.
        # We need to count how many are uncovered.
        
        # If this is too slow, we can randomize or use a heuristic.
        # For now, let's assume it's fast enough for ~50-100 iterations.
        
        for i in range(n_blocks):
            if i in selected_set:
                continue
            subs = block_subsets[i]
            score = 0
            for s_idx in subs:
                if uncovered[s_idx]:
                    score += 1
            if score > best_score:
                best_score = score
                best_block = i
                
        if best_block == -1:
            break
            
        selected.append(best_block)
        selected_set.add(best_block)
        
        # Update uncovered and counts
        for s_idx in block_subsets[best_block]:
            counts[s_idx] += 1
            if counts[s_idx] == 1:
                uncovered[s_idx] = False
                
    return selected, counts, selected_set

def _local_search(selected, counts, selected_set, block_subsets, blocks_covering_subset, n_subsets, deadline, rng):
    # selected: list of block indices
    # counts: list of int
    # selected_set: set of block indices
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Remove redundant blocks
        # A block is redundant if it doesn't cover any subset with count == 1
        i = 0
        while i < len(selected):
            b = selected[i]
            is_redundant = True
            for s_idx in block_subsets[b]:
                if counts[s_idx] == 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                # Remove block b
                for s_idx in block_subsets[b]:
                    counts[s_idx] -= 1
                selected.pop(i)
                selected_set.remove(b)
                improved = True
                # Don't increment i, check the same position again (next block shifted)
            else:
                i += 1
                
        if time.time() > deadline:
            break
            
        # Phase 2: Swap
        # Find a block to swap out and a block to swap in
        # Heuristic: Pick the block with the most "critical" subsets (count == 1)
        # Actually, if we just removed all redundant blocks, no block is redundant.
        # So every block has at least one critical subset.
        
        # We want to swap A with B such that B covers all critical subsets of A.
        # This maintains validity.
        
        # To find a good swap, we iterate through selected blocks.
        # To avoid O(N^2) behavior, we can pick one at random or the "worst" one.
        # Let's pick the one with the most critical subsets, as it's the most "constrained".
        
        best_swap_out = -1
        best_crit_count = 0
        best_crit_subs = []
        
        for b in selected:
            crit_subs = [s_idx for s_idx in block_subsets[b] if counts[s_idx] == 1]
            if len(crit_subs) > best_crit_count:
                best_crit_count = len(crit_subs)
                best_swap_out = b
                best_crit_subs = crit_subs
                
        if best_swap_out == -1:
            break
            
        # Find candidates for B
        # B must cover all s in best_crit_subs
        if not best_crit_subs:
            # Should not happen if we removed redundant blocks
            continue
            
        # Intersect lists of blocks covering each critical subset
        # Start with the subset that has the fewest covering blocks
        # to minimize the intersection size
        
        # Sort critical subsets by the size of their covering block list
        best_crit_subs_sorted = sorted(best_crit_subs, key=lambda s: len(blocks_covering_subset[s]))
        
        candidates = set(blocks_covering_subset[best_crit_subs_sorted[0]])
        for s_idx in best_crit_subs_sorted[1:]:
            candidates &= set(blocks_covering_subset[s_idx])
            
        # Remove currently selected blocks
        candidates -= selected_set
        
        if not candidates:
            # No valid swap for this A. Try another A? 
            # For simplicity, let's just break or try random swaps later.
            # A better strategy: try all A's if the first fails? 
            # That's expensive. Let's just break for now, or pick a random A.
            # Let's try a few random A's if the best one fails.
            found_candidate = False
            for _ in range(5):
                b = rng.choice(selected)
                crit_subs = [s_idx for s_idx in block_subsets[b] if counts[s_idx] == 1]
                if not crit_subs:
                    continue
                crit_subs_sorted = sorted(crit_subs, key=lambda s: len(blocks_covering_subset[s]))
                cands = set(blocks_covering_subset[crit_subs_sorted[0]])
                for s_idx in crit_subs_sorted[1:]:
                    cands &= set(blocks_covering_subset[s_idx])
                cands -= selected_set
                if cands:
                    best_swap_out = b
                    candidates = cands
                    found_candidate = True
                    break
            
            if not found_candidate:
                break
                
        # Pick the best B from candidates
        # Heuristic: Maximize coverage of subsets with count 1 or 2
        best_b = -1
        best_b_score = -1
        
        for b in candidates:
            score = 0
            for s_idx in block_subsets[b]:
                c = counts[s_idx]
                if c == 1:
                    score += 10
                elif c == 2:
                    score += 5
                elif c == 0:
                    score += 2
            if score > best_b_score:
                best_b_score = score
                best_b = b
                
        if best_b == -1:
            break
            
        # Perform swap
        # Remove best_swap_out
        for s_idx in block_subsets[best_swap_out]:
            counts[s_idx] -= 1
        selected_set.remove(best_swap_out)
        selected.remove(best_swap_out)
        
        # Add best_b
        for s_idx in block_subsets[best_b]:
            counts[s_idx] += 1
        selected_set.add(best_b)
        selected.append(best_b)
        
        improved = True
        
    return selected, counts, selected_set

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 110.0  # Safety margin
    
    all_blocks, block_subsets, blocks_covering_subset, n_blocks, n_subsets = _prepare(v, k, t)
    
    rng = random.Random(12345)
    
    # Phase 1: Greedy Construction
    # We use 50% of the time for construction
    cons_deadline = t0 + (deadline - t0) * 0.5
    
    selected, counts, selected_set = _greedy_construct(v, k, t, cons_deadline, rng)
    
    # Phase 2: Local Search
    ls_deadline = deadline - 5.0
    selected, counts, selected_set = _local_search(
        selected, counts, selected_set, 
        block_subsets, blocks_covering_subset, 
        n_subsets, ls_deadline, rng
    )
    
    # Phase 3: Second Local Search Pass (if time remains)
    if time.time() < deadline - 10.0:
        selected, counts, selected_set = _local_search(
            selected, counts, selected_set, 
            block_subsets, blocks_covering_subset, 
            n_subsets, deadline - 5.0, rng
        )
    
    # Convert to list of lists of points
    result = [list(all_blocks[i]) for i in selected]
    
    return result
