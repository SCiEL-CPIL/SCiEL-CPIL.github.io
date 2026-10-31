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
    blocks_covering_subset = [[] for _ in range(n_subsets)]
    for i, subs in enumerate(block_subsets):
        for s_idx in subs:
            blocks_covering_subset[s_idx].append(i)
            
    # Convert blocks_covering_subset to numpy arrays for faster intersection?
    # Actually, lists are fine for intersection if we use sets, but numpy is faster for boolean ops.
    # Let's keep lists for now, as intersection of small lists is fast.
    
    _CACHE[key] = (all_blocks, block_subsets, blocks_covering_subset, n_blocks, n_subsets)
    return _CACHE[key]

def _greedy_construct(v, k, t, deadline, rng):
    all_blocks, block_subsets, blocks_covering_subset, n_blocks, n_subsets = _prepare(v, k, t)
    
    # NumPy arrays for speed
    # gains[i] = number of uncovered subsets covered by block i
    gains = np.full(n_blocks, len(block_subsets[0]), dtype=np.int32)
    uncovered = np.ones(n_subsets, dtype=bool)
    
    selected = []
    selected_set = set()
    
    while np.any(uncovered) and time.time() < deadline:
        # Find block with max gain
        b = np.argmax(gains)
        if gains[b] == 0:
            break
        
        selected.append(b)
        selected_set.add(b)
        
        # Update gains for all blocks that share any subset with b
        # We only need to update blocks that cover any of the subsets covered by b
        # and were previously uncovered? No, gains is "potential new coverage".
        # If a subset is already covered, its contribution to gain is 0.
        # So we decrement gains for all blocks covering the subsets that b covers,
        # but only if those subsets were uncovered before this step.
        
        # Optimization: 
        # For each subset s in block_subsets[b]:
        #   if uncovered[s]:
        #     uncovered[s] = False
        #     for each block j covering s:
        #        gains[j] -= 1
        
        subs_b = block_subsets[b]
        for s_idx in subs_b:
            if uncovered[s_idx]:
                uncovered[s_idx] = False
                # Update gains for all blocks covering s_idx
                # blocks_covering_subset[s_idx] is a list of block indices
                # We can vectorize this if we convert to array, but list iteration might be faster for small lists
                for j in blocks_covering_subset[s_idx]:
                    gains[j] -= 1
                    
    return selected, selected_set

def _local_search(selected, selected_set, block_subsets, blocks_covering_subset, n_subsets, deadline, rng):
    # Maintain cover_count
    cover_count = np.zeros(n_subsets, dtype=np.int32)
    for b in selected:
        for s_idx in block_subsets[b]:
            cover_count[s_idx] += 1
            
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
                if cover_count[s_idx] == 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                # Remove block b
                for s_idx in block_subsets[b]:
                    cover_count[s_idx] -= 1
                selected.pop(i)
                selected_set.remove(b)
                improved = True
            else:
                i += 1
                
        if time.time() > deadline:
            break
            
        # Phase 2: Swap
        # Find a block to swap out (one with fewest unique subsets)
        # and a block to swap in (covering all unique subsets of the out block)
        
        # Calculate unique counts
        unique_counts = {}
        unique_subs_map = {}
        
        for b in selected:
            u_count = 0
            u_list = []
            for s_idx in block_subsets[b]:
                if cover_count[s_idx] == 1:
                    u_count += 1
                    u_list.append(s_idx)
            unique_counts[b] = u_count
            unique_subs_map[b] = u_list
            
        # Find block with minimum unique count (>0)
        min_unique = min(unique_counts.values())
        
        # Candidates for removal are those with min_unique
        candidates_out = [b for b, c in unique_counts.items() if c == min_unique]
        
        if not candidates_out or min_unique == 0:
            # If min_unique is 0, it means a block has no unique subsets? 
            # That would mean it's redundant, but we just removed redundancies.
            # So min_unique should be > 0.
            # If no candidates, break.
            break
            
        # Shuffle to break symmetry
        rng.shuffle(candidates_out)
        
        swap_made = False
        for b_out in candidates_out:
            crit_subs = unique_subs_map[b_out]
            
            if not crit_subs:
                continue
                
            # Find candidates for B_in
            # Intersect lists of blocks covering each critical subset
            # Sort by list size to minimize intersection cost
            crit_subs_sorted = sorted(crit_subs, key=lambda s: len(blocks_covering_subset[s]))
            
            # Start with the first subset's covering blocks
            # Convert to set for intersection
            candidates_in = set(blocks_covering_subset[crit_subs_sorted[0]])
            
            for s_idx in crit_subs_sorted[1:]:
                candidates_in &= set(blocks_covering_subset[s_idx])
                
            # Remove currently selected blocks
            candidates_in -= selected_set
            
            if not candidates_in:
                continue
                
            # Pick a candidate
            # Heuristic: Maximize coverage of subsets with count 2?
            # Or just random to avoid loops?
            # Let's pick the one that covers the most "tight" subsets (count 2)
            best_b_in = -1
            best_score = -1
            
            for b_in in candidates_in:
                score = 0
                for s_idx in block_subsets[b_in]:
                    c = cover_count[s_idx]
                    if c == 2:
                        score += 10
                    elif c == 3:
                        score += 5
                    elif c == 1:
                        score += 1 # Should be covered by crit_subs, so these are the ones we're swapping for
                        # Actually, b_in covers crit_subs, so these counts will go up.
                        # We want b_in to cover other low-count subsets too.
                if score > best_score:
                    best_score = score
                    best_b_in = b_in
                    
            if best_b_in == -1:
                continue
                
            # Perform swap
            # Remove b_out
            for s_idx in block_subsets[b_out]:
                cover_count[s_idx] -= 1
            selected_set.remove(b_out)
            selected.remove(b_out)
            
            # Add b_in
            for s_idx in block_subsets[best_b_in]:
                cover_count[s_idx] += 1
            selected_set.add(best_b_in)
            selected.append(best_b_in)
            
            swap_made = True
            improved = True
            break # Restart loop after one swap
            
        if not swap_made:
            # No swap possible with min_unique blocks.
            # Try next min_unique?
            # Or break?
            # To be safe, break.
            break
            
    return selected, selected_set

def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin
    
    all_blocks, block_subsets, blocks_covering_subset, n_blocks, n_subsets = _prepare(v, k, t)
    
    rng = random.Random(12345)
    
    # Phase 1: Greedy Construction
    # Use 20% of time for greedy, as it's fast with NumPy
    cons_deadline = t0 + (deadline - t0) * 0.2
    
    selected, selected_set = _greedy_construct(v, k, t, cons_deadline, rng)
    
    # Phase 2: Local Search
    # Use remaining time
    selected, selected_set = _local_search(
        selected, selected_set,
        block_subsets, blocks_covering_subset,
        n_subsets, deadline, rng
    )
    
    # Convert to list of lists of points
    result = [list(all_blocks[i]) for i in selected]
    
    return result
