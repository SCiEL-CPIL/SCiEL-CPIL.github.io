import time
import random
import itertools
import numpy as np

# Cache for precomputed data
_CACHE = {}

def _prepare(v, k, t):
    key = (v, k, t)
    if key in _CACHE:
        return _CACHE[key]
    
    points = range(v)
    all_kblocks = list(itertools.combinations(points, k))
    all_tsubs = list(itertools.combinations(points, t))
    
    n_blocks = len(all_kblocks)
    n_tsubs = len(all_tsubs)
    
    # Map t-subsets to indices
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Build coverage matrix: Cover[i, j] = 1 if tsub j is in block i
    # Using uint8 for memory efficiency and speed
    Cover = np.zeros((n_blocks, n_tsubs), dtype=np.uint8)
    
    # Also build the list of t-subset indices for each block for fast access
    block_tsubs = []
    
    for i, B in enumerate(all_kblocks):
        B_set = set(B)
        indices = []
        for ts in itertools.combinations(B, t):
            j = tsub_to_idx[ts]
            Cover[i, j] = 1
            indices.append(j)
        block_tsubs.append(indices)
        
    # Build inverse mapping: for each t-subset, which blocks cover it?
    # This is crucial for fast gain updates
    tsub_to_blocks = [[] for _ in range(n_tsubs)]
    for i, indices in enumerate(block_tsubs):
        for j in indices:
            tsub_to_blocks[j].append(i)
    
    _CACHE[key] = (all_kblocks, all_tsubs, Cover, block_tsubs, tsub_to_blocks, n_blocks, n_tsubs)
    return _CACHE[key]

def _greedy_construct(Cover, block_tsubs, tsub_to_blocks, n_blocks, n_tsubs, deadline, rng):
    """
    Greedy construction using sparse gain updates.
    """
    uncovered = np.ones(n_tsubs, dtype=bool)
    # Initial gain for each block is the number of t-subsets it covers
    # Which is C(k, t)
    k_choose_t = len(block_tsubs[0])
    gains = np.full(n_blocks, k_choose_t, dtype=np.int32)
    
    selected_indices = []
    
    while np.any(uncovered):
        if time.time() > deadline:
            break
            
        # Find the block with the maximum gain
        best_idx = np.argmax(gains)
        
        if gains[best_idx] == 0:
            break
            
        selected_indices.append(best_idx)
        
        # Update gains for all blocks that share any t-subset with best_idx
        # We only need to update gains for blocks that cover t-subsets
        # that were previously uncovered.
        for s in block_tsubs[best_idx]:
            if uncovered[s]:
                uncovered[s] = False
                # Decrement gain for all blocks covering s
                # tsub_to_blocks[s] is a list of block indices
                for b in tsub_to_blocks[s]:
                    gains[b] -= 1
                    
    return selected_indices

def _local_search(selected_indices, Cover, block_tsubs, tsub_to_blocks, n_blocks, n_tsubs, deadline, rng):
    """
    Local search: Remove redundant blocks and perform swaps.
    """
    # Convert to list for mutability
    sel = list(selected_indices)
    sel_set = set(sel)
    
    # Compute cover counts
    cover_count = np.zeros(n_tsubs, dtype=np.int32)
    for i in sel:
        for s in block_tsubs[i]:
            cover_count[s] += 1
            
    improved = True
    rounds = 0
    max_rounds = 10000
    
    while improved and rounds < max_rounds and time.time() < deadline:
        improved = False
        rounds += 1
        
        # 1) Try dropping redundant blocks
        drop_idx = -1
        for pos, i in enumerate(sel):
            # Check if any t-subset covered by i has count == 1
            # If no such t-subset exists, i is redundant
            is_redundant = True
            for s in block_tsubs[i]:
                if cover_count[s] == 1:
                    is_redundant = False
                    break
            if is_redundant:
                drop_idx = pos
                break
        
        if drop_idx != -1:
            i = sel[drop_idx]
            for s in block_tsubs[i]:
                cover_count[s] -= 1
            sel.pop(drop_idx)
            sel_set.remove(i)
            improved = True
            continue
            
        # 2) Try swapping
        # For each block i in sel, find its critical t-subsets (count == 1)
        # If there is a block j not in sel that covers all critical t-subsets of i, swap i with j
        swap_done = False
        for pos, i in enumerate(sel):
            crit_subs = [s for s in block_tsubs[i] if cover_count[s] == 1]
            
            if not crit_subs:
                # Should have been caught by redundancy check, but safe to skip
                continue
                
            # To find a candidate j, it must cover crit_subs[0]
            # Get all blocks covering crit_subs[0]
            candidates = tsub_to_blocks[crit_subs[0]]
            
            # Filter candidates: must not be in sel_set
            # And must cover all other crit_subs
            
            found_swap = False
            for j in candidates:
                if j in sel_set:
                    continue
                    
                # Check if j covers all crit_subs
                # We can use a simple loop since crit_subs is small
                covers_all = True
                for s in crit_subs[1:]:
                    if not Cover[j, s]:
                        covers_all = False
                        break
                            
                if covers_all:
                    # Perform swap: remove i, add j
                    # Update cover_count for i
                    for s in block_tsubs[i]:
                        cover_count[s] -= 1
                    # Update cover_count for j
                    for s in block_tsubs[j]:
                        cover_count[s] += 1
                        
                    sel[pos] = j
                    sel_set.remove(i)
                    sel_set.add(j)
                    improved = True
                    found_swap = True
                    break
                    
            if found_swap:
                swap_done = True
                break
        # End swap loop
        
    return sel

def construct(v, k, t):
    t0 = time.time()
    # Safety margin: 120s limit, aim to finish by 115s
    deadline = t0 + 115.0
    
    # Allocate 80% of time to greedy, 20% to local search
    # Greedy is usually fast, so give it a tight deadline first
    # Then use remaining time for local search
    greedy_deadline = t0 + 30.0  # Greedy should take < 5s usually
    ls_deadline = deadline
    
    data = _prepare(v, k, t)
    all_kblocks, all_tsubs, Cover, block_tsubs, tsub_to_blocks, n_blocks, n_tsubs = data
    
    # Phase 1: Greedy Construction
    selected_indices = _greedy_construct(Cover, block_tsubs, tsub_to_blocks, n_blocks, n_tsubs, greedy_deadline, random.Random(123))
    
    # Phase 2: Local Search
    selected_indices = _local_search(selected_indices, Cover, block_tsubs, tsub_to_blocks, n_blocks, n_tsubs, ls_deadline, random.Random(456))
    
    # Final validity check and repair
    # Check for uncovered t-subsets
    covered_mask = np.zeros(n_tsubs, dtype=bool)
    for i in selected_indices:
        covered_mask[Cover[i] == 1] = True
        
    if not np.all(covered_mask):
        # If invalid, greedily add blocks to cover remaining
        uncovered = ~covered_mask
        if np.any(uncovered):
            # Re-initialize gains for repair
            gains = np.zeros(n_blocks, dtype=np.int32)
            # Only consider blocks not already selected
            # Actually, we can just run a mini-greedy on the uncovered part
            # But we must ensure we don't pick duplicates if possible, or it doesn't matter for validity
            # The problem says blocks must be distinct.
            sel_set = set(selected_indices)
            
            # Compute gains only for uncovered subsets
            # gains[i] = sum(Cover[i, j] for j in uncovered)
            # This is Cover[:, uncovered].sum(axis=1)
            gains = Cover[:, uncovered].sum(axis=1)
            
            while np.any(uncovered) and time.time() < deadline - 1.0:
                best_idx = np.argmax(gains)
                if gains[best_idx] == 0:
                    break
                if best_idx in sel_set:
                    # If the best block is already selected, it means no new coverage?
                    # No, gains is calculated on uncovered. If best_idx is in sel_set, 
                    # it means it covers some uncovered subsets? 
                    # Wait, if it's in sel_set, it was added before. 
                    # Why is it covering uncovered subsets? 
                    # Because covered_mask was updated. 
                    # So if best_idx is in sel_set, it should have covered those subsets already.
                    # This implies a bug in logic or state.
                    # Let's just skip it and set gain to 0 to avoid infinite loop.
                    gains[best_idx] = 0
                    continue
                    
                selected_indices.append(best_idx)
                sel_set.add(best_idx)
                
                # Update uncovered
                new_covered = Cover[best_idx] & uncovered
                uncovered[Cover[best_idx] == 1] = False
                
                # Update gains for other blocks
                # For each newly covered t-subset s, decrement gain of all blocks covering s
                # We can do this efficiently
                newly_covered_indices = np.where(new_covered)[0]
                for s in newly_covered_indices:
                    for b in tsub_to_blocks[s]:
                        if b in sel_set:
                            # We don't care about gains of selected blocks for future selection
                            # unless we allow re-selection? No, blocks must be distinct.
                            # So we can ignore selected blocks in gain update for repair.
                            pass
                        else:
                            gains[b] -= 1

    # Convert indices to blocks
    result = [list(all_kblocks[i]) for i in selected_indices]
    return result
