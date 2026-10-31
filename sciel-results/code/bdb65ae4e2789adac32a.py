"""
Covering design C(19,7,4): minimize number of 7-subsets covering all 4-subsets.
Strategy: Fast Multi-start Greedy (using matrix-vector product) + Aggressive Local Search
with removal and guided/randomized swaps.
"""
import time
import itertools
import random
import numpy as np

_DEADLINE = 115.0  # seconds, safety margin under 120s
_START = None


def _elapsed():
    return time.time() - _START


def _precompute(v, k, t):
    """Precompute all t-subsets, k-subsets, and coverage structures."""
    all_t = list(itertools.combinations(range(v), t))
    n_t = len(all_t)
    t_index = {c: i for i, c in enumerate(all_t)}
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    
    # Build coverage matrix (N_k x N_t) as uint8 numpy array
    # n_k = 50388, n_t = 3876
    # Size ~ 195 MB. Acceptable.
    cov_matrix = np.zeros((n_k, n_t), dtype=np.uint8)
    
    # Build CSR structure for fast row access in local search
    sizes = np.zeros(n_k, dtype=np.int32)
    all_indices = []
    
    for i, ks in enumerate(k_subsets):
        ts_list = []
        for ts in itertools.combinations(ks, t):
            idx = t_index[ts]
            cov_matrix[i, idx] = 1
            ts_list.append(idx)
        sizes[i] = len(ts_list)
        all_indices.append(ts_list)
        
    indptr = np.zeros(n_k + 1, dtype=np.int32)
    np.cumsum(sizes, out=indptr[1:])
    indices = np.concatenate(all_indices) if all_indices else np.array([], dtype=np.int32)
    
    return all_t, t_index, k_subsets, cov_matrix, indptr, indices, n_t, n_k


def _greedy_construct(v, k, t, rng, cov_matrix, n_t, n_k, seed_offset=0):
    """Greedy set cover using NumPy matrix-vector product for gains."""
    uncovered = np.ones(n_t, dtype=np.uint8)
    uncovered_count = n_t
    blocks = []
    rng_local = random.Random(12345 + seed_offset)
    
    while uncovered_count > 0:
        if _elapsed() > _DEADLINE - 20:
            break
        
        # Calculate gain: number of new t-subsets covered
        # cov_matrix is uint8, uncovered is uint8.
        # Dot product is fast and avoids creating a full intermediate matrix.
        gains = cov_matrix @ uncovered
        
        best_gain = gains.max()
        if best_gain == 0:
            break
            
        # Randomized tie-breaking among blocks with max gain
        best_indices = np.where(gains == best_gain)[0]
        best_i = best_indices[rng_local.randrange(len(best_indices))]
        
        blocks.append(best_i)
        
        # Update uncovered
        # Only update where cov_matrix[best_i] is 1
        mask = cov_matrix[best_i] == 1
        uncovered[mask] = 0
        uncovered_count -= best_gain
        
    return blocks


def _local_search(blocks, cov_matrix, indptr, indices, n_t, n_k, rng, time_budget):
    """
    Local search using CSR for fast index retrieval and NumPy for counts.
    Moves:
    1. Removal: Remove any block that doesn't cover a unique t-subset.
    2. Swap: Replace a block with another to enable removals.
    """
    blocks = list(blocks)
    n_blocks = len(blocks)
    
    # Initialize counts
    counts = np.zeros(n_t, dtype=np.int32)
    for b in blocks:
        start = indptr[b]
        end = indptr[b+1]
        if end > start:
            counts[indices[start:end]] += 1
            
    def is_valid():
        return np.all(counts > 0)
        
    if not is_valid():
        # Repair
        while not is_valid() and _elapsed() < time_budget:
            missing = np.where(counts == 0)[0]
            if len(missing) == 0: break
            gains = np.sum(cov_matrix[:, missing], axis=1)
            best_i = np.argmax(gains)
            if gains[best_i] == 0: break
            blocks.append(best_i)
            start = indptr[best_i]
            end = indptr[best_i+1]
            if end > start:
                counts[indices[start:end]] += 1
            n_blocks += 1

    best_n_blocks = n_blocks
    best_blocks = list(blocks)
    
    improved = True
    stall_count = 0
    
    while improved and _elapsed() < time_budget:
        improved = False
        stall_count += 1
        
        # --- Removal Pass ---
        # Iterate until no more blocks can be removed
        removal_found = True
        while removal_found and _elapsed() < time_budget:
            removal_found = False
            for i in range(n_blocks - 1, -1, -1):
                b = blocks[i]
                start = indptr[b]
                end = indptr[b+1]
                if end == start: 
                    blocks.pop(i)
                    n_blocks -= 1
                    improved = True
                    removal_found = True
                    if n_blocks < best_n_blocks:
                        best_n_blocks = n_blocks
                        best_blocks = list(blocks)
                    continue
                # Check if any t-subset in this block has count 1
                if counts[indices[start:end]].min() > 1:
                    counts[indices[start:end]] -= 1
                    blocks.pop(i)
                    n_blocks -= 1
                    improved = True
                    removal_found = True
                    if n_blocks < best_n_blocks:
                        best_n_blocks = n_blocks
                        best_blocks = list(blocks)
        
        if _elapsed() > time_budget:
            break

        # --- Swap Pass ---
        # Try random swaps
        swaps_tried = 0
        max_swaps = 1000 # Increased attempts
        while swaps_tried < max_swaps and _elapsed() < time_budget:
            swaps_tried += 1
            if n_blocks == 0:
                break
                
            i = rng.randrange(n_blocks)
            # Pick a random block not currently in the set
            # To avoid O(N) check, we can just pick random and check set membership
            # But n_k is 50k, n_blocks is ~140. 
            # We can maintain a set of current block indices for O(1) lookup.
            # However, maintaining the set adds overhead. 
            # Let's just pick random j and check if j is in blocks. 
            # To make it faster, we can keep a set.
            
            # Optimization: Keep a set of current blocks
            # (We don't have it in scope, so we'll just check list membership 
            # which is O(N). N=140 is small. 1000 swaps * 140 checks = 140k ops. Fine.)
            
            j = rng.randrange(n_k)
            if j in blocks:
                continue
                
            old = blocks[i]
            
            # Apply swap
            blocks[i] = j
            
            # Update counts
            start_old = indptr[old]
            end_old = indptr[old+1]
            if end_old > start_old:
                counts[indices[start_old:end_old]] -= 1
            start_new = indptr[j]
            end_new = indptr[j+1]
            if end_new > start_new:
                counts[indices[start_new:end_new]] += 1
                
            if not is_valid():
                # Revert
                if end_new > start_new:
                    counts[indices[start_new:end_new]] -= 1
                if end_old > start_old:
                    counts[indices[start_old:end_old]] += 1
                blocks[i] = old
                continue
                
            # If valid, try to remove redundant blocks
            # We know the state is valid. Now check if we can remove anything.
            # We can check the new block j first, then others.
            # Actually, just run a limited removal pass.
            
            removals = 0
            # Check if new block j is redundant
            if end_new > start_new and counts[indices[start_new:end_new]].min() > 1:
                counts[indices[start_new:end_new]] -= 1
                blocks.pop(i)
                n_blocks -= 1
                removals += 1
                improved = True
            else:
                # Check a few random other blocks
                for _check in range(10):
                    ci = rng.randrange(n_blocks)
                    cb = blocks[ci]
                    cs = indptr[cb]
                    ce = indptr[cb+1]
                    if ce > cs and counts[indices[cs:ce]].min() > 1:
                        counts[indices[cs:ce]] -= 1
                        blocks.pop(ci)
                        n_blocks -= 1
                        removals += 1
                        improved = True
                        break
                        
            if removals == 0:
                # Revert swap
                if end_new > start_new:
                    counts[indices[start_new:end_new]] -= 1
                if end_old > start_old:
                    counts[indices[start_old:end_old]] += 1
                blocks[i] = old
            else:
                if n_blocks < best_n_blocks:
                    best_n_blocks = n_blocks
                    best_blocks = list(blocks)
                stall_count = 0

        # --- Perturbation if stalled ---
        if not improved and stall_count > 5 and n_blocks > 10:
            if _elapsed() < time_budget - 10:
                # Remove a random block and repair
                i = rng.randrange(n_blocks)
                removed_block = blocks.pop(i)
                start = indptr[removed_block]
                end = indptr[removed_block+1]
                if end > start:
                    counts[indices[start:end]] -= 1
                n_blocks -= 1
                
                # Repair
                while not is_valid() and _elapsed() < time_budget - 5:
                    missing = np.where(counts == 0)[0]
                    if len(missing) == 0: break
                    gains = np.sum(cov_matrix[:, missing], axis=1)
                    best_i = np.argmax(gains)
                    if gains[best_i] == 0: break
                    blocks.append(best_i)
                    start = indptr[best_i]
                    end = indptr[best_i+1]
                    if end > start:
                        counts[indices[start:end]] += 1
                    n_blocks += 1
                stall_count = 0

    # Return the best found
    if len(best_blocks) < n_blocks:
        return best_blocks
    return blocks


def construct(v, k, t):
    global _START
    _START = time.time()
    rng = random.Random(12345)

    all_t, t_index, k_subsets, cov_matrix, indptr, indices, n_t, n_k = _precompute(v, k, t)

    # Multi-start greedy
    best_blocks = None
    best_count = float('inf')
    
    # We have more time now due to faster greedy.
    # Allocate 20s for greedy phase, rest for local search.
    greedy_deadline = _DEADLINE - 80 # Leave 80s for local search
    
    n_starts = 0
    while _elapsed() < greedy_deadline:
        n_starts += 1
        blocks = _greedy_construct(v, k, t, rng, cov_matrix, n_t, n_k, seed_offset=n_starts)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
        # If we find a very good one (e.g., < 140), we can stop early?
        # No, let's keep searching to see if we can beat it.
        
    # Local search on best
    time_budget = _DEADLINE - 5
    blocks = _local_search(best_blocks, cov_matrix, indptr, indices, n_t, n_k, rng, time_budget)

    # Safety: ensure valid; if not, do a quick repair
    counts = np.zeros(n_t, dtype=np.int32)
    for b in blocks:
        start = indptr[b]
        end = indptr[b+1]
        if end > start:
            counts[indices[start:end]] += 1
            
    if not np.all(counts > 0):
        while not np.all(counts > 0) and _elapsed() < _DEADLINE - 2:
            missing = np.where(counts == 0)[0]
            if len(missing) == 0: break
            gains = np.sum(cov_matrix[:, missing], axis=1)
            best_i = np.argmax(gains)
            if gains[best_i] == 0: break
            blocks.append(best_i)
            start = indptr[best_i]
            end = indptr[best_i+1]
            if end > start:
                counts[indices[start:end]] += 1

    # Return as list of k-subsets (lists of ints)
    result = [list(k_subsets[b]) for b in blocks]
    seen = set()
    dedup = []
    for blk in result:
        key = tuple(sorted(blk))
        if key not in seen:
            seen.add(key)
            dedup.append(list(key))
    return dedup
