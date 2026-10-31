"""
Covering design C(19,7,4): minimize number of 7-subsets covering all 4-subsets.
Strategy: Multi-start greedy with randomized tie-breaking + aggressive local search
with removal, swap, and 2-swap moves using NumPy-vectorized coverage checks.
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
    
    # Build coverage matrix (N_k x N_t) as boolean numpy array
    # For v=19, k=7, t=4:
    # n_k = C(19,7) = 50388
    # n_t = C(19,4) = 3876
    # cov_matrix size: 50388 x 3876 -> ~195M elements. 
    # Using uint8, this is ~195 MB. This is acceptable.
    
    # However, we can be smarter. We only need to check coverage of current blocks.
    # Let's build a sparse representation or use the matrix if it fits.
    # 195 MB is fine for modern systems.
    
    cov_matrix = np.zeros((n_k, n_t), dtype=np.uint8)
    
    # For CSR: indptr and indices
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
    """Greedy set cover using NumPy vectorization with randomized tie-breaking."""
    uncovered = np.ones(n_t, dtype=np.uint8)
    blocks = []
    rng_local = random.Random(12345 + seed_offset)
    
    while True:
        if _elapsed() > _DEADLINE - 30:
            break
        
        # Calculate gain for each block: number of new t-subsets covered
        # gains[i] = sum(cov_matrix[i] * uncovered)
        gains = np.sum(cov_matrix * uncovered, axis=1)
        
        best_gain = gains.max()
        if best_gain == 0:
            break
            
        # Randomized tie-breaking among blocks with max gain
        best_indices = np.where(gains == best_gain)[0]
        best_i = best_indices[rng_local.randrange(len(best_indices))]
        
        blocks.append(best_i)
        uncovered[cov_matrix[best_i] == 1] = 0
        
        if np.sum(uncovered) == 0:
            break
            
    return blocks


def _local_search(blocks, cov_matrix, indptr, indices, n_t, n_k, rng, time_budget):
    """Local search using CSR for fast index retrieval and NumPy for counts."""
    blocks = list(blocks)
    
    # Initialize counts
    counts = np.zeros(n_t, dtype=np.int32)
    for b in blocks:
        start = indptr[b]
        end = indptr[b+1]
        if end > start:
            counts[indices[start:end]] += 1
            
    def is_valid(counts):
        return np.all(counts > 0)
        
    if not is_valid(counts):
        # Repair
        while not is_valid(counts) and _elapsed() < time_budget:
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
                
    improved = True
    stall_count = 0
    n_blocks = len(blocks)
    best_n_blocks = n_blocks
    best_blocks = list(blocks)
    
    while improved and _elapsed() < time_budget:
        improved = False
        stall_count += 1
        
        # --- Removal pass ---
        removable_found = False
        for i in range(n_blocks):
            b = blocks[i]
            start = indptr[b]
            end = indptr[b+1]
            if end == start: continue
            if counts[indices[start:end]].min() > 1:
                counts[indices[start:end]] -= 1
                blocks.pop(i)
                n_blocks -= 1
                improved = True
                stall_count = 0
                removable_found = True
                if n_blocks < best_n_blocks:
                    best_n_blocks = n_blocks
                    best_blocks = list(blocks)
                break
        
        if removable_found:
            continue
            
        # --- Single swap pass ---
        # Try to swap a block with another block that covers more uncovered or enables removal
        swaps_tried = 0
        max_swaps = 200
        while swaps_tried < max_swaps and _elapsed() < time_budget:
            swaps_tried += 1
            if n_blocks == 0:
                break
                
            i = rng.randrange(n_blocks)
            j = rng.randrange(n_k)
            if j in blocks:
                continue
                
            old = blocks[i]
            
            # Apply swap
            blocks[i] = j
            
            start_old = indptr[old]
            end_old = indptr[old+1]
            if end_old > start_old:
                counts[indices[start_old:end_old]] -= 1
            start_new = indptr[j]
            end_new = indptr[j+1]
            if end_new > start_new:
                counts[indices[start_new:end_new]] += 1
                
            if not is_valid(counts):
                # Revert
                counts[indices[start_new:end_new]] -= 1
                if end_old > start_old:
                    counts[indices[start_old:end_old]] += 1
                blocks[i] = old
                continue
                
            # Try removal after swap
            start_j = indptr[j]
            end_j = indptr[j+1]
            if end_j > start_j and counts[indices[start_j:end_j]].min() > 1:
                counts[indices[start_j:end_j]] -= 1
                blocks.pop(i)
                n_blocks -= 1
                improved = True
                stall_count = 0
                if n_blocks < best_n_blocks:
                    best_n_blocks = n_blocks
                    best_blocks = list(blocks)
                continue
                
            # Check if removing old block's coverage makes something else removable
            # Check a few random blocks
            for _check in range(5):
                ci = rng.randrange(n_blocks)
                cb = blocks[ci]
                cs = indptr[cb]
                ce = indptr[cb+1]
                if ce > cs and counts[indices[cs:ce]].min() > 1:
                    counts[indices[cs:ce]] -= 1
                    blocks.pop(ci)
                    n_blocks -= 1
                    improved = True
                    stall_count = 0
                    if n_blocks < best_n_blocks:
                        best_n_blocks = n_blocks
                        best_blocks = list(blocks)
                    break
                    
            if not improved:
                # Revert swap
                counts[indices[start_new:end_new]] -= 1
                if end_old > start_old:
                    counts[indices[start_old:end_old]] += 1
                blocks[i] = old

        # --- Perturbation if stalled ---
        if not improved and stall_count > 3 and n_blocks > 10:
            if _elapsed() < time_budget - 10:
                i = rng.randrange(n_blocks)
                removed_block = blocks.pop(i)
                start = indptr[removed_block]
                end = indptr[removed_block+1]
                if end > start:
                    counts[indices[start:end]] -= 1
                n_blocks -= 1
                while not is_valid(counts) and _elapsed() < time_budget - 5:
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
    
    n_starts = 5
    for s in range(n_starts):
        if _elapsed() > _DEADLINE - 40:
            break
        blocks = _greedy_construct(v, k, t, rng, cov_matrix, n_t, n_k, seed_offset=s)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
    
    # Local search on best
    time_budget = _DEADLINE - 8
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
