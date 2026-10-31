"""Covering design C(19,7,4): minimize number of 7-subsets covering all 4-subsets.
Strategy: NumPy-vectorized greedy construction + CSR-based local search with removal,
swap, and 2-swap moves.
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
    
    # Build coverage matrix (N_k x N_t) as boolean numpy array for greedy
    # And CSR structure for local search
    cov_matrix = np.zeros((n_k, n_t), dtype=np.uint8)
    
    # For CSR: indptr and indices
    # First pass: count sizes
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
    
    # Build inverted index: for each t-subset, list of blocks containing it
    # inv_index[t_idx] = list of block indices
    inv_index = [list() for _ in range(n_t)]
    for i in range(n_k):
        start = indptr[i]
        end = indptr[i+1]
        for j in range(start, end):
            inv_index[indices[j]].append(i)
    
    return all_t, t_index, k_subsets, cov_matrix, indptr, indices, n_t, n_k, inv_index


def _greedy_construct(v, k, t, rng, cov_matrix, n_t, n_k):
    """Greedy set cover using NumPy vectorization."""
    uncovered = np.ones(n_t, dtype=np.uint8)
    blocks = []
    
    while True:
        if _elapsed() > _DEADLINE - 25:
            break
        
        # Calculate gain for each block
        # cov_matrix is uint8, uncovered is uint8. Product is 1 if covered and uncovered.
        gains = np.sum(cov_matrix * uncovered, axis=1)
        
        best_i = np.argmax(gains)
        best_gain = gains[best_i]
        
        if best_gain == 0:
            break
            
        blocks.append(best_i)
        uncovered[cov_matrix[best_i] == 1] = 0
        
        if np.sum(uncovered) == 0:
            break
            
    return blocks


def _local_search(blocks, cov_matrix, indptr, indices, n_t, n_k, rng, time_budget, inv_index):
    """Local search using CSR for fast index retrieval and NumPy for counts."""
    blocks = list(blocks)
    
    # Initialize counts
    counts = np.zeros(n_t, dtype=np.int32)
    for b in blocks:
        start = indptr[b]
        end = indptr[b+1]
        if end > start:
            counts[indices[start:end]] += 1
            
    # Verify initial coverage
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
    
    while improved and _elapsed() < time_budget:
        improved = False
        stall_count += 1
        
        # --- Removal pass ---
        # Find a block where all covered t-subsets have count > 1
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
                break
        
        if removable_found:
            continue
            
        # --- Single swap pass ---
        # Try to replace block i with j, then check if any block becomes removable
        for _ in range(100):
            if _elapsed() >= time_budget:
                break
            if n_blocks == 0:
                break
                
            i = rng.randrange(n_blocks)
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
                
            # Check validity
            if not is_valid(counts):
                # Revert
                counts[indices[start_new:end_new]] -= 1
                if end_old > start_old:
                    counts[indices[start_old:end_old]] += 1
                blocks[i] = old
                continue
                
            # Try to find a removable block after the swap
            # Check the new block j first
            start_j = indptr[j]
            end_j = indptr[j+1]
            removed_block = None
            
            if end_j > start_j and counts[indices[start_j:end_j]].min() > 1:
                removed_block = i  # Remove the new block itself
            else:
                # Use inverted index to find blocks that overlap with j
                # and check if they are removable
                candidate_blocks = set()
                for idx in range(start_j, end_j):
                    t_idx = indices[idx]
                    for b_idx in inv_index[t_idx]:
                        if b_idx != j:
                            candidate_blocks.add(b_idx)
                
                # Check candidates for removability
                for cand in candidate_blocks:
                    if cand in blocks:
                        cand_pos = blocks.index(cand)
                        start_c = indptr[cand]
                        end_c = indptr[cand+1]
                        if end_c > start_c and counts[indices[start_c:end_c]].min() > 1:
                            removed_block = cand_pos
                            break
            
            if removed_block is not None:
                # Remove the block at removed_block position
                rem_idx = removed_block
                rem_b = blocks[rem_idx]
                start_rem = indptr[rem_b]
                end_rem = indptr[rem_b+1]
                if end_rem > start_rem:
                    counts[indices[start_rem:end_rem]] -= 1
                blocks.pop(rem_idx)
                n_blocks -= 1
                improved = True
                stall_count = 0
                continue # Restart loop
                
            # Revert swap
            counts[indices[start_j:end_j]] -= 1
            if end_old > start_old:
                counts[indices[start_old:end_old]] += 1
            blocks[i] = old

        # --- Perturbation if stalled ---
        if not improved and stall_count > 5 and n_blocks > 10:
            if _elapsed() < time_budget - 5:
                i = rng.randrange(n_blocks)
                removed_block = blocks.pop(i)
                start = indptr[removed_block]
                end = indptr[removed_block+1]
                if end > start:
                    counts[indices[start:end]] -= 1
                n_blocks -= 1
                # Repair
                while not is_valid(counts) and _elapsed() < time_budget - 2:
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

    return blocks


def construct(v, k, t):
    global _START
    _START = time.time()
    rng = random.Random(12345)

    all_t, t_index, k_subsets, cov_matrix, indptr, indices, n_t, n_k, inv_index = _precompute(v, k, t)

    # Greedy construction
    blocks = _greedy_construct(v, k, t, rng, cov_matrix, n_t, n_k)

    # Local search
    time_budget = _DEADLINE - 5
    blocks = _local_search(blocks, cov_matrix, indptr, indices, n_t, n_k, rng, time_budget, inv_index)

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
    # Deduplicate
    seen = set()
    dedup = []
    for blk in result:
        key = tuple(sorted(blk))
        if key not in seen:
            seen.add(key)
            dedup.append(list(key))
    return dedup
