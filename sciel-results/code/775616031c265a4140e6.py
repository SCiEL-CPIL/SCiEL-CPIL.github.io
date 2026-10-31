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
    
    return all_t, t_index, k_subsets, cov_matrix, indptr, indices, n_t, n_k


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
            
    # Verify initial coverage
    # A block is valid if all its t-subsets have count >= 1
    # If any t-subset has count 0, we need to repair
    
    def is_valid(counts):
        return np.all(counts > 0)
        
    if not is_valid(counts):
        # Repair
        while not is_valid(counts) and _elapsed() < time_budget:
            missing = np.where(counts == 0)[0]
            if len(missing) == 0: break
            # Find block that covers most missing
            # We can use cov_matrix, but we need to mask only missing parts
            # gain = sum(cov_matrix[:, missing], axis=1)
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
            # Check min count in this block
            if counts[indices[start:end]].min() > 1:
                # Remove
                counts[indices[start:end]] -= 1
                blocks.pop(i)
                n_blocks -= 1
                improved = True
                stall_count = 0
                removable_found = True
                break # Restart removal pass? Or continue? Continue is fine, but indices shift.
                # Since we popped, subsequent indices are shifted.
                # It's safer to break and restart the outer loop or just break this pass.
                break
        
        if removable_found:
            continue # Restart loop to re-evaluate removals/swaps with new state
            
        # --- Single swap pass ---
        # Try to replace block i with j, then remove if possible
        # To be efficient, we can just try random swaps and check validity + removability
        
        for _ in range(50): # Limit iterations per stall
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
            # Remove old
            start_old = indptr[old]
            end_old = indptr[old+1]
            if end_old > start_old:
                counts[indices[start_old:end_old]] -= 1
            # Add new
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
                
            # Try removal
            # Look for any block that is now removable
            # Optimization: Only check blocks that were affected? 
            # The new block j might be removable? Or an old block?
            # Actually, any block could be removable if its counts went up.
            # But counts only went up for blocks covered by j, and down for blocks covered by old.
            # Blocks covered by old might now have count 1 (bad for removal).
            # Blocks covered by j might now have count > 1 (good for removal).
            # So we only need to check blocks that intersect with j's coverage?
            # And maybe blocks that intersect with old's coverage if they are not j?
            # Let's just check a few random blocks or the new block itself.
            
            # Check if new block j is removable
            start_j = indptr[j]
            end_j = indptr[j+1]
            if end_j > start_j and counts[indices[start_j:end_j]].min() > 1:
                # Remove j
                counts[indices[start_j:end_j]] -= 1
                blocks.pop(i)
                n_blocks -= 1
                improved = True
                stall_count = 0
                continue # Restart loop
                
            # Check if old block (now gone) caused others to be removable? 
            # No, removing old decreased counts. So old's neighbors are less removable.
            # Adding j increased counts. So j's neighbors are more removable.
            # We should check blocks that overlap with j.
            # Get indices of j
            j_indices = indices[start_j:end_j]
            # Find blocks that contain these indices? 
            # This requires an inverted index (t-subset -> list of blocks).
            # We don't have that built. 
            # For now, just checking j is a good first step.
            # We can also check a random other block.
            
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

    all_t, t_index, k_subsets, cov_matrix, indptr, indices, n_t, n_k = _precompute(v, k, t)

    # Greedy construction
    blocks = _greedy_construct(v, k, t, rng, cov_matrix, n_t, n_k)

    # Local search
    time_budget = _DEADLINE - 5
    blocks = _local_search(blocks, cov_matrix, indptr, indices, n_t, n_k, rng, time_budget)

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
