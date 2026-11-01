"""Covering design C(19,7,4): minimize number of 7-subsets covering all 4-subsets.
Strategy: NumPy-accelerated greedy + local search with removal/swap.
"""
import time
import itertools
import random
import numpy as np

_DEADLINE = 115.0
_START = None


def _elapsed():
    return time.time() - _START


def _precompute(v, k, t):
    all_t = list(itertools.combinations(range(v), t))
    n_t = len(all_t)
    t_index = {c: i for i, c in enumerate(all_t)}
    
    all_k = list(itertools.combinations(range(v), k))
    n_k = len(all_k)
    
    # Precompute coverage indices for each block
    # block_cov_flat is a 2D array of shape (n_k, C(k, t))
    # Each row contains the indices of the t-subsets covered by that block
    cov_size = len(list(itertools.combinations(range(k), t)))
    block_cov_flat = np.zeros((n_k, cov_size), dtype=np.int32)
    
    for b, ks in enumerate(all_k):
        for j, ts in enumerate(itertools.combinations(ks, t)):
            block_cov_flat[b, j] = t_index[ts]
            
    # t_to_blocks[s] = list of block indices that cover t-subset s
    t_to_blocks = [[] for _ in range(n_t)]
    for b in range(n_k):
        for s in block_cov_flat[b]:
            t_to_blocks[s].append(b)
            
    # Convert to numpy arrays for faster access in repair
    t_to_blocks_arr = [np.array(blks, dtype=np.int32) for blks in t_to_blocks]
    
    return all_t, t_index, all_k, block_cov_flat, t_to_blocks_arr, n_t, n_k


def _greedy_construct(v, k, t, rng):
    all_t, t_index, all_k, block_cov_flat, t_to_blocks_arr, n_t, n_k = _precompute(v, k, t)
    
    selected = set()
    cover_count = np.zeros(n_t, dtype=np.int32)
    uncovered_count = n_t
    
    while uncovered_count > 0 and _elapsed() < _DEADLINE - 25:
        # Compute gains for all blocks in one vectorized operation
        # cover_count[block_cov_flat] has shape (n_k, cov_size)
        # We want to count how many of these are 0 for each block
        gains = np.sum(cover_count[block_cov_flat] == 0, axis=1)
        
        best_b = np.argmax(gains)
        best_gain = gains[best_b]
        
        if best_gain == 0:
            break
            
        selected.add(best_b)
        # Update cover counts for the selected block
        cov_indices = block_cov_flat[best_b]
        for s in cov_indices:
            if cover_count[s] == 0:
                uncovered_count -= 1
            cover_count[s] += 1
            
    return selected, all_k, block_cov_flat, t_to_blocks_arr, n_t, n_k


def _local_search(selected, block_cov_flat, t_to_blocks_arr, n_t, n_k, rng, time_budget):
    selected = set(selected)
    cov_size = block_cov_flat.shape[1]
    
    # Compute cover count
    cover_count = np.zeros(n_t, dtype=np.int32)
    for b in selected:
        for s in block_cov_flat[b]:
            cover_count[s] += 1
            
    improved = True
    while improved and _elapsed() < time_budget:
        improved = False
        
        # --- Removal pass ---
        # Try removing each block if it doesn't have unique coverage
        # To speed up, we can check uniqueness more efficiently
        blocks_list = list(selected)
        for b in blocks_list:
            if _elapsed() >= time_budget:
                break
            if b not in selected:
                continue
            
            # Check if any t-subset in this block is uniquely covered by it
            # i.e., cover_count[s] == 1
            cov_indices = block_cov_flat[b]
            # If any of these have count 1, we cannot remove
            if np.any(cover_count[cov_indices] == 1):
                continue
                
            # Safe to remove
            selected.discard(b)
            for s in cov_indices:
                cover_count[s] -= 1
            improved = True
            # Restart removal pass after a removal to catch cascading redundancies
            # But for efficiency, we can just continue and check again in next outer loop
            # However, the current code restarts. Let's keep it simple: break and restart
            break 
            
        # --- Swap pass ---
        # Try to swap a block with another to enable a removal
        # This is more complex. We try to replace b with b' such that 
        # the new set is a cover, and then we can remove a redundant block.
        # Actually, if we replace b with b', the size is the same. 
        # We only improve if we can then remove another block.
        
        for _ in range(50):
            if _elapsed() >= time_budget:
                break
            if not selected:
                break
            b = rng.choice(list(selected))
            new_b = rng.randrange(n_k)
            if new_b in selected:
                continue
            
            # Tentative swap
            # 1. Remove b
            cov_b = block_cov_flat[b]
            for s in cov_b:
                cover_count[s] -= 1
            selected.discard(b)
            
            # 2. Add new_b
            cov_new = block_cov_flat[new_b]
            for s in cov_new:
                cover_count[s] += 1
            selected.add(new_b)
            
            # 3. Check if we can remove any block (including new_b or others)
            # We look for a block c in selected such that no s in block_cov_flat[c] has cover_count[s] == 1
            removed = False
            # Randomize order to avoid bias? Or just iterate
            for cb in list(selected):
                cov_cb = block_cov_flat[cb]
                if not np.any(cover_count[cov_cb] == 1):
                    # Remove cb
                    selected.discard(cb)
                    for s in cov_cb:
                        cover_count[s] -= 1
                    removed = True
                    break
            
            if not removed:
                # Revert
                selected.discard(new_b)
                for s in cov_new:
                    cover_count[s] -= 1
                selected.add(b)
                for s in cov_b:
                    cover_count[s] += 1
            else:
                improved = True
                # If we improved, we can break the swap loop for this iteration 
                # or continue to try more swaps.
                # Let's break to allow the removal pass to run again in the outer loop
                break

    return selected


def construct(v, k, t):
    global _START
    _START = time.time()
    rng = random.Random(42)
    
    # 1. Greedy construction
    selected, all_k, block_cov_flat, t_to_blocks_arr, n_t, n_k = _greedy_construct(v, k, t, rng)
    
    # 2. Local search
    time_budget = _DEADLINE - 5
    selected = _local_search(selected, block_cov_flat, t_to_blocks_arr, n_t, n_k, rng, time_budget)
    
    # 3. Safety check & repair
    cover_count = np.zeros(n_t, dtype=np.int32)
    for b in selected:
        for s in block_cov_flat[b]:
            cover_count[s] += 1
    
    missing = np.where(cover_count == 0)[0]
    if len(missing) > 0:
        missing_set = set(missing.tolist())
        while missing_set and _elapsed() < _DEADLINE - 2:
            # Find block covering most missing
            gain_counts = {}
            for s in missing_set:
                for b in t_to_blocks_arr[s]:
                    gain_counts[b] = gain_counts.get(b, 0) + 1
            if not gain_counts:
                break
            best_b = max(gain_counts, key=gain_counts.get)
            selected.add(best_b)
            for s in block_cov_flat[best_b]:
                if s in missing_set:
                    missing_set.discard(s)
    
    # Return as list of lists
    result = [list(all_k[b]) for b in selected]
    seen = set()
    dedup = []
    for blk in result:
        key = tuple(sorted(blk))
        if key not in seen:
            seen.add(key)
            dedup.append(list(key))
    return dedup
