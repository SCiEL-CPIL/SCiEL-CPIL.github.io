import itertools
import random
import time
import numpy as np

def _precompute(v, k, t):
    """Precompute t-subsets, k-blocks, and mappings."""
    tsubs = list(itertools.combinations(range(v), t))
    tsub_idx = {ts: i for i, ts in enumerate(tsubs)}
    n_tsubs = len(tsubs)
    
    blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(blocks)
    block_idx = {b: i for i, b in enumerate(blocks)}
    
    # For each block, list of t-subset indices it covers
    block_to_ts = [[] for _ in range(n_blocks)]
    # For each t-subset, list of block indices that cover it
    ts_to_blocks = [[] for _ in range(n_tsubs)]
    
    for bi, b in enumerate(blocks):
        for ts in itertools.combinations(b, t):
            tsi = tsub_idx[ts]
            block_to_ts[bi].append(tsi)
            ts_to_blocks[tsi].append(bi)
            
    # Convert to numpy arrays for faster access in loops
    block_to_ts_arr = [np.array(lst, dtype=np.int32) for lst in block_to_ts]
    ts_to_blocks_arr = [np.array(lst, dtype=np.int32) for lst in ts_to_blocks]
    
    return tsubs, tsub_idx, blocks, block_idx, block_to_ts_arr, ts_to_blocks_arr, n_tsubs, n_blocks

def _run_greedy(v, k, t, seed, precomp, deadline):
    """Vectorized randomized greedy set cover."""
    tsubs, tsub_idx, blocks, block_idx, block_to_ts_arr, ts_to_blocks_arr, n_tsubs, n_blocks = precomp
    rng = random.Random(seed)
    
    # gain[bi] = number of uncovered t-subsets covered by block bi
    gain = np.full(n_blocks, C(k, t), dtype=np.int32)
    uncovered = np.ones(n_tsubs, dtype=bool)
    selected = []
    
    while np.any(uncovered):
        if time.time() > deadline - 5.0:
            return None # Timeout, return None to indicate failure to complete
        
        # Find block with max gain
        max_gain = np.max(gain)
        if max_gain <= 0:
            break
            
        # Random tie-breaking
        candidates = np.where(gain == max_gain)[0]
        best_bi = candidates[rng.randrange(len(candidates))]
        
        selected.append(best_bi)
        
        # Get t-subsets covered by this block
        covered_ts = block_to_ts_arr[best_bi]
        
        # Identify which of these were actually uncovered
        was_uncovered_mask = uncovered[covered_ts]
        newly_covered_ts = covered_ts[was_uncovered_mask]
        
        if len(newly_covered_ts) == 0:
            # No new coverage, shouldn't happen if gain > 0, but safety
            gain[best_bi] = -1
            continue
            
        # Update uncovered status
        uncovered[newly_covered_ts] = False
        
        # Update gains for all blocks that cover any of the newly covered t-subsets
        # Collect all block indices
        all_block_indices = []
        for tsi in newly_covered_ts:
            all_block_indices.extend(ts_to_blocks_arr[tsi])
            
        if all_block_indices:
            # Count how many times each block index appears
            # minlength ensures array size is at least n_blocks
            counts = np.bincount(all_block_indices, minlength=n_blocks)
            gain -= counts
            
        # Mark selected block as unavailable (gain = -1)
        gain[best_bi] = -1
        
    return selected

def _local_search(v, k, t, block_indices, precomp, deadline):
    """Local search: redundancy removal and 1-swaps."""
    tsubs, tsub_idx, blocks, block_idx, block_to_ts_arr, ts_to_blocks_arr, n_tsubs, n_blocks = precomp
    rng = random.Random(42)
    
    # Work with a list of block indices
    sol = list(block_indices)
    
    # Build coverage array
    cov = np.zeros(n_tsubs, dtype=np.int32)
    for bi in sol:
        cov[block_to_ts_arr[bi]] += 1
        
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        
        # Phase 1: Remove redundant blocks
        # Shuffle to avoid bias
        indices = list(range(len(sol)))
        rng.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline - 2.0:
                return sol
            bi = sol[i]
            ts_idxs = block_to_ts_arr[bi]
            if np.all(cov[ts_idxs] > 1):
                # Redundant
                cov[ts_idxs] -= 1
                sol.pop(i)
                improved = True
                break # Restart loop to re-evaluate redundancies
                
        if improved:
            continue
            
        # Phase 2: Try 1-swaps
        # Try to replace a block with another that covers its critical t-subsets
        # We try a limited number of swaps per round
        num_swaps_tried = 0
        max_swaps = 20
        
        while num_swaps_tried < max_swaps and time.time() < deadline - 2.0:
            num_swaps_tried += 1
            if len(sol) == 0:
                break
                
            i = rng.randrange(len(sol))
            bi = sol[i]
            ts_idxs = block_to_ts_arr[bi]
            
            # Find critical t-subsets (covered only by bi)
            critical_mask = cov[ts_idxs] == 1
            critical_ts = ts_idxs[critical_mask]
            
            if len(critical_ts) == 0:
                # Should have been removed in Phase 1
                cov[ts_idxs] -= 1
                sol.pop(i)
                improved = True
                break
                
            # Find a replacement block that covers all critical_ts
            # Pick the critical t-subset with the fewest covering blocks to minimize search space
            # Sort critical_ts by len(ts_to_blocks[tsi])
            crit_sorted = sorted(critical_ts, key=lambda x: len(ts_to_blocks_arr[x]))
            first_crit = crit_sorted[0]
            
            found_replacement = False
            # Iterate through blocks covering first_crit
            # Randomize order to escape local minima
            candidates = list(ts_to_blocks_arr[first_crit])
            rng.shuffle(candidates)
            
            for bi2 in candidates:
                if bi2 == bi:
                    continue
                # Check if bi2 covers all critical_ts
                ts_idxs_bi2 = block_to_ts_arr[bi2]
                # Check intersection
                # We want critical_ts to be a subset of ts_idxs_bi2
                # Using np.isin is O(N*M), but N (critical) is small, M (bi2 ts) is 56
                # So this is fast
                if np.all(np.isin(critical_ts, ts_idxs_bi2)):
                    # Valid replacement
                    # Update coverage
                    cov[ts_idxs] -= 1
                    cov[ts_idxs_bi2] += 1
                    sol[i] = bi2
                    improved = True
                    found_replacement = True
                    break
                    
            if found_replacement:
                break # Restart loop to check for new redundancies
                
    return sol

def C(n, r):
    if r > n or r < 0:
        return 0
    if r == 0 or r == n:
        return 1
    r = min(r, n - r)
    result = 1
    for i in range(1, r + 1):
        result = result * (n - i + 1) // i
    return result

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    precomp = _precompute(v, k, t)
    
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple randomized greedy restarts
    # Each restart is fast due to vectorization
    num_restarts = 0
    while time.time() < deadline - 60.0: # Reserve 60s for local search
        seed = num_restarts
        blocks = _run_greedy(v, k, t, seed, precomp, deadline)
        num_restarts += 1
        if blocks is not None:
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
                # If we hit a very good count, we might stop early?
                # No, let's keep running to find potentially better ones.
        
    if best_blocks is None:
        # Fallback: simple greedy without vectorization if something went wrong
        # Or just return empty if no time? No, must return valid.
        # Use the first successful greedy result.
        blocks = _run_greedy(v, k, t, 0, precomp, deadline)
        if blocks is not None:
            best_blocks = blocks
        else:
            # Extreme fallback: just pick blocks until covered (slow, but safe)
            uncovered = set(itertools.combinations(range(v), t))
            best_blocks = []
            while uncovered and time.time() < deadline - 5.0:
                target = next(iter(uncovered))
                best_block = None
                best_gain = -1
                remaining = [x for x in range(v) if x not in target]
                need = k - t
                for comb in itertools.combinations(remaining, need):
                    block = tuple(sorted(target + comb))
                    gain = sum(1 for sub in itertools.combinations(block, t) if sub in uncovered)
                    if gain > best_gain:
                        best_gain = gain
                        best_block = block
                if best_block is None:
                    break
                best_blocks.append(best_block)
                uncovered.difference_update(itertools.combinations(best_block, t))
            # Convert to indices for local search? No, local search expects indices.
            # If we used fallback, we can't easily use local search.
            # But fallback is very unlikely.
            return [list(b) for b in best_blocks]

    # Local search on the best solution
    if time.time() < deadline - 5.0:
        best_blocks = _local_search(v, k, t, best_blocks, precomp, deadline)
        
    # Convert indices to actual blocks
    tsubs, tsub_idx, blocks_list, block_idx, block_to_ts_arr, ts_to_blocks_arr, n_tsubs, n_blocks = precomp
    final_blocks = []
    seen = set()
    for bi in best_blocks:
        b = blocks_list[bi]
        key = tuple(sorted(b))
        if key not in seen:
            seen.add(key)
            final_blocks.append(list(b))
            
    return final_blocks
