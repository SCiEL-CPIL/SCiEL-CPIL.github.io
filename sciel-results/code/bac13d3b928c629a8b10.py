import itertools
import random
import time
import numpy as np

def _precompute(v, k, t):
    """Precompute t-subsets, k-blocks, and mappings with bitsets."""
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
    
    # Bitset representation for each block: a Python int where bit i is set if block covers t-subset i
    # Since n_tsubs = C(19,5) = 11628, we need integers with ~11628 bits
    # Python handles big ints natively
    block_bitsets = [0] * n_blocks
    for bi in range(n_blocks):
        bits = 0
        for tsi in block_to_ts_arr[bi]:
            bits |= (1 << tsi)
        block_bitsets[bi] = bits
    
    # Also precompute bitsets for t-subset to blocks mapping (for quick intersection checks)
    # Actually, for VND we need to check if a set of t-subsets is covered by a block
    # We can use the bitset of the block and check if it contains all bits in a target set
    
    return tsubs, tsub_idx, blocks, block_idx, block_to_ts_arr, ts_to_blocks_arr, n_tsubs, n_blocks, block_bitsets

def _run_greedy(v, k, t, seed, precomp, deadline):
    """Vectorized randomized greedy set cover."""
    tsubs, tsub_idx, blocks, block_idx, block_to_ts_arr, ts_to_blocks_arr, n_tsubs, n_blocks, block_bitsets = precomp
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
        
        # Update gains: for each newly covered t-subset, decrement gain of blocks covering it
        # Use a more efficient approach: process in batches
        for tsi in newly_covered_ts:
            blocks_covering = ts_to_blocks_arr[tsi]
            gain[blocks_covering] -= 1
            
        # Mark selected block as unavailable (gain = -1)
        gain[best_bi] = -1
        
    return selected

def _vnd_local_search(v, k, t, block_indices, precomp, deadline):
    """Variable Neighborhood Descent local search with 1-swap and 2-swap moves."""
    tsubs, tsub_idx, blocks, block_idx, block_to_ts_arr, ts_to_blocks_arr, n_tsubs, n_blocks, block_bitsets = precomp
    rng = random.Random(42)
    
    # Work with a list of block indices
    sol = list(block_indices)
    
    # Build coverage array
    cov = np.zeros(n_tsubs, dtype=np.int32)
    for bi in sol:
        cov[block_to_ts_arr[bi]] += 1
        
    # Precompute sorted block_to_ts arrays for faster intersection checks
    block_to_ts_sorted = [np.sort(block_to_ts_arr[bi]) for bi in range(n_blocks)]
    
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        
        # Phase 1: Remove redundant blocks
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
        num_swaps_tried = 0
        max_swaps = 30
        
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
            # Use the most constrained t-subset to filter candidates
            crit_sorted = sorted(critical_ts, key=lambda x: len(ts_to_blocks_arr[x]))
            first_crit = crit_sorted[0]
            
            found_replacement = False
            candidates = list(ts_to_blocks_arr[first_crit])
            rng.shuffle(candidates)
            
            for bi2 in candidates:
                if bi2 == bi:
                    continue
                # Check if bi2 covers all critical_ts using sorted arrays
                ts_bi2_sorted = block_to_ts_sorted[bi2]
                # Binary search for each critical t-subset in ts_bi2_sorted
                all_covered = True
                for ct in critical_ts:
                    pos = np.searchsorted(ts_bi2_sorted, ct)
                    if pos >= len(ts_bi2_sorted) or ts_bi2_sorted[pos] != ct:
                        all_covered = False
                        break
                if all_covered:
                    # Valid replacement
                    cov[ts_idxs] -= 1
                    cov[block_to_ts_arr[bi2]] += 1
                    sol[i] = bi2
                    improved = True
                    found_replacement = True
                    break
                    
            if found_replacement:
                break # Restart loop to check for new redundancies
                
        if improved:
            continue
            
        # Phase 3: Try 2-swaps (VND level 2)
        # Try to replace two blocks with two different blocks
        if len(sol) >= 2:
            num_2swaps_tried = 0
            max_2swaps = 10
            
            while num_2swaps_tried < max_2swaps and time.time() < deadline - 2.0:
                num_2swaps_tried += 1
                
                # Pick two random blocks to remove
                i1 = rng.randrange(len(sol))
                i2 = rng.randrange(len(sol))
                if i1 == i2:
                    continue
                if i1 > i2:
                    i1, i2 = i2, i1
                    
                bi1 = sol[i1]
                bi2 = sol[i2]
                
                # Find t-subsets that would become uncovered if we remove both
                ts_idxs_1 = block_to_ts_arr[bi1]
                ts_idxs_2 = block_to_ts_arr[bi2]
                
                # Simulate removal
                temp_cov = cov.copy()
                temp_cov[ts_idxs_1] -= 1
                temp_cov[ts_idxs_2] -= 1
                
                # Find t-subsets that would become uncovered
                newly_uncovered = np.where(temp_cov == 0)[0]
                
                if len(newly_uncovered) == 0:
                    # Both blocks are redundant together
                    cov[ts_idxs_1] -= 1
                    cov[ts_idxs_2] -= 1
                    # Remove both, keeping indices sorted
                    sol.pop(i2)
                    sol.pop(i1)
                    improved = True
                    break
                    
                # We need to find two blocks that cover all newly_uncovered t-subsets
                # Strategy: pick the most constrained t-subset, find blocks covering it,
                # then for each candidate pair, check if they cover all newly_uncovered
                
                # Sort newly_uncovered by number of covering blocks
                newly_uncovered_sorted = sorted(newly_uncovered, key=lambda x: len(ts_to_blocks_arr[x]))
                first_nu = newly_uncovered_sorted[0]
                
                # Candidates for first block
                cand1 = list(ts_to_blocks_arr[first_nu])
                rng.shuffle(cand1)
                
                found_2swap = False
                for bi_new1 in cand1[:20]:  # Limit candidates
                    if bi_new1 == bi1 or bi_new1 == bi2:
                        continue
                    
                    ts_idxs_new1 = block_to_ts_arr[bi_new1]
                    ts_new1_sorted = block_to_ts_sorted[bi_new1]
                    
                    # Which newly_uncovered are still not covered by bi_new1?
                    remaining_uncovered = []
                    for tsi in newly_uncovered:
                        pos = np.searchsorted(ts_new1_sorted, tsi)
                        if pos >= len(ts_new1_sorted) or ts_new1_sorted[pos] != tsi:
                            remaining_uncovered.append(tsi)
                    
                    if len(remaining_uncovered) == 0:
                        # bi_new1 alone covers all newly_uncovered
                        continue
                    
                    # Find second block to cover remaining_uncovered
                    # Pick the most constrained remaining
                    remaining_sorted = sorted(remaining_uncovered, key=lambda x: len(ts_to_blocks_arr[x]))
                    second_nu = remaining_sorted[0]
                    
                    cand2 = list(ts_to_blocks_arr[second_nu])
                    rng.shuffle(cand2)
                    
                    for bi_new2 in cand2[:20]:
                        if bi_new2 == bi1 or bi_new2 == bi2 or bi_new2 == bi_new1:
                            continue
                        
                        ts_idxs_new2 = block_to_ts_arr[bi_new2]
                        ts_new2_sorted = block_to_ts_sorted[bi_new2]
                        
                        # Check if bi_new1 and bi_new2 together cover all newly_uncovered
                        all_covered = True
                        for tsi in newly_uncovered:
                            # Check if covered by bi_new1
                            pos1 = np.searchsorted(ts_new1_sorted, tsi)
                            covered_by_1 = pos1 < len(ts_new1_sorted) and ts_new1_sorted[pos1] == tsi
                            if covered_by_1:
                                continue
                            # Check if covered by bi_new2
                            pos2 = np.searchsorted(ts_new2_sorted, tsi)
                            covered_by_2 = pos2 < len(ts_new2_sorted) and ts_new2_sorted[pos2] == tsi
                            if not covered_by_2:
                                all_covered = False
                                break
                        
                        if all_covered:
                            # Valid 2-swap
                            # Update coverage
                            cov[ts_idxs_1] -= 1
                            cov[ts_idxs_2] -= 1
                            cov[ts_idxs_new1] += 1
                            cov[ts_idxs_new2] += 1
                            
                            # Update sol
                            sol[i1] = bi_new1
                            sol[i2] = bi_new2
                            improved = True
                            found_2swap = True
                            break
                            
                    if found_2swap:
                        break
                        
            if found_2swap:
                break # Restart main loop
                
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
    # Use adaptive time allocation: spend more time on greedy if we're finding good solutions
    num_restarts = 0
    greedy_deadline = deadline - 45.0  # Reserve 45s for local search initially
    
    while time.time() < greedy_deadline:
        seed = num_restarts
        blocks = _run_greedy(v, k, t, seed, precomp, deadline)
        num_restarts += 1
        if blocks is not None:
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
                # If we found a good solution, reduce local search time to allow more restarts
                if best_count <= 340:
                    greedy_deadline = min(deadline - 20.0, greedy_deadline + 5.0)
                elif best_count <= 350:
                    greedy_deadline = min(deadline - 30.0, greedy_deadline + 3.0)
        
    if best_blocks is None:
        # Fallback
        blocks = _run_greedy(v, k, t, 0, precomp, deadline)
        if blocks is not None:
            best_blocks = blocks
        else:
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
            return [list(b) for b in best_blocks]

    # VND local search on the best solution
    if time.time() < deadline - 5.0:
        best_blocks = _vnd_local_search(v, k, t, best_blocks, precomp, deadline)
        
    # Convert indices to actual blocks
    tsubs, tsub_idx, blocks_list, block_idx, block_to_ts_arr, ts_to_blocks_arr, n_tsubs, n_blocks, block_bitsets = precomp
    final_blocks = []
    seen = set()
    for bi in best_blocks:
        b = blocks_list[bi]
        key = tuple(sorted(b))
        if key not in seen:
            seen.add(key)
            final_blocks.append(list(b))
            
    return final_blocks
