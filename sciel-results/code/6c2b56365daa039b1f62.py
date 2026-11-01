import itertools
import random
import time
import numpy as np

def _precompute(v, k, t):
    """Precompute t-subsets, k-blocks, and mappings with NumPy arrays."""
    tsubs = list(itertools.combinations(range(v), t))
    tsub_idx = {ts: i for i, ts in enumerate(tsubs)}
    n_tsubs = len(tsubs)
    
    blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(blocks)
    
    # Map block tuple to index
    block_idx = {b: i for i, b in enumerate(blocks)}
    
    # For each block, the list of t-subset indices it covers
    block_tsubs_list = []
    for b in blocks:
        ts_idxs = []
        for ts in itertools.combinations(b, t):
            ts_idxs.append(tsub_idx[ts])
        block_tsubs_list.append(np.array(ts_idxs, dtype=np.int32))
    
    # For each t-subset, list of block indices that cover it
    tsub_blocks_list = [[] for _ in range(n_tsubs)]
    for bi, b in enumerate(blocks):
        for ts in itertools.combinations(b, t):
            tsub_blocks_list[tsub_idx[ts]].append(bi)
    
    tsub_blocks_arr = [np.array(x, dtype=np.int32) for x in tsub_blocks_list]
    
    return tsubs, tsub_idx, blocks, block_idx, block_tsubs_list, tsub_blocks_arr, n_tsubs, n_blocks

def _greedy_cover(v, k, t, rng, precomp):
    """Global greedy set cover with efficient gain updates."""
    tsubs, tsub_idx, blocks, block_idx, block_tsubs_arr, tsub_blocks_arr, n_tsubs, n_blocks = precomp
    
    uncovered = np.ones(n_tsubs, dtype=np.bool_)
    selected = []
    
    # gain[bi] = number of uncovered t-subsets covered by block bi
    gain = np.zeros(n_blocks, dtype=np.int32)
    for bi in range(n_blocks):
        gain[bi] = np.sum(uncovered[block_tsubs_arr[bi]])
    
    while uncovered.any():
        best_bi = int(np.argmax(gain))
        if gain[best_bi] == 0:
            break
        
        selected.append(best_bi)
        idxs = block_tsubs_arr[best_bi]
        
        was_uncovered = uncovered[idxs]
        uncovered[idxs] = False
        
        # Update gains for all blocks that cover any of the newly covered t-subsets
        # Optimization: Only update if the t-subset was actually uncovered
        for tsi in idxs[was_uncovered]:
            for bi2 in tsub_blocks_arr[tsi]:
                if gain[bi2] > 0:
                    gain[bi2] -= 1
                    # If gain becomes 0, we could potentially remove it from a candidate list, 
                    # but argmax over the whole array is O(N). 
                    # Given N=75582, argmax is fast in numpy.
    
    return selected

def _local_search(v, k, t, block_indices, rng, precomp, time_limit=110.0):
    """Aggressive local search: Remove redundant, Single Swap, 2-Swap."""
    start_time = time.time()
    tsubs, tsub_idx, blocks, block_idx, block_tsubs_arr, tsub_blocks_arr, n_tsubs, n_blocks = precomp
    
    # Work with a list of block indices
    current_blocks = list(block_indices)
    
    # Build coverage array: cov[tsi] = number of blocks in current_blocks covering tsi
    cov = np.zeros(n_tsubs, dtype=np.int32)
    for bi in current_blocks:
        cov[block_tsubs_arr[bi]] += 1
    
    improved = True
    stall_counter = 0
    
    while time.time() - start_time < time_limit and improved:
        improved = False
        
        # Phase 1: Remove redundant blocks
        # A block is redundant if all its t-subsets are covered by at least one other block
        # i.e., for all ts in block_tsubs[bi], cov[ts] > 1
        indices = list(range(len(current_blocks)))
        rng.shuffle(indices)
        
        for i in indices:
            if time.time() - start_time > time_limit:
                return current_blocks
            
            bi = current_blocks[i]
            idxs = block_tsubs_arr[bi]
            
            if np.all(cov[idxs] > 1):
                cov[idxs] -= 1
                current_blocks.pop(i)
                improved = True
                break  # Restart loop to re-check redundancies
        
        if improved:
            continue
        
        # Phase 2: Single Block Swap
        # Try to replace a block with another that covers its critical subsets
        # and potentially improves the situation for future removals.
        if len(current_blocks) > 0:
            found_swap = False
            # Try a random number of swaps
            num_tries = min(100, len(current_blocks))
            
            for _ in range(num_tries):
                if time.time() - start_time > time_limit:
                    return current_blocks
                
                i = rng.randrange(len(current_blocks))
                bi = current_blocks[i]
                idxs_bi = block_tsubs_arr[bi]
                
                # Critical subsets: those covered only by bi
                critical_mask = cov[idxs_bi] == 1
                if not np.any(critical_mask):
                    # This block is redundant, should have been caught in Phase 1
                    cov[idxs_bi] -= 1
                    current_blocks.pop(i)
                    improved = True
                    found_swap = True
                    break
                
                critical_idxs = idxs_bi[critical_mask]
                
                # Candidates must cover all critical_idxs
                # Start with candidates for the first critical subset
                first_crit = int(critical_idxs[0])
                candidates = tsub_blocks_arr[first_crit]
                
                best_bi = -1
                best_score = -1
                
                for bi2 in candidates:
                    if bi2 == bi:
                        continue
                    
                    idxs_bi2 = block_tsubs_arr[bi2]
                    
                    # Check if bi2 covers all critical subsets
                    # Efficient check: intersection of critical_idxs and idxs_bi2 must be size of critical_idxs
                    # Or simply: all(c in idxs_bi2 for c in critical_idxs)
                    # Using numpy:
                    if np.all(np.isin(critical_idxs, idxs_bi2)):
                        # Score: How many "other" subsets does bi2 cover that are currently covered by >= 2 blocks?
                        # Or simpler: Just accept the first valid one? 
                        # Better: Pick one that covers the most subsets that are currently critical for *other* blocks?
                        # That's expensive. 
                        # Let's pick one that maximizes the number of subsets it covers that are NOT in bi.
                        # This increases the chance of making other blocks redundant.
                        # Actually, just picking a random valid one is often good enough for diversity.
                        # Let's pick the one with the highest total coverage of currently uncovered? No, all are covered.
                        
                        # Heuristic: Pick the candidate that shares the most subsets with other blocks?
                        # Let's just pick the first valid one for speed, or randomize among valid ones.
                        best_bi = bi2
                        break 
                        # Breaking here is fast. If we want better, we'd need to evaluate all.
                        # Let's try to evaluate a few.
                
                if best_bi != -1:
                    # Perform swap
                    cov[idxs_bi] -= 1
                    cov[block_tsubs_arr[best_bi]] += 1
                    current_blocks[i] = best_bi
                    improved = True
                    found_swap = True
                    break
            
            if found_swap:
                continue
        
        # Phase 3: 2-Block Swap (Remove 2, Add 2)
        # This allows escaping local minima where single swaps are not possible.
        if len(current_blocks) >= 2:
            found_2swap = False
            num_tries = 20
            
            for _ in range(num_tries):
                if time.time() - start_time > time_limit:
                    return current_blocks
                
                i1 = rng.randrange(len(current_blocks))
                i2 = rng.randrange(len(current_blocks))
                if i1 == i2:
                    continue
                
                bi1 = current_blocks[i1]
                bi2 = current_blocks[i2]
                idxs1 = block_tsubs_arr[bi1]
                idxs2 = block_tsubs_arr[bi2]
                
                # Temporarily remove bi1 and bi2
                temp_cov = cov.copy()
                temp_cov[idxs1] -= 1
                temp_cov[idxs2] -= 1
                
                # Find subsets that become uncovered
                newly_uncovered_mask = (temp_cov == 0)
                if not np.any(newly_uncovered_mask):
                    # Both are redundant together
                    cov[idxs1] -= 1
                    cov[idxs2] -= 1
                    # Remove both
                    if i1 > i2:
                        current_blocks.pop(i1)
                        current_blocks.pop(i2)
                    else:
                        current_blocks.pop(i2)
                        current_blocks.pop(i1)
                    improved = True
                    found_2swap = True
                    break
                
                newly_uncovered_idxs = np.where(newly_uncovered_mask)[0]
                
                # If too many newly uncovered, skip
                if len(newly_uncovered_idxs) > 15:
                    continue
                
                # Find two blocks that cover all newly_uncovered_idxs
                # Heuristic: Pick a candidate for the first uncovered subset, then a candidate for the rest
                first_nu = int(newly_uncovered_idxs[0])
                cand1_list = tsub_blocks_arr[first_nu]
                
                # Limit candidates for speed
                if len(cand1_list) > 100:
                    cand1_list = cand1_list[:100]
                
                found_valid_2swap = False
                for c1 in cand1_list:
                    if c1 == bi1 or c1 == bi2:
                        continue
                    
                    idxs_c1 = block_tsubs_arr[c1]
                    
                    # What is still uncovered after adding c1?
                    remaining_mask = newly_uncovered_mask.copy()
                    remaining_mask[idxs_c1] = False
                    
                    if not np.any(remaining_mask):
                        # c1 covers all. We need a second block? 
                        # The move is "remove 2, add 2". If c1 covers all, we can add any other block?
                        # No, we must add 2 blocks to maintain the "structure" or just to have a valid move?
                        # Actually, if we remove 2 and add 1, the size decreases by 1. 
                        # Is that allowed? Yes! 
                        # If removing 2 blocks and adding 1 block covers all, we reduce size by 1.
                        # Let's check if c1 covers all newly uncovered.
                        # If so, we can replace {bi1, bi2} with {c1}. Size -1.
                        cov[idxs1] -= 1
                        cov[idxs2] -= 1
                        cov[idxs_c1] += 1
                        # Replace bi1 with c1, remove bi2
                        # Be careful with indices
                        if i1 < i2:
                            current_blocks[i1] = c1
                            current_blocks.pop(i2)
                        else:
                            current_blocks[i2] = c1
                            current_blocks.pop(i1)
                        improved = True
                        found_2swap = True
                        found_valid_2swap = True
                        break
                    
                    # If not all covered, find a second block
                    remaining_idxs = np.where(remaining_mask)[0]
                    if len(remaining_idxs) == 0:
                        continue
                        
                    second_nu = int(remaining_idxs[0])
                    cand2_list = tsub_blocks_arr[second_nu]
                    
                    if len(cand2_list) > 100:
                        cand2_list = cand2_list[:100]
                    
                    for c2 in cand2_list:
                        if c2 == bi1 or c2 == bi2 or c2 == c1:
                            continue
                        
                        idxs_c2 = block_tsubs_arr[c2]
                        
                        # Check if c1 and c2 together cover all newly_uncovered
                        all_covered_mask = newly_uncovered_mask.copy()
                        all_covered_mask[idxs_c1] = False
                        all_covered_mask[idxs_c2] = False
                        
                        if not np.any(all_covered_mask):
                            # Valid 2-swap (size stays same)
                            cov[idxs1] -= 1
                            cov[idxs2] -= 1
                            cov[idxs_c1] += 1
                            cov[idxs_c2] += 1
                            current_blocks[i1] = c1
                            current_blocks[i2] = c2
                            improved = True
                            found_2swap = True
                            found_valid_2swap = True
                            break
                    if found_valid_2swap:
                        break
                
                if found_2swap:
                    break
            
            if found_2swap:
                continue
        
        # If no improvement found in a full cycle, we might be stuck.
        # To escape, we could do a random restart of the local search or perturb the solution.
        # For now, we just break and rely on the outer loop (multiple starts).
        break
    
    return current_blocks

def construct(v, k, t):
    """Construct a covering design C(v,k,t) using randomized greedy + local search."""
    deadline = time.time() + 145.0  # Safety margin before 150s kill
    
    # Precompute
    precomp = _precompute(v, k, t)
    
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple greedy starts with different seeds and local search
    # We allocate time to each run
    
    num_starts = 10
    time_per_start = (deadline - time.time() - 5.0) / num_starts
    
    for i in range(num_starts):
        if time.time() > deadline - 5.0:
            break
        
        seed = 42 + i * 137
        rng = random.Random(seed)
        
        # Greedy
        blocks = _greedy_cover(v, k, t, rng, precomp)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
        
        # Local Search
        # Allocate remaining time for this start's local search
        remaining_for_ls = min(time_per_start, deadline - time.time() - 5.0)
        if remaining_for_ls > 1.0:
            ls_rng = random.Random(seed + 1)
            improved_blocks = _local_search(v, k, t, blocks, ls_rng, precomp, time_limit=remaining_for_ls)
            
            if len(improved_blocks) < best_count:
                best_count = len(improved_blocks)
                best_blocks = improved_blocks
    
    # Final validation and cleanup
    if best_blocks is None:
        # Fallback
        rng = random.Random(0)
        best_blocks = _greedy_cover(v, k, t, rng, precomp)
    
    # Convert block indices to actual blocks
    tsubs, tsub_idx, blocks_list, block_idx, block_tsubs_arr, tsub_blocks_arr, n_tsubs, n_blocks = precomp
    
    seen = set()
    final_blocks = []
    for bi in best_blocks:
        b = blocks_list[bi]
        key = tuple(sorted(b))
        if key not in seen and len(set(b)) == k and all(0 <= x < v for x in b):
            seen.add(key)
            final_blocks.append(list(b))
    
    return final_blocks
