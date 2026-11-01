"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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
        block_tsubs_list.append(ts_idxs)
    
    # Build a dense coverage matrix: coverage_matrix[block_idx, tsub_idx] = 1 if block covers tsub
    # This is n_blocks x n_tsubs, which for C(19,8,5): 75582 x 11628 ~ 878M entries - too large!
    # Instead, keep sparse representation.
    
    # For each t-subset, list of block indices that cover it
    tsub_blocks = [[] for _ in range(n_tsubs)]
    for bi, b in enumerate(blocks):
        for ts in itertools.combinations(b, t):
            tsub_blocks[tsub_idx[ts]].append(bi)
    
    # Convert to numpy arrays for fast access
    block_tsubs_arr = [np.array(x, dtype=np.int32) for x in block_tsubs_list]
    tsub_blocks_arr = [np.array(x, dtype=np.int32) for x in tsub_blocks]
    
    return tsubs, tsub_idx, blocks, block_idx, block_tsubs_arr, tsub_blocks_arr, n_tsubs, n_blocks


def _greedy_cover(v, k, t, rng, precomp):
    """Global greedy set cover: always pick the block covering most uncovered t-subsets."""
    tsubs, tsub_idx, blocks, block_idx, block_tsubs_arr, tsub_blocks_arr, n_tsubs, n_blocks = precomp
    
    uncovered = np.ones(n_tsubs, dtype=np.bool_)
    selected = []
    
    # Precompute: for each block, its t-subset indices
    # We need to efficiently find the block with max uncovered coverage
    
    # Strategy: maintain a gain array, update after each selection
    # gain[bi] = number of uncovered t-subsets covered by block bi
    gain = np.zeros(n_blocks, dtype=np.int32)
    for bi in range(n_blocks):
        gain[bi] = np.sum(uncovered[block_tsubs_arr[bi]])
    
    while uncovered.any():
        # Find block with max gain
        best_bi = int(np.argmax(gain))
        if gain[best_bi] == 0:
            break  # No block can cover any more uncovered t-subsets
        
        selected.append(best_bi)
        idxs = block_tsubs_arr[best_bi]
        
        # Mark as covered
        was_uncovered = uncovered[idxs]
        uncovered[idxs] = False
        
        # Update gains: for each block that covers any of the newly covered t-subsets, decrement
        # This is expensive; do it lazily or approximately
        # For speed, just recompute gain for all blocks that share t-subsets with best_bi
        # Actually, let's just recompute gain for all blocks - it's O(n_blocks * avg_tsubs_per_block)
        # For C(19,8,5): 75582 * C(8,5)=56 ~ 4.2M ops per iteration, too slow for many iterations
        
        # Better: only update gains for blocks that cover any of the newly covered t-subsets
        for tsi in idxs[was_uncovered]:
            for bi2 in tsub_blocks_arr[tsi]:
                if gain[bi2] > 0:
                    gain[bi2] -= 1
    
    return selected


def _greedy_cover_randomized(v, k, t, rng, precomp, alpha=0.3):
    """Randomized greedy: with probability alpha, pick a random uncovered t-subset and 
    find a good block for it; otherwise use global greedy."""
    tsubs, tsub_idx, blocks, block_idx, block_tsubs_arr, tsub_blocks_arr, n_tsubs, n_blocks = precomp
    
    uncovered = np.ones(n_tsubs, dtype=np.bool_)
    selected = []
    
    # Precompute gain
    gain = np.zeros(n_blocks, dtype=np.int32)
    for bi in range(n_blocks):
        gain[bi] = np.sum(uncovered[block_tsubs_arr[bi]])
    
    while uncovered.any():
        if rng.random() < alpha:
            # Pick random uncovered t-subset
            uncovered_indices = np.where(uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            target_tsi = uncovered_indices[rng.randrange(len(uncovered_indices))]
            
            # Find best block covering this t-subset
            candidates = tsub_blocks_arr[target_tsi]
            if len(candidates) == 0:
                break
            best_bi = -1
            best_gain = -1
            for bi in candidates:
                g = np.sum(uncovered[block_tsubs_arr[bi]])
                if g > best_gain:
                    best_gain = g
                    best_bi = bi
            if best_bi == -1 or best_gain <= 0:
                break
        else:
            # Global greedy
            best_bi = int(np.argmax(gain))
            if gain[best_bi] == 0:
                break
        
        selected.append(best_bi)
        idxs = block_tsubs_arr[best_bi]
        was_uncovered = uncovered[idxs]
        uncovered[idxs] = False
        
        # Update gains
        for tsi in idxs[was_uncovered]:
            for bi2 in tsub_blocks_arr[tsi]:
                if gain[bi2] > 0:
                    gain[bi2] -= 1
    
    return selected


def _local_search(v, k, t, block_indices, rng, precomp, time_limit=110.0):
    """Local search: delete redundant blocks, swap blocks, 2-swap moves."""
    start_time = time.time()
    tsubs, tsub_idx, blocks, block_idx, block_tsubs_arr, tsub_blocks_arr, n_tsubs, n_blocks = precomp
    
    blocks = list(block_indices)
    n = len(blocks)
    
    # Build coverage array
    cov = np.zeros(n_tsubs, dtype=np.int32)
    for bi in blocks:
        cov[block_tsubs_arr[bi]] += 1
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Phase 1: Delete redundant blocks
        indices = list(range(len(blocks)))
        rng.shuffle(indices)
        
        for i in indices:
            if time.time() - start_time > time_limit:
                break
            bi = blocks[i]
            idxs = block_tsubs_arr[bi]
            if np.all(cov[idxs] > 1):
                cov[idxs] -= 1
                blocks.pop(i)
                improved = True
                break
        
        if improved:
            continue
        
        # Phase 2: Try single-block swaps
        if len(blocks) == 0:
            break
        
        found_swap = False
        num_tries = min(50, len(blocks))
        for _ in range(num_tries):
            if time.time() - start_time > time_limit:
                break
            
            i = rng.randrange(len(blocks))
            bi = blocks[i]
            idxs_bi = block_tsubs_arr[bi]
            
            # Find critical t-subsets (covered only by bi)
            critical_mask = cov[idxs_bi] == 1
            critical_idxs = idxs_bi[critical_mask]
            
            if len(critical_idxs) == 0:
                # Redundant - should have been caught in Phase 1
                cov[idxs_bi] -= 1
                blocks.pop(i)
                improved = True
                found_swap = True
                break
            
            # Find replacement: must cover all critical t-subsets
            first_crit_tsi = int(critical_idxs[0])
            candidates = tsub_blocks_arr[first_crit_tsi]
            
            best_replacement = None
            best_new_gain = -1
            for bi2 in candidates:
                if bi2 == bi:
                    continue
                idxs_bi2 = block_tsubs_arr[bi2]
                # Check if bi2 covers all critical t-subsets
                if np.all(np.isin(critical_idxs, idxs_bi2)):
                    # Compute net change in coverage (for potential future use)
                    # For now, just accept any valid replacement
                    best_replacement = bi2
                    break
            
            if best_replacement is not None:
                cov[idxs_bi] -= 1
                cov[block_tsubs_arr[best_replacement]] += 1
                blocks[i] = best_replacement
                improved = True
                found_swap = True
                break
        
        if improved:
            continue
        
        # Phase 3: Try 2-block removal + 2-block addition
        # Remove two blocks, see if we can cover all newly uncovered t-subsets with two other blocks
        if len(blocks) >= 2:
            found_2swap = False
            for _ in range(20):
                if time.time() - start_time > time_limit:
                    break
                
                i1 = rng.randrange(len(blocks))
                i2 = rng.randrange(len(blocks))
                if i1 == i2:
                    continue
                
                bi1 = blocks[i1]
                bi2 = blocks[i2]
                idxs1 = block_tsubs_arr[bi1]
                idxs2 = block_tsubs_arr[bi2]
                
                # t-subsets that would become uncovered if we remove both
                # These are t-subsets covered by bi1 or bi2 but not by any other block
                temp_cov = cov.copy()
                temp_cov[idxs1] -= 1
                temp_cov[idxs2] -= 1
                
                newly_uncovered = (temp_cov == 0)
                if not newly_uncovered.any():
                    # Both blocks are redundant together
                    cov[idxs1] -= 1
                    cov[idxs2] -= 1
                    # Remove both (remove higher index first)
                    if i1 > i2:
                        blocks.pop(i1)
                        blocks.pop(i2)
                    else:
                        blocks.pop(i2)
                        blocks.pop(i1)
                    improved = True
                    found_2swap = True
                    break
                
                # Find two blocks that cover all newly uncovered t-subsets
                # This is expensive; try a simple approach
                # Pick first newly uncovered t-subset, find candidates
                nu_idxs = np.where(newly_uncovered)[0]
                if len(nu_idxs) > 10:  # Too many to cover with 2 blocks likely
                    continue
                
                first_nu_tsi = int(nu_idxs[0])
                cand1 = tsub_blocks_arr[first_nu_tsi]
                
                if len(cand1) > 50:
                    continue
                
                for c1 in cand1[:20]:
                    if c1 == bi1 or c1 == bi2:
                        continue
                    idxs_c1 = block_tsubs_arr[c1]
                    # What's still uncovered after adding c1?
                    remaining = newly_uncovered.copy()
                    remaining[idxs_c1] = False
                    if not remaining.any():
                        # c1 alone covers all - but we need to add 2 blocks total
                        # Actually, we're replacing 2 blocks with 2, so we need exactly 2
                        # If c1 covers all, we still need a second block (could be any)
                        # This gets complex; skip for now
                        continue
                    
                    # Find second block
                    second_nu_tsi = int(np.where(remaining)[0][0])
                    cand2 = tsub_blocks_arr[second_nu_tsi]
                    
                    for c2 in cand2[:20]:
                        if c2 == bi1 or c2 == bi2 or c2 == c1:
                            continue
                        idxs_c2 = block_tsubs_arr[c2]
                        # Check if c1 and c2 together cover all newly uncovered
                        all_covered = newly_uncovered.copy()
                        all_covered[idxs_c1] = False
                        all_covered[idxs_c2] = False
                        if not all_covered.any():
                            # Valid 2-swap!
                            cov[idxs1] -= 1
                            cov[idxs2] -= 1
                            cov[idxs_c1] += 1
                            cov[idxs_c2] += 1
                            # Replace in blocks list
                            blocks[i1] = c1
                            blocks[i2] = c2
                            improved = True
                            found_2swap = True
                            break
                    if found_2swap:
                        break
            
            if improved:
                continue
        
        # No improvement found in this round
        break
    
    return blocks


def construct(v, k, t):
    """Construct a covering design C(v,k,t) using randomized greedy + local search."""
    deadline = time.time() + 145.0  # Safety margin before 150s kill
    
    # Precompute
    precomp = _precompute(v, k, t)
    
    # Multiple randomized greedy restarts with different strategies
    best_blocks = None
    best_count = float('inf')
    
    strategies = [
        (0.0, None),    # Pure global greedy
        (0.3, 42),      # Randomized with alpha=0.3
        (0.5, 123),     # Randomized with alpha=0.5
        (0.7, 456),     # Randomized with alpha=0.7
        (0.0, 789),     # Pure global greedy, different seed
    ]
    
    for alpha, seed in strategies:
        if time.time() > deadline - 60:
            break
        
        if alpha == 0.0:
            rng = random.Random(seed if seed else 0)
            blocks = _greedy_cover(v, k, t, rng, precomp)
        else:
            rng = random.Random(seed if seed else 0)
            blocks = _greedy_cover_randomized(v, k, t, rng, precomp, alpha=alpha)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
    
    # Additional pure greedy restarts with different seeds
    for r in range(5):
        if time.time() > deadline - 60:
            break
        rng = random.Random(1000 + r * 137)
        blocks = _greedy_cover(v, k, t, rng, precomp)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
    
    # Local search on the best solution
    if best_blocks is not None:
        if time.time() < deadline - 5:
            rng = random.Random(42)
            remaining = min(110.0, deadline - time.time() - 3.0)
            best_blocks = _local_search(v, k, t, best_blocks, rng, precomp, time_limit=remaining)
    
    # Final validation and cleanup
    if best_blocks is None:
        # Fallback: simple greedy
        rng = random.Random(0)
        best_blocks = _greedy_cover(v, k, t, rng, precomp)
    
    # Convert block indices to actual blocks
    tsubs, tsub_idx, blocks_list, block_idx, block_tsubs_arr, tsub_blocks_arr, n_tsubs, n_blocks = precomp
    
    # Ensure blocks are distinct and valid
    seen = set()
    final_blocks = []
    for bi in best_blocks:
        b = blocks_list[bi]
        key = tuple(sorted(b))
        if key not in seen and len(set(b)) == k and all(0 <= x < v for x in b):
            seen.add(key)
            final_blocks.append(list(b))
    
    return final_blocks
# EVOLVE-BLOCK-END
