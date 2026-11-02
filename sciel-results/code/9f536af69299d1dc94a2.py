import numpy as np
import itertools
import time
import random
from collections import Counter

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0
    
    points = range(v)
    all_ts = list(itertools.combinations(points, t))
    n_ts = len(all_ts)
    idx_map = {comb: i for i, comb in enumerate(all_ts)}
    
    all_ks = list(itertools.combinations(points, k))
    n_ks = len(all_ks)
    
    def nCr(n, r):
        if r > n: return 0
        if r == 0: return 1
        res = 1
        for i in range(r):
            res *= (n - i)
            res //= (i + 1)
        return res
            
    t_count = nCr(k, t)
    
    # Precompute: for each block, which t-subsets it covers
    # Use int32 for indices
    block_t_indices = np.zeros((n_ks, t_count), dtype=np.int32)
    for i, block in enumerate(all_ks):
        subs = itertools.combinations(block, t)
        for j, sub in enumerate(subs):
            block_t_indices[i, j] = idx_map[sub]

    # Binary matrix: block_t_mat[i, j] = 1 iff block i covers t-subset j
    block_t_mat = np.zeros((n_ks, n_ts), dtype=np.int8)
    for i in range(n_ks):
        block_t_mat[i, block_t_indices[i, :t_count]] = 1

    # Inverse index: for each t-subset, which blocks cover it
    triple_to_blocks = [[] for _ in range(n_ts)]
    for i in range(n_ks):
        for j in range(t_count):
            triple_to_blocks[int(block_t_indices[i, j])].append(i)
    triple_to_blocks = [np.array(lst, dtype=np.int32) if lst else np.array([], dtype=np.int32) for lst in triple_to_blocks]
            
    best_solution = None
    best_score = float('inf')
    
    def aggressive_deletion(current_blocks, coverage_count):
        """Try harder to remove blocks, including blocks where most subsets are over-covered."""
        improved = True
        while improved:
            if time.time() - start_time > time_limit - 3.0:
                break
            improved = False
            new_blocks = []
            
            for b_idx in current_blocks:
                # FIX: Only look at valid t_count entries
                subs_in_block = block_t_indices[b_idx, :t_count]
                # Count how many of this block's t-subsets are covered by other blocks
                other_covers = coverage_count[subs_in_block] - 1
                if np.all(other_covers > 0):
                    # Safe to remove
                    improved = True
                    coverage_count[subs_in_block] -= 1
                else:
                    new_blocks.append(b_idx)
                    
            current_blocks = new_blocks
            
            # Second pass: try removing blocks where >= 90% of subsets are over-covered
            # and verify the few uncovered ones are covered elsewhere
            improved = False
            new_blocks = []
            for b_idx in current_blocks:
                subs_in_block = block_t_indices[b_idx, :t_count]
                other_covers = coverage_count[subs_in_block] - 1
                # If at least 95% are covered by others, try removing
                if np.mean(other_covers > 0) >= 0.95:
                    uncovered_mask = other_covers <= 0
                    if np.any(uncovered_mask):
                        uncovered_subs = subs_in_block[uncovered_mask]
                        # Fast check using precomputed inverse index
                        can_remove = True
                        for unc_sub in uncovered_subs:
                            unc_sub = int(unc_sub)
                            # coverage_count[unc_sub] == 1 means only b_idx covers it
                            if coverage_count[unc_sub] == 1:
                                can_remove = False
                                break
                        if can_remove:
                            improved = True
                            coverage_count[subs_in_block] -= 1
                            continue
                    else:
                        improved = True
                        coverage_count[subs_in_block] -= 1
                        continue
                new_blocks.append(b_idx)
            current_blocks = new_blocks
            
        return current_blocks

    def local_search_swap(current_blocks, coverage_count):
        """
        Try to improve the cover by swapping blocks.
        If we can't remove any block, try to swap one out and one in,
        which might allow removing another block.
        """
        if not current_blocks:
            return current_blocks, coverage_count
            
        improved = True
        while improved and (time.time() - start_time < time_limit - 5.0):
            improved = False
            n_blocks = len(current_blocks)
            if n_blocks == 0: break
            
            # Try swapping each block with a random few others
            # To be efficient, we only try a subset of swaps per iteration
            # We look for a swap that reduces the number of blocks (by enabling a removal)
            
            # First, identify blocks that are "critical" (cannot be removed)
            # We already did this in aggressive_deletion, so all remaining blocks are critical.
            
            # Strategy: Pick a block B_out. Find a block B_in not in set.
            # Check if removing B_out and adding B_in allows removing some B_rem.
            
            # To keep it fast, we try limited random swaps.
            attempts = 0
            max_attempts = 100
            while attempts < max_attempts and not improved:
                attempts += 1
                if time.time() - start_time > time_limit - 5.0:
                    break
                
                out_idx = random.randint(0, n_blocks - 1)
                b_out = current_blocks[out_idx]
                
                # Pick a random candidate block to add
                b_in = random.randint(0, n_ks - 1)
                if b_in == b_out:
                    continue
                if b_in in current_blocks:
                    continue
                    
                # Simulate swap: remove b_out, add b_in
                # Update coverage
                subs_out = block_t_indices[b_out, :t_count]
                subs_in = block_t_indices[b_in, :t_count]
                
                coverage_count[subs_out] -= 1
                coverage_count[subs_in] += 1
                
                # Check if any block in the new set (excluding b_in) can be removed
                # We check if b_in is redundant (unlikely immediately) or if some other block is now redundant.
                # Actually, we want to see if the NEW set allows a deletion.
                # So we run a quick deletion check on the new set.
                
                # To be very fast, we just check if b_in covers something that allows removing b_out? No, b_out is gone.
                # We check if any block in (current_blocks \ {b_out}) + {b_in} is redundant.
                
                # This is expensive to check for all.
                # Heuristic: Check if b_in is redundant in the new context?
                # Or check if the "gap" left by b_out is covered by b_in well enough that another block becomes redundant.
                
                # Let's just check if we can remove ANY block from the new set.
                # We prioritize checking blocks that were previously "close" to being removable.
                
                # For simplicity and speed, we just check if b_in itself is redundant (i.e., its subs are covered by others).
                # If b_in is redundant, we just remove it and we are back to original (no gain).
                # We want to find a case where swapping allows removing a DIFFERENT block.
                
                # Let's try to find a block B_rem in the new set that is redundant.
                # We check a random subset of blocks in the new set.
                
                found_redundant = -1
                check_indices = random.sample(range(len(current_blocks)), min(10, len(current_blocks)))
                # Note: indices in current_blocks are positions, not block IDs.
                # We need to map positions to block IDs.
                
                # Actually, let's just check if b_in covers the "critical" parts of b_out.
                # If b_in covers all the critical triples of b_out, then b_out was redundant? No, b_out is removed.
                
                # Let's revert and try a simpler heuristic:
                # If the swap doesn't immediately allow a removal, revert.
                
                # Check if any block in the new set is redundant.
                # We only check a few.
                potential_redundant = -1
                for pos in check_indices:
                    if pos == out_idx: continue
                    b_check = current_blocks[pos]
                    if b_check == b_in: continue # b_in is new, check if it's redundant
                    subs_check = block_t_indices[b_check, :t_count]
                    other_covers_check = coverage_count[subs_check] - 1
                    if np.all(other_covers_check > 0):
                        potential_redundant = pos
                        break
                
                if potential_redundant != -1:
                    # We found a redundant block in the new set.
                    # Apply the swap and the removal.
                    b_rem = current_blocks[potential_redundant]
                    subs_rem = block_t_indices[b_rem, :t_count]
                    coverage_count[subs_rem] -= 1
                    
                    # Update current_blocks
                    # Remove b_out and b_rem, add b_in
                    new_blocks = []
                    for b in current_blocks:
                        if b != b_out and b != b_rem:
                            new_blocks.append(b)
                    new_blocks.append(b_in)
                    
                    current_blocks = new_blocks
                    improved = True
                    # Note: coverage_count is already updated for b_out, b_in, b_rem
                else:
                    # Revert
                    coverage_count[subs_out] += 1
                    coverage_count[subs_in] -= 1
        
        return current_blocks, coverage_count

    def run_greedy(seed):
        random.seed(seed)
        np.random.seed(seed)
        
        covered = np.zeros(n_ts, dtype=bool)
        blocks_indices = []
        remaining = n_ts
        
        # Track how many times each t-subset has been covered (for weighted scoring)
        cover_freq = np.zeros(n_ts, dtype=np.int32)
        
        while remaining > 0:
            if time.time() - start_time > time_limit - 5.0:
                return None, None

            # Compute gain via single vectorized mat-vec: how many uncovered triples each block covers
            gains = block_t_mat @ (~covered).astype(np.int8)

            max_gain = int(gains.max())
            if max_gain == 0:
                break

            # Among blocks with max gain, use weighted score based on rarity
            max_gain_blocks = np.where(gains == max_gain)[0]

            if len(max_gain_blocks) > 40:
                candidates = max_gain_blocks[random.sample(range(len(max_gain_blocks)), min(40, len(max_gain_blocks)))]
            else:
                candidates = max_gain_blocks

            # Score candidates: gain + bonus for covering rare subsets
            best_score_candidate = -1.0
            best_candidate = int(candidates[0])
            for cand in candidates:
                subs = block_t_indices[cand, :t_count]
                newly_covered = ~covered[subs]
                if np.any(newly_covered):
                    bonus = float(np.sum(1.0 / (1.0 + cover_freq[subs[newly_covered]])))
                else:
                    bonus = 0.0
                score_cand = max_gain + bonus * 0.1
                if score_cand > best_score_candidate:
                    best_score_candidate = score_cand
                    best_candidate = int(cand)

            blocks_indices.append(best_candidate)

            subs_to_cover = block_t_indices[best_candidate, :t_count]
            newly_covered = ~covered[subs_to_cover]
            new_covers = int(np.sum(newly_covered))
            remaining -= new_covers
            cover_freq[subs_to_cover] += 1
            covered[subs_to_cover] = True

        current_blocks = list(blocks_indices)
        
        # Build coverage count
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in current_blocks:
            coverage_count[block_t_indices[b_idx, :t_count]] += 1
        
        # Aggressive deletion
        current_blocks = aggressive_deletion(current_blocks, coverage_count)
        
        # Rebuild coverage count after deletion
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in current_blocks:
            coverage_count[block_t_indices[b_idx, :t_count]] += 1
        
        # Local Search with Swaps
        if time.time() - start_time < time_limit - 10.0:
            current_blocks, coverage_count = local_search_swap(current_blocks, coverage_count)
            
        # Final deletion pass
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in current_blocks:
            coverage_count[block_t_indices[b_idx, :t_count]] += 1
        current_blocks = aggressive_deletion(current_blocks, coverage_count)
        
        return current_blocks, len(current_blocks)
    
    # Phase 1: Run multiple greedy restarts
    restart_count = 0
    while time.time() - start_time < time_limit - 20.0:
        seed = restart_count
        blocks, score = run_greedy(seed)
        
        if blocks is None:
            break
            
        if score < best_score:
            best_score = score
            best_solution = blocks
            
        restart_count += 1
    
    # Phase 2: Final greedy restarts with remaining time
    if time.time() - start_time < time_limit - 5.0:
        while time.time() - start_time < time_limit - 2.0:
            seed = restart_count + 1000
            blocks, score = run_greedy(seed)
            
            if blocks is None:
                break
                
            if score < best_score:
                best_score = score
                best_solution = blocks
                
            restart_count += 1
    
    if best_solution is None:
        blocks, score = run_greedy(0)
        if blocks is not None:
            best_solution = blocks
            best_score = score
        else:
            return []

    final_blocks = [list(all_ks[i]) for i in best_solution]
    
    return final_blocks
