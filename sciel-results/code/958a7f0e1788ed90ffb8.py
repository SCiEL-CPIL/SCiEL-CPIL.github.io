import numpy as np
import itertools
import time
import random

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
    block_t_indices = np.zeros((n_ks, t_count), dtype=np.int32)
    for i, block in enumerate(all_ks):
        subs = itertools.combinations(block, t)
        for j, sub in enumerate(subs):
            block_t_indices[i, j] = idx_map[sub]
            
    # Precompute point frequencies for each block (k per block)
    # block_points[i] is the block tuple itself, but we want point indices
    # Actually all_ks[i] is the tuple of points.
    # We can compute point_load dynamically or precompute point counts per block?
    # point_counts[i, p] = 1 if p in block i else 0.
    # Size: 496k * 24 * 1 byte = ~12MB. Feasible.
    # But we only need it for the heuristic.
    # Let's just use the tuple directly for point access in the heuristic, 
    # or precompute a set? No, sets are slow.
    # Let's precompute a boolean mask for points per block?
    # Or just use the tuple. all_ks[i] is a tuple of k ints.
    
    best_solution = None
    best_score = float('inf')
    
    def aggressive_deletion(current_blocks, coverage_count):
        """
        Remove any block that is redundant (all its t-subsets are covered by other blocks).
        """
        improved = True
        while improved:
            if time.time() - start_time > time_limit - 3.0:
                break
            improved = False
            new_blocks = []
            for b_idx in current_blocks:
                subs_in_block = block_t_indices[b_idx]
                # Check if all subsets in this block have coverage > 1
                if np.all(coverage_count[subs_in_block] > 1):
                    improved = True
                    coverage_count[subs_in_block] -= 1
                else:
                    new_blocks.append(b_idx)
            current_blocks = new_blocks
        return current_blocks

    def local_search_polish(current_blocks, coverage_count, max_iterations=200):
        """
        Try to reduce block count via swap operations.
        """
        if time.time() - start_time > time_limit - 5.0:
            return current_blocks, coverage_count
        
        best_blocks = list(current_blocks)
        best_coverage = coverage_count.copy()
        best_count = len(best_blocks)
        
        for iteration in range(max_iterations):
            if time.time() - start_time > time_limit - 5.0:
                break
            
            # Try removing each block
            removed_any = False
            new_blocks = list(best_blocks)
            new_coverage = best_coverage.copy()
            
            for i in range(len(new_blocks)):
                b_idx = new_blocks[i]
                subs_in_block = block_t_indices[b_idx]
                
                # Temporarily remove this block
                new_coverage[subs_in_block] -= 1
                
                # Check if coverage is still complete
                if np.all(new_coverage > 0):
                    # Safe to remove
                    new_blocks.pop(i)
                    removed_any = True
                    break
                else:
                    # Restore
                    new_coverage[subs_in_block] += 1
            
            if removed_any:
                best_blocks = new_blocks
                best_coverage = new_coverage
                best_count -= 1
                continue
            
            # Try swap: remove one block, add another
            if len(best_blocks) > 0:
                # Calculate unique coverage for each block
                contributions = []
                for b_idx in best_blocks:
                    subs = block_t_indices[b_idx]
                    unique_count = np.sum(best_coverage[subs] == 1)
                    contributions.append(unique_count)
                
                contributions = np.array(contributions)
                min_contrib = np.min(contributions)
                low_contrib_indices = np.where(contributions == min_contrib)[0]
                
                if len(low_contrib_indices) > 0:
                    # Pick a random low-contribution block to try removing
                    remove_idx = low_contrib_indices[random.randint(0, len(low_contrib_indices) - 1)]
                    remove_b_idx = best_blocks[remove_idx]
                    remove_subs = block_t_indices[remove_b_idx]
                    
                    # Find uncovered subsets after removal
                    temp_coverage = best_coverage.copy()
                    temp_coverage[remove_subs] -= 1
                    uncovered_mask = (temp_coverage == 0)
                    uncovered_indices = np.where(uncovered_mask)[0]
                    
                    if len(uncovered_indices) == 0:
                        # Can remove without adding
                        new_blocks = [b for i, b in enumerate(best_blocks) if i != remove_idx]
                        new_coverage = temp_coverage
                        best_blocks = new_blocks
                        best_coverage = new_coverage
                        best_count -= 1
                        continue
                    
                    # Try to find a block that covers all uncovered subsets
                    # Heuristic: Sample candidates
                    sample_size = min(200, n_ks)
                    existing_set = set(best_blocks)
                    candidates = []
                    while len(candidates) < sample_size:
                        cand = random.randint(0, n_ks - 1)
                        if cand not in existing_set:
                            candidates.append(cand)
                    
                    best_candidate = -1
                    best_gain = -1
                    
                    for cand in candidates:
                        cand_subs = block_t_indices[cand]
                        if np.isin(uncovered_indices, cand_subs).all():
                            # Perfect swap
                            best_candidate = cand
                            best_gain = 1000
                            break
                        else:
                            pass
                    
                    if best_candidate >= 0:
                        # Accept the swap
                        new_blocks = [b for i, b in enumerate(best_blocks) if i != remove_idx]
                        new_blocks.append(best_candidate)
                        new_coverage = temp_coverage
                        new_coverage[block_t_indices[best_candidate]] += 1
                        best_blocks = new_blocks
                        best_coverage = new_coverage
        
        return best_blocks, best_coverage

    def run_greedy(seed):
        random.seed(seed)
        np.random.seed(seed)
        
        covered = np.zeros(n_ts, dtype=bool)
        blocks_indices = []
        remaining = n_ts
        
        # Track how many times each t-subset has been covered
        cover_freq = np.zeros(n_ts, dtype=np.int32)
        
        # NEW: Track point load (how many times each point has been used in selected blocks)
        point_load = np.zeros(v, dtype=np.int32)
        
        while remaining > 0:
            if time.time() - start_time > time_limit - 5.0:
                return None, None
            
            # Compute gain: number of new t-subsets covered
            covered_counts = covered[block_t_indices].sum(axis=1)
            gains = t_count - covered_counts
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            # Among blocks with max gain, pick one.
            max_gain_blocks = np.where(gains == max_gain)[0]
            
            # If many, sample
            if len(max_gain_blocks) > 50:
                candidates = max_gain_blocks[random.sample(range(len(max_gain_blocks)), 50)]
            else:
                candidates = max_gain_blocks
            
            # Score candidates: gain + bonus for covering rare subsets + penalty for high point load
            best_score_candidate = -1
            best_candidate = candidates[0]
            for cand in candidates:
                subs = block_t_indices[cand]
                # Bonus: sum of inverse (1 + cover_freq) for newly covered subsets
                newly_covered = ~covered[subs]
                if np.any(newly_covered):
                    freqs = cover_freq[subs[newly_covered]]
                    bonus = np.sum(1.0 / (1.0 + freqs))
                else:
                    bonus = 0.0
                
                # NEW: Penalty based on point load
                # Get the points in this block
                block_points = all_ks[cand]
                # Sum of point loads for these points
                # point_load is an array of size v
                # We need the sum of point_load[p] for p in block_points
                # Since k=7, this is a small loop
                load_penalty = 0
                for p in block_points:
                    load_penalty += point_load[p]
                
                # Normalize penalty
                # Typical point load might be ~10-20 in a good solution
                # We want to penalize heavily if a point is already used a lot
                # Use a soft penalty: load_penalty / (v/k * 2) or similar
                # Or simply: penalty = load_penalty * 0.5
                
                score_cand = max_gain + bonus * 0.1 - load_penalty * 0.5
                
                if score_cand > best_score_candidate:
                    best_score_candidate = score_cand
                    best_candidate = cand
            
            blocks_indices.append(best_candidate)
            
            subs_to_cover = block_t_indices[best_candidate]
            newly_covered = ~covered[subs_to_cover]
            new_covers = np.sum(newly_covered)
            remaining -= new_covers
            cover_freq[subs_to_cover] += 1
            covered[subs_to_cover] = True
            
            # Update point load
            for p in all_ks[best_candidate]:
                point_load[p] += 1
            
        current_blocks = list(blocks_indices)
        
        # Build coverage count
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in current_blocks:
            coverage_count[block_t_indices[b_idx]] += 1
        
        # Aggressive deletion
        current_blocks = aggressive_deletion(current_blocks, coverage_count)
        
        # Rebuild coverage count after deletion
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in current_blocks:
            coverage_count[block_t_indices[b_idx]] += 1
        
        # Local search polish
        current_blocks, coverage_count = local_search_polish(current_blocks, coverage_count)
        
        # Final deletion pass
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in current_blocks:
            coverage_count[block_t_indices[b_idx]] += 1
        current_blocks = aggressive_deletion(current_blocks, coverage_count)
        
        return current_blocks, len(current_blocks)

    # Phase 1: Run multiple greedy restarts
    restart_count = 0
    while time.time() - start_time < time_limit - 10.0:
        seed = restart_count
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
    
    # Phase 2: Additional polish on best solution
    if time.time() - start_time < time_limit - 5.0:
        # Rebuild coverage count for best solution
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in best_solution:
            coverage_count[block_t_indices[b_idx]] += 1
        
        # Run local search with more iterations
        polished_blocks, polished_coverage = local_search_polish(
            list(best_solution), coverage_count, max_iterations=500
        )
        
        if len(polished_blocks) < best_score:
            best_solution = polished_blocks
            best_score = len(polished_blocks)
        
        # Final aggressive deletion
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in best_solution:
            coverage_count[block_t_indices[b_idx]] += 1
        best_solution = aggressive_deletion(list(best_solution), coverage_count)
    
    final_blocks = [list(all_ks[i]) for i in best_solution]
    
    return final_blocks
