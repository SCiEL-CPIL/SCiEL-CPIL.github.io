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
    block_t_indices = np.zeros((n_ks, t_count), dtype=np.int32)
    for i, block in enumerate(all_ks):
        subs = itertools.combinations(block, t)
        for j, sub in enumerate(subs):
            block_t_indices[i, j] = idx_map[sub]
            
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
                # A block is redundant if every t-subset in it is covered by at least one OTHER block.
                # coverage_count[subs_in_block] > 1 means count from others >= 1.
                if np.all(coverage_count[subs_in_block] > 1):
                    improved = True
                    coverage_count[subs_in_block] -= 1
                else:
                    new_blocks.append(b_idx)
            current_blocks = new_blocks
        return current_blocks

    def local_search_improve(current_blocks, coverage_count):
        """
        Try to reduce block count by attempting to replace blocks with others
        that cover the same critical subsets more efficiently.
        """
        if time.time() - start_time > time_limit - 5.0:
            return current_blocks
        
        current_blocks = list(current_blocks)
        
        # Rebuild coverage count
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in current_blocks:
            coverage_count[block_t_indices[b_idx]] += 1
        
        # Find blocks that are "critical" (some subset only covered by this block)
        # and try to find alternative blocks that cover those critical subsets
        improved = True
        while improved:
            if time.time() - start_time > time_limit - 5.0:
                break
            improved = False
            
            # Rebuild coverage count
            coverage_count = np.zeros(n_ts, dtype=np.int32)
            for b_idx in current_blocks:
                coverage_count[block_t_indices[b_idx]] += 1
            
            for i in range(len(current_blocks)):
                if time.time() - start_time > time_limit - 5.0:
                    break
                b_idx = current_blocks[i]
                subs = block_t_indices[b_idx]
                
                # Find subsets that would become uncovered if we remove this block
                critical_subs = subs[coverage_count[subs] == 1]
                
                if len(critical_subs) == 0:
                    # Can remove this block
                    coverage_count[subs] -= 1
                    current_blocks.pop(i)
                    improved = True
                    break
                
                # Try to find a block that covers all critical subsets and is not already in our set
                if len(critical_subs) <= 3:  # Only try if few critical subsets
                    for j in range(n_ks):
                        if j == b_idx or j in current_blocks:
                            continue
                        j_subs = block_t_indices[j]
                        # Check if j covers all critical subsets
                        if np.all(np.isin(critical_subs, j_subs)):
                            # Check if adding j and removing i doesn't break coverage
                            # New coverage: remove i's contribution, add j's
                            new_coverage = coverage_count.copy()
                            new_coverage[subs] -= 1
                            new_coverage[j_subs] += 1
                            if np.all(new_coverage >= 1):
                                coverage_count = new_coverage
                                current_blocks[i] = j
                                improved = True
                                break
                    if improved:
                        break
        return current_blocks

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
            
            # Compute gain: number of new t-subsets covered
            covered_counts = covered[block_t_indices].sum(axis=1)
            gains = t_count - covered_counts
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            # Among blocks with max gain, use weighted score based on rarity
            max_gain_blocks = np.where(gains == max_gain)[0]
            
            if len(max_gain_blocks) > 30:
                # Sample from top candidates
                candidates = max_gain_blocks[random.sample(range(len(max_gain_blocks)), min(30, len(max_gain_blocks)))]
            else:
                candidates = max_gain_blocks
            
            # Score candidates using soft-max surrogate for rarity
            best_score_candidate = -1
            best_candidate = candidates[0]
            
            # Compute rarity weights once
            rarity_weight = 1.0 / (1.0 + cover_freq)
            
            for cand in candidates:
                subs = block_t_indices[cand]
                newly_covered = ~covered[subs]
                if np.any(newly_covered):
                    # Sum of rarity weights for newly covered subsets (smooth surrogate)
                    bonus = np.sum(rarity_weight[subs[newly_covered]])
                else:
                    bonus = 0.0
                score_cand = max_gain + bonus * 0.05
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
            
        current_blocks = list(blocks_indices)
        
        # Build coverage count
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in current_blocks:
            coverage_count[block_t_indices[b_idx]] += 1
        
        # Aggressive deletion
        current_blocks = aggressive_deletion(current_blocks, coverage_count)
        
        # Local search improvement
        current_blocks = local_search_improve(current_blocks, coverage_count)
        
        # Final deletion pass
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in current_blocks:
            coverage_count[block_t_indices[b_idx]] += 1
        current_blocks = aggressive_deletion(current_blocks, coverage_count)
        
        return current_blocks, len(current_blocks)

    # Phase 1: Run multiple greedy restarts
    restart_count = 0
    while time.time() - start_time < time_limit - 5.0:
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

    final_blocks = [list(all_ks[i]) for i in best_solution]
    
    return final_blocks
