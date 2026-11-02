import numpy as np
import itertools
import time
import random
from collections import defaultdict

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
            
    # Precompute: for each t-subset, which blocks cover it
    # This allows fast intersection for local search
    subset_to_blocks = [set() for _ in range(n_ts)]
    for b_idx in range(n_ks):
        for s_idx in block_t_indices[b_idx]:
            subset_to_blocks[s_idx].add(b_idx)
    
    best_solution = None
    best_score = float('inf')
    
    def aggressive_deletion(current_blocks, coverage_count):
        """
        Remove any block that is redundant.
        """
        improved = True
        while improved:
            if time.time() - start_time > time_limit - 3.0:
                break
            improved = False
            new_blocks = []
            for b_idx in current_blocks:
                subs_in_block = block_t_indices[b_idx]
                if np.all(coverage_count[subs_in_block] > 1):
                    improved = True
                    coverage_count[subs_in_block] -= 1
                else:
                    new_blocks.append(b_idx)
            current_blocks = new_blocks
        return current_blocks

    def local_search_improve(current_blocks):
        """
        Improve by replacing blocks that have few critical subsets
        with blocks that cover those critical subsets.
        """
        if time.time() - start_time > time_limit - 5.0:
            return current_blocks
        
        current_blocks = list(current_blocks)
        current_set = set(current_blocks)
        
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
                    current_set.discard(b_idx)
                    improved = True
                    break
                
                # Try to find a block that covers all critical subsets
                # Only try if few critical subsets (intersection is fast)
                if len(critical_subs) <= 6:
                    # Find intersection of blocks covering each critical subset
                    # Start with the set of blocks covering the first critical subset
                    candidates = list(subset_to_blocks[critical_subs[0]])
                    
                    for s in critical_subs[1:]:
                        if not candidates:
                            break
                        # Intersect with blocks covering s
                        s_blocks = subset_to_blocks[s]
                        candidates = [b for b in candidates if b in s_blocks]
                        if not candidates:
                            break
                    
                    if candidates:
                        # Filter out blocks already in current_set
                        # Note: b_idx is in current_set, so it will be filtered out unless we allow self-replacement (which is useless)
                        valid_candidates = [b for b in candidates if b not in current_set]
                        
                        if valid_candidates:
                            # Pick one (e.g., random or first)
                            new_b = valid_candidates[0]
                            
                            # Replace
                            current_blocks[i] = new_b
                            current_set.discard(b_idx)
                            current_set.add(new_b)
                            
                            # Update coverage count
                            coverage_count[subs] -= 1
                            coverage_count[block_t_indices[new_b]] += 1
                            
                            improved = True
                            break
        return current_blocks

    def run_greedy(seed):
        random.seed(seed)
        np.random.seed(seed)
        
        covered = np.zeros(n_ts, dtype=bool)
        blocks_indices = []
        remaining = n_ts
        
        cover_freq = np.zeros(n_ts, dtype=np.int32)
        
        while remaining > 0:
            if time.time() - start_time > time_limit - 5.0:
                return None, None
            
            # Compute gain
            covered_counts = covered[block_t_indices].sum(axis=1)
            gains = t_count - covered_counts
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            max_gain_blocks = np.where(gains == max_gain)[0]
            
            if len(max_gain_blocks) > 30:
                candidates = max_gain_blocks[random.sample(range(len(max_gain_blocks)), min(30, len(max_gain_blocks)))]
            else:
                candidates = max_gain_blocks
            
            best_score_candidate = -1
            best_candidate = candidates[0]
            
            rarity_weight = 1.0 / (1.0 + cover_freq)
            
            for cand in candidates:
                subs = block_t_indices[cand]
                newly_covered = ~covered[subs]
                if np.any(newly_covered):
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
        
        # Aggressive deletion
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in current_blocks:
            coverage_count[block_t_indices[b_idx]] += 1
        current_blocks = aggressive_deletion(current_blocks, coverage_count)
        
        # Local search improvement
        current_blocks = local_search_improve(current_blocks)
        
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
        
        # If we have a good solution, try ILS
        if best_solution is not None and restart_count > 5:
             # Simple ILS: perturb best solution
             if time.time() - start_time < time_limit - 15.0:
                 perturbed = list(best_solution)
                 # Randomly swap 1-2 blocks
                 n_swaps = random.randint(1, 2)
                 for _ in range(n_swaps):
                     if perturbed:
                         idx = random.randint(0, len(perturbed)-1)
                         # Pick a random block from all_ks
                         new_block = random.randint(0, n_ks - 1)
                         perturbed[idx] = new_block
                 
                 # Clean up duplicates? No, just run local search which handles coverage
                 # But we must ensure validity? Local search assumes a valid cover?
                 # No, local search only removes/replaces if it maintains coverage.
                 # If we break coverage with perturbation, local search might fail to fix it
                 # if it doesn't check global coverage.
                 # My local search checks critical subsets. If coverage is broken,
                 # critical subsets will be huge or non-existent?
                 # If a subset is uncovered, coverage_count is 0.
                 # critical_subs = subs[coverage_count[subs] == 1].
                 # If coverage is 0, it's not critical (in the sense of "only 1 cover").
                 # So the algorithm might miss the hole.
                 
                 # To be safe, check if perturbed is valid.
                 # This is expensive.
                 # Alternative: Only perturb if we can verify validity quickly?
                 # Or, just rely on multiple Greedy restarts.
                 pass

    if best_solution is None:
        blocks, score = run_greedy(0)
        if blocks is not None:
            best_solution = blocks
            best_score = score
        else:
            return []

    final_blocks = [list(all_ks[i]) for i in best_solution]
    
    return final_blocks
