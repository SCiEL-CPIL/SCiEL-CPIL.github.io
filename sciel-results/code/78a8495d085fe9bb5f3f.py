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
    
    # Build reverse index: for each t-subset, which blocks cover it
    # This enables fast swap checks
    t_to_blocks = [set() for _ in range(n_ts)]
    for i in range(n_ks):
        for j in range(t_count):
            t_to_blocks[block_t_indices[i, j]].add(i)
    
    best_solution = None
    best_score = float('inf')
    
    def fast_deletion(current_blocks, coverage_count):
        """Remove all blocks whose t-subsets are all covered by other blocks."""
        if time.time() - start_time > time_limit - 3.0:
            return current_blocks
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
    
    def swap_local_search(current_blocks, coverage_count):
        """Try to replace pairs of blocks with a single block that covers all their unique t-subsets."""
        if time.time() - start_time > time_limit - 3.0:
            return current_blocks, coverage_count
        
        blocks_set = set(current_blocks)
        improved = True
        while improved:
            if time.time() - start_time > time_limit - 3.0:
                break
            improved = False
            
            # For each pair of blocks, check if a single block can replace them
            # Only try pairs where the union of their unique t-subsets is small
            n = len(current_blocks)
            tried = set()
            for i in range(n):
                if time.time() - start_time > time_limit - 3.0:
                    return current_blocks, coverage_count
                b1 = current_blocks[i]
                for j in range(i + 1, n):
                    b2 = current_blocks[j]
                    pair_key = (min(b1, b2), max(b1, b2))
                    if pair_key in tried:
                        continue
                    tried.add(pair_key)
                    
                    # Find t-subsets uniquely covered by b1 or b2
                    subs1 = block_t_indices[b1]
                    subs2 = block_t_indices[b2]
                    
                    # Unique to b1: covered by b1 but not by any other block except b2
                    # Unique to b2: covered by b2 but not by any other block except b1
                    # A replacement block must cover all of these
                    unique_to_1 = []
                    unique_to_2 = []
                    for sub in subs1:
                        if coverage_count[sub] == 1:
                            unique_to_1.append(sub)
                    for sub in subs2:
                        if coverage_count[sub] == 1:
                            unique_to_2.append(sub)
                    
                    if not unique_to_1 and not unique_to_2:
                        # Both blocks are redundant
                        improved = True
                        coverage_count[subs1] -= 1
                        coverage_count[subs2] -= 1
                        current_blocks = [b for b in current_blocks if b not in (b1, b2)]
                        blocks_set.discard(b1)
                        blocks_set.discard(b2)
                        break
                    elif unique_to_1 and not unique_to_2:
                        # Need a block covering all of unique_to_1
                        # Find candidate blocks that cover all unique_to_1
                        candidates = t_to_blocks[unique_to_1[0]]
                        for cand in candidates:
                            if cand in blocks_set and cand != b1 and cand != b2:
                                # Check if cand covers all unique_to_1
                                cand_subs = set(block_t_indices[cand].tolist())
                                if all(s in cand_subs for s in unique_to_1):
                                    improved = True
                                    coverage_count[subs1] -= 1
                                    coverage_count[subs2] -= 1
                                    coverage_count[block_t_indices[cand]] += 1
                                    current_blocks = [b for b in current_blocks if b not in (b1, b2)]
                                    blocks_set.discard(b1)
                                    blocks_set.discard(b2)
                                    blocks_set.add(cand)
                                    break
                        if improved:
                            break
                    elif unique_to_2 and not unique_to_1:
                        # Need a block covering all of unique_to_2
                        candidates = t_to_blocks[unique_to_2[0]]
                        for cand in candidates:
                            if cand in blocks_set and cand != b1 and cand != b2:
                                cand_subs = set(block_t_indices[cand].tolist())
                                if all(s in cand_subs for s in unique_to_2):
                                    improved = True
                                    coverage_count[subs1] -= 1
                                    coverage_count[subs2] -= 1
                                    coverage_count[block_t_indices[cand]] += 1
                                    current_blocks = [b for b in current_blocks if b not in (b1, b2)]
                                    blocks_set.discard(b1)
                                    blocks_set.discard(b2)
                                    blocks_set.add(cand)
                                    break
                        if improved:
                            break
                    else:
                        # Both have unique subsets - skip (too complex)
                        pass
            if improved:
                # Rebuild coverage_count after modification
                coverage_count = np.zeros(n_ts, dtype=np.int32)
                for b_idx in current_blocks:
                    coverage_count[block_t_indices[b_idx]] += 1
                blocks_set = set(current_blocks)
        return current_blocks, coverage_count
    
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
            
            # Compute gain: number of new t-subsets covered
            covered_counts = covered[block_t_indices].sum(axis=1)
            gains = t_count - covered_counts
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            # Among blocks with max gain, use weighted score based on rarity
            max_gain_blocks = np.where(gains == max_gain)[0]
            
            if len(max_gain_blocks) > 50:
                candidates = max_gain_blocks[random.sample(range(len(max_gain_blocks)), min(50, len(max_gain_blocks)))]
            else:
                candidates = max_gain_blocks
            
            # Score candidates: gain + bonus for covering rare subsets
            best_score_candidate = -1
            best_candidate = candidates[0]
            for cand in candidates:
                subs = block_t_indices[cand]
                newly_covered = ~covered[subs]
                if np.any(newly_covered):
                    bonus = np.sum(1.0 / (1.0 + cover_freq[subs[newly_covered]]))
                else:
                    bonus = 0.0
                score_cand = max_gain + bonus * 0.1
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
        
        # Fast deletion
        current_blocks = fast_deletion(current_blocks, coverage_count)
        
        # Rebuild coverage count after deletion
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in current_blocks:
            coverage_count[block_t_indices[b_idx]] += 1
        
        # Swap local search
        current_blocks, coverage_count = swap_local_search(current_blocks, coverage_count)
        
        # Final deletion pass
        current_blocks = fast_deletion(current_blocks, coverage_count)
        
        return current_blocks, len(current_blocks)

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

    final_blocks = [list(all_ks[i]) for i in best_solution]
    
    return final_blocks
