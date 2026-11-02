import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    time_limit = 118.0  # Safety margin
    
    points = list(range(v))
    all_ts = list(itertools.combinations(points, t))
    n_ts = len(all_ts)
    idx_map = {comb: i for i, comb in enumerate(all_ts)}
    
    all_ks = list(itertools.combinations(points, k))
    n_ks = len(all_ks)
    
    # Precompute t-subset indices for each k-block
    # shape: (n_ks, C(k,t))
    t_count = 1
    for i in range(t):
        t_count *= (k - i)
        t_count //= (i + 1)
        
    sub_indices = np.zeros((n_ks, t_count), dtype=np.int32)
    for i, block in enumerate(all_ks):
        for j, sub in enumerate(itertools.combinations(block, t)):
            sub_indices[i, j] = idx_map[sub]
    
    # Precompute for faster operations
    # For each block, store the set of t-subsets it covers as a frozenset for quick lookup
    # Also precompute a lookup: for each t-subset, which blocks cover it
    blocks_covering = [set() for _ in range(n_ts)]
    for i in range(n_ks):
        for j in range(t_count):
            ts_idx = sub_indices[i, j]
            blocks_covering[ts_idx].add(i)
    
    best_solution = None
    best_score = float('inf')
    
    def greedy_cover(seed):
        random.seed(seed)
        np.random.seed(seed)
        
        uncovered = np.ones(n_ts, dtype=bool)
        selected_blocks = []
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        
        while uncovered.sum() > 0:
            if time.time() - start_time > time_limit - 5.0:
                return None
            
            # Vectorized gain calculation
            gains = uncovered[sub_indices].sum(axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            candidates = np.where(gains == max_gain)[0]
            best = random.choice(candidates)
            
            selected_blocks.append(best)
            subs = sub_indices[best]
            coverage_count[subs] += 1
            uncovered[subs] = False
            
        return selected_blocks

    def reduce_cover(blocks, seed=0):
        random.seed(seed)
        
        if not blocks:
            return []
            
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b in blocks:
            coverage_count[sub_indices[b]] += 1
            
        current_blocks = list(blocks)
        improved = True
        
        while improved:
            if time.time() - start_time > time_limit - 5.0:
                break
            improved = False
            
            order = list(range(len(current_blocks)))
            random.shuffle(order)
            
            new_blocks = []
            for idx in order:
                b = current_blocks[idx]
                subs = sub_indices[b]
                if np.all(coverage_count[subs] > 1):
                    coverage_count[subs] -= 1
                    improved = True
                else:
                    new_blocks.append(b)
                    
            current_blocks = new_blocks
            
        return current_blocks

    def polish(blocks, seed=0):
        random.seed(seed)
        if not blocks:
            return blocks
            
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b in blocks:
            coverage_count[sub_indices[b]] += 1
            
        current_blocks = list(blocks)
        
        # First do a reduction pass
        current_blocks = reduce_cover(current_blocks, seed + 500)
        if not current_blocks:
            return []
        
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b in current_blocks:
            coverage_count[sub_indices[b]] += 1
        
        block_set = set(current_blocks)
        
        improved = True
        max_iters = 300
        iter_count = 0
        
        while improved and iter_count < max_iters:
            if time.time() - start_time > time_limit - 5.0:
                break
            improved = False
            iter_count += 1
            
            n_blocks = len(current_blocks)
            if n_blocks == 0:
                break
            
            order = list(range(n_blocks))
            random.shuffle(order)
            
            for idx in order:
                if time.time() - start_time > time_limit - 5.0:
                    break
                    
                b = current_blocks[idx]
                
                # Temporarily remove block b
                subs_b = sub_indices[b]
                coverage_count[subs_b] -= 1
                
                # Find uncovered triples
                uncovered_mask = (coverage_count == 0)
                n_uncovered = int(np.sum(uncovered_mask))
                
                if n_uncovered == 0:
                    # Block was redundant - remove it
                    current_blocks.pop(idx)
                    block_set.discard(b)
                    improved = True
                    block_set = set(current_blocks)
                    break
                
                # Try to find a single block that covers all uncovered triples
                gains = uncovered_mask[sub_indices].sum(axis=1)
                full_cover_candidates = np.where(gains == n_uncovered)[0]
                
                valid_candidates = [c for c in full_cover_candidates if c not in block_set]
                
                if valid_candidates:
                    # Pick candidate with best "synergy" - covers most already-covered subsets
                    # to maximize future reduction potential
                    best_cand = None
                    best_score_cand = -1
                    for c in valid_candidates[:5]:  # Limit to top 5 for speed
                        cand_subs = sub_indices[c]
                        # How many of its subsets are already well-covered (count > 1)?
                        synergy = int(np.sum(coverage_count[cand_subs] > 1))
                        if synergy > best_score_cand:
                            best_score_cand = synergy
                            best_cand = c
                    if best_cand is None and valid_candidates:
                        best_cand = random.choice(valid_candidates)
                    
                    if best_cand is not None:
                        current_blocks[idx] = best_cand
                        block_set.discard(b)
                        block_set.add(best_cand)
                        coverage_count[sub_indices[best_cand]] += 1
                        improved = True
                        break
                else:
                    coverage_count[subs_b] += 1
                    continue
                    
        return current_blocks

    # Main search loop
    restart_count = 0
    while time.time() - start_time < time_limit - 10.0:
        seed = restart_count
        blocks = greedy_cover(seed)
        
        if blocks is None:
            break
            
        reduced = reduce_cover(blocks, seed + 100)
        polished = polish(reduced, seed + 200)
        
        if polished is not None and len(polished) < best_score:
            best_score = len(polished)
            best_solution = list(polished)
        
        restart_count += 1

    if best_solution is None:
        blocks = greedy_cover(0)
        if blocks:
            best_solution = reduce_cover(blocks, 1)
            best_solution = polish(best_solution, 2)
            
    if best_solution is None:
        return []

    final_blocks = [list(all_ks[i]) for i in best_solution]
    
    return final_blocks
