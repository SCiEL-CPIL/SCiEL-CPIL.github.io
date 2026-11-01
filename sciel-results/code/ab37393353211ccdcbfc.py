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
        subs = itertools.combinations(block, t)
        for j, sub in enumerate(subs):
            sub_indices[i, j] = idx_map[sub]
            
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
            # uncovered[sub_indices] is a boolean array of shape (n_ks, t_count)
            # Summing along axis 1 gives the number of newly covered subsets for each block
            gains = uncovered[sub_indices].sum(axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break # Should not happen if uncovered > 0 and we have valid blocks
            
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
            
        # Build coverage count
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b in blocks:
            coverage_count[sub_indices[b]] += 1
            
        current_blocks = list(blocks)
        improved = True
        
        while improved:
            if time.time() - start_time > time_limit - 5.0:
                break
            improved = False
            
            # Calculate min coverage for each block's subsets
            # This helps prioritize blocks that are "redundant"
            # min_cov[b] = min(coverage_count[sub_indices[b]])
            # If min_cov[b] > 1, it is safe to remove.
            
            # We can just iterate and check. To be faster, we can sort.
            # But sorting every iteration is O(N log N). N is ~70. It's fine.
            
            # Calculate safety score: how "redundant" is the block?
            # A block is redundant if all its subsets are covered by others.
            # Let's just try to remove blocks in a random order or sorted by "uniqueness".
            
            # Heuristic: Remove blocks that cover the most "common" subsets first?
            # Or just random shuffle to avoid bias.
            
            order = list(range(len(current_blocks)))
            random.shuffle(order)
            
            new_blocks = []
            for idx in order:
                b = current_blocks[idx]
                subs = sub_indices[b]
                # Check if all subsets in this block are covered by other blocks
                # i.e., coverage_count[subs] > 1
                if np.all(coverage_count[subs] > 1):
                    # Safe to remove
                    coverage_count[subs] -= 1
                    improved = True
                else:
                    new_blocks.append(b)
                    
            current_blocks = new_blocks
            
        return current_blocks

    def polish(blocks, seed=0):
        """
        Try to improve the solution by swapping blocks and removing redundant ones.
        Hill climbing with swap moves.
        """
        random.seed(seed)
        if not blocks:
            return blocks
            
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b in blocks:
            coverage_count[sub_indices[b]] += 1
            
        current_blocks = list(blocks)
        n_blocks = len(current_blocks)
        
        # First do a reduction pass
        current_blocks = reduce_cover(current_blocks, seed + 500)
        if not current_blocks:
            return []
        
        # Rebuild coverage count after reduction
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b in current_blocks:
            coverage_count[sub_indices[b]] += 1
        
        n_blocks = len(current_blocks)
        block_set = set(current_blocks)
        
        improved = True
        max_iters = 200
        iter_count = 0
        
        while improved and iter_count < max_iters:
            if time.time() - start_time > time_limit - 5.0:
                break
            improved = False
            iter_count += 1
            
            n_blocks = len(current_blocks)
            if n_blocks == 0:
                break
            
            # Try swap moves: remove one block, add another
            # Randomly pick a block to try removing
            order = list(range(n_blocks))
            random.shuffle(order)
            
            for idx in order:
                if time.time() - start_time > time_limit - 5.0:
                    break
                    
                b = current_blocks[idx]
                
                # Temporarily remove block b
                coverage_count[sub_indices[b]] -= 1
                
                # Find uncovered triples
                uncovered_mask = (coverage_count == 0)
                n_uncovered = np.sum(uncovered_mask)
                
                if n_uncovered == 0:
                    # Block was redundant - remove it
                    current_blocks.pop(idx)
                    block_set.discard(b)
                    improved = True
                    # Rebuild block_set and coverage
                    block_set = set(current_blocks)
                    # coverage_count already updated
                    break
                
                # Try to find a single block that covers all uncovered triples
                # and is not already in the set
                gains = uncovered_mask[sub_indices].sum(axis=1)
                
                # We need a block that covers ALL uncovered triples
                # So we need gains[i] == n_uncovered for some i
                full_cover_candidates = np.where(gains == n_uncovered)[0]
                
                # Filter out blocks already in the set
                valid_candidates = [c for c in full_cover_candidates if c not in block_set]
                
                if valid_candidates:
                    # Pick a candidate that maximizes "overlap" with existing blocks
                    # to help future reductions
                    # For simplicity, pick randomly
                    new_block = random.choice(valid_candidates)
                    
                    # Perform the swap
                    current_blocks[idx] = new_block
                    block_set.discard(b)
                    block_set.add(new_block)
                    coverage_count[sub_indices[new_block]] += 1
                    improved = True
                    break
                else:
                    # No single block can cover all uncovered - revert
                    coverage_count[sub_indices[b]] += 1
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
        # Fallback: run one greedy without time limit check inside (should be fast)
        blocks = greedy_cover(0)
        if blocks:
            best_solution = reduce_cover(blocks, 1)
            best_solution = polish(best_solution, 2)
            
    if best_solution is None:
        return []

    # Convert indices to actual blocks
    final_blocks = [list(all_ks[i]) for i in best_solution]
    
    return final_blocks
