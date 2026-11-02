import numpy as np
import itertools
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    time_limit = 118.0  # Safety margin for 120s limit
    
    # Precomputation
    points = range(v)
    all_ts = list(itertools.combinations(points, t))
    n_ts = len(all_ts)
    idx_map = {comb: i for i, comb in enumerate(all_ts)}
    
    # Generate all k-subsets
    all_ks = list(itertools.combinations(points, k))
    n_ks = len(all_ks)
    
    # Precompute the t-subsets contained in each k-subset
    def nCr(n, r):
        if r > n: return 0
        if r == 0: return 1
        res = 1
        for i in range(r):
            res *= (n - i)
            res //= (i + 1)
        return res
        
    t_count = nCr(k, t)
    
    # block_t_indices[i] is an array of indices into all_ts for the i-th block
    block_t_indices = np.zeros((n_ks, t_count), dtype=np.int32)
    for i, block in enumerate(all_ks):
        subs = list(itertools.combinations(block, t))
        for j, sub in enumerate(subs):
            block_t_indices[i, j] = idx_map[sub]
    
    # Build reverse index: for each t-subset, which blocks contain it
    # This is useful for local search swaps
    triple_to_blocks = [[] for _ in range(n_ts)]
    for b_idx in range(n_ks):
        for sub_idx in block_t_indices[b_idx]:
            triple_to_blocks[sub_idx].append(b_idx)
    triple_to_blocks = [np.array(lst, dtype=np.int32) for lst in triple_to_blocks]
    
    best_blocks_global = []
    best_count_global = float('inf')
    
    # Multi-restart Greedy
    # Estimate time per restart. Precomputation took some time.
    # Gain calculation is ~0.05s per step? 76 steps * 0.05s = 4s per restart.
    # We can afford ~10-20 restarts.
    
    greedy_deadline = start_time + time_limit * 0.40
    rng = random.Random(42)
    
    restart_count = 0
    while time.time() < greedy_deadline:
        restart_count += 1
        covered = np.zeros(n_ts, dtype=bool)
        blocks_indices = []
        
        while True:
            if np.all(covered):
                break
                
            # Calculate gain for all blocks
            # uncovered_flags = ~covered
            # gains = sum of uncovered_flags over each block's triples
            # Using NumPy: 
            # covered is bool array of size n_ts
            # block_t_indices is int array of size (n_ks, t_count)
            # ~covered[block_t_indices] gives bool array of size (n_ks, t_count)
            # .sum(axis=1) gives gains for each block
            
            gains = np.sum(~covered[block_t_indices], axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                # No block can cover any new triple? Should not happen if covered is not all True
                break
                
            candidates = np.where(gains == max_gain)[0]
            
            # Random tie-breaking
            if len(candidates) > 1:
                # Add a small random noise to break ties more effectively?
                # Or just pick randomly.
                best_idx = candidates[rng.randint(0, len(candidates)-1)]
            else:
                best_idx = candidates[0]
                
            blocks_indices.append(best_idx)
            covered[block_t_indices[best_idx]] = True
            
        if len(blocks_indices) < best_count_global:
            best_count_global = len(blocks_indices)
            best_blocks_global = list(blocks_indices)
            
    # If no blocks found (should not happen), fallback
    if not best_blocks_global:
        # Fallback to simple greedy without restarts if time was too short for even one?
        # But the loop above runs at least once.
        pass

    # Local Search on the best greedy solution
    current_blocks = best_blocks_global
    ls_start = time.time()
    ls_time_limit = time_limit - (ls_start - start_time) - 5.0
    
    # 1. Redundancy Removal
    def remove_redundant(blocks):
        if not blocks:
            return blocks
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in blocks:
            coverage_count[block_t_indices[b_idx]] += 1
            
        new_blocks = []
        for b_idx in blocks:
            subs_in_block = block_t_indices[b_idx]
            # If every triple in this block is covered by at least 2 blocks, 
            # then this block is redundant.
            if np.min(coverage_count[subs_in_block]) > 1:
                coverage_count[subs_in_block] -= 1
            else:
                new_blocks.append(b_idx)
        return new_blocks

    current_blocks = remove_redundant(current_blocks)
    
    # 2. 1-for-1 Swap Local Search
    # Try to replace a block B_i with a block B_j such that B_j covers all 
    # triples that were uniquely covered by B_i.
    
    improved = True
    while improved and (time.time() - ls_start) < ls_time_limit * 0.5:
        improved = False
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in current_blocks:
            coverage_count[block_t_indices[b_idx]] += 1
            
        new_blocks = list(current_blocks)
        
        for i, b_idx in enumerate(current_blocks):
            if (time.time() - ls_start) > ls_time_limit * 0.5:
                break
                
            subs_in_block = block_t_indices[b_idx]
            # Find triples uniquely covered by b_idx
            unique_mask = coverage_count[subs_in_block] == 1
            if not np.any(unique_mask):
                # No unique triples, block is redundant (handled by remove_redundant)
                # But if we didn't remove it, we can remove it.
                # Actually, remove_redundant handles this. 
                # If we are here, it means it has at least one unique triple.
                pass
                
            unique_subs = subs_in_block[unique_mask]
            
            # Find candidate replacement blocks that cover all unique_subs
            # Candidate blocks must contain all triples in unique_subs.
            # We can iterate through blocks containing the first unique triple.
            if len(unique_subs) == 0:
                continue
                
            first_unique = unique_subs[0]
            candidates = triple_to_blocks[first_unique]
            
            found_swap = False
            for cand_idx in candidates:
                if cand_idx in current_blocks:
                    continue
                    
                # Check if cand_idx covers all unique_subs
                cand_subs = block_t_indices[cand_idx]
                # Efficient check: 
                # We need to verify if all elements of unique_subs are in cand_subs.
                # Since cand_subs is a numpy array, we can use np.in1d or set logic.
                # For small sizes (35 triples), a set is fast.
                cand_set = set(cand_subs.tolist())
                if all(sub in cand_set for sub in unique_subs):
                    # Valid swap
                    # Replace b_idx with cand_idx
                    new_blocks[i] = cand_idx
                    improved = True
                    found_swap = True
                    break
                    
            if found_swap:
                current_blocks = new_blocks
                # Restart the loop to recalculate coverage counts
                break
                
    current_blocks = remove_redundant(current_blocks)
    
    # 3. 2-for-1 Replacement Local Search
    # Try to replace two blocks B_i, B_j with a single block B_k
    # such that B_k covers all triples that were uniquely covered by B_i or B_j 
    # (considering the state where B_i and B_j are removed).
    
    improved = True
    while improved and (time.time() - ls_start) < ls_time_limit * 0.8:
        improved = False
        coverage_count = np.zeros(n_ts, dtype=np.int32)
        for b_idx in current_blocks:
            coverage_count[block_t_indices[b_idx]] += 1
            
        n_blocks = len(current_blocks)
        
        for i in range(n_blocks):
            if (time.time() - ls_start) > ls_time_limit * 0.8:
                break
            for j in range(i + 1, n_blocks):
                if (time.time() - ls_start) > ls_time_limit * 0.8:
                    break
                    
                b1 = current_blocks[i]
                b2 = current_blocks[j]
                
                subs1 = block_t_indices[b1]
                subs2 = block_t_indices[b2]
                
                # Simulate removing b1 and b2
                # A triple is critical if its coverage count becomes 0 after removing b1 and b2.
                # Coverage count of sub s:
                # c_s = coverage_count[s]
                # If s is in b1, c_s -= 1
                # If s is in b2, c_s -= 1 (if b1 != b2, which is true)
                # If s is in both, c_s -= 2
                
                # We need to find all triples s such that:
                # (s in b1 and c_s == 1) OR (s in b2 and c_s == 1) OR (s in b1 and s in b2 and c_s == 2)
                
                critical_set = set()
                
                for sub_idx in subs1:
                    if coverage_count[sub_idx] == 1:
                        critical_set.add(sub_idx)
                        
                for sub_idx in subs2:
                    if coverage_count[sub_idx] == 1:
                        critical_set.add(sub_idx)
                        
                # Check for overlap
                # If a sub is in both, and count is 2, it becomes 0.
                # We can check this by intersecting subs1 and subs2
                # and checking if count is 2.
                subs1_set = set(subs1.tolist())
                for sub_idx in subs2:
                    if sub_idx in subs1_set and coverage_count[sub_idx] == 2:
                        critical_set.add(sub_idx)
                        
                if not critical_set:
                    # Both blocks are redundant?
                    # If no critical triples, we can just remove both.
                    # This is handled by remove_redundant, but let's be safe.
                    # Actually, if no critical triples, we can remove one or both.
                    # Let's just remove one to be conservative, or both?
                    # If we remove both, we save 2 blocks.
                    # But we need to ensure we don't break coverage.
                    # If critical_set is empty, it means every triple in b1 or b2 is covered by others.
                    # So we can remove both.
                    new_blocks = [b for idx, b in enumerate(current_blocks) if idx != i and idx != j]
                    current_blocks = new_blocks
                    improved = True
                    break
                else:
                    # Find a block B_k that covers all critical triples
                    # Candidate blocks must contain at least one critical triple.
                    candidate_blocks = set()
                    for c_idx in critical_set:
                        for blk in triple_to_blocks[c_idx]:
                            candidate_blocks.add(blk)
                            
                    candidate_blocks.discard(b1)
                    candidate_blocks.discard(b2)
                    
                    found_replacement = False
                    for cand in candidate_blocks:
                        if cand in current_blocks:
                            continue
                            
                        cand_subs = set(block_t_indices[cand].tolist())
                        all_covered = True
                        for c in critical_set:
                            if c not in cand_subs:
                                all_covered = False
                                break
                                
                        if all_covered:
                            # Replace b1, b2 with cand
                            new_blocks = [b for idx, b in enumerate(current_blocks) if idx != i and idx != j]
                            if cand not in new_blocks:
                                new_blocks.append(cand)
                            current_blocks = new_blocks
                            improved = True
                            found_replacement = True
                            break
                            
                    if found_replacement:
                        break
                        
            if improved:
                break
                
    current_blocks = remove_redundant(current_blocks)
    
    # Final validation and conversion
    final_blocks = [list(all_ks[i]) for i in current_blocks]
    
    # Quick check if valid (should be, but just in case)
    # In a real scenario, we might want to verify. 
    # But the local search logic is sound.
    
    return final_blocks
