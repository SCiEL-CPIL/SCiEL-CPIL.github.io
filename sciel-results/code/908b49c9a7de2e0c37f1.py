import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    def elapsed():
        return time.time() - start_time
    
    # 1. Precompute all t-subsets and map them to indices
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {sub: i for i, sub in enumerate(all_tsubs)}
    
    # 2. Generate all k-subsets (candidate blocks)
    all_ksubs = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubs)
    
    # 3. Precompute coverage mask for each k-subset as a Python int bitset
    ksub_masks = [0] * n_ksubs
    for j, ks in enumerate(all_ksubs):
        mask = 0
        for tsub in itertools.combinations(ks, t):
            idx = tsub_to_idx[tsub]
            mask |= (1 << idx)
        ksub_masks[j] = mask
    
    # Also build reverse mapping: t-subset index -> list of block indices covering it
    tsub_to_blocks = [[] for _ in range(n_tsubs)]
    for j, mask in enumerate(ksub_masks):
        m = mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            tsub_to_blocks[idx].append(j)
            m ^= lsb
    
    # Helper: greedy with randomization
    def greedy_random(seed_offset=0):
        """Greedy set cover with optional random tie-breaking."""
        rng = random.Random(seed_offset)
        uncovered_mask = (1 << n_tsubs) - 1
        remaining = n_tsubs
        blocks_indices = []
        used_blocks = set()
        
        while remaining > 0 and elapsed() < deadline - 20.0:
            best_j = -1
            best_gain = 0
            
            # Sample a subset of candidates if too many, or scan all
            # For efficiency, scan all but with early termination on time
            for j in range(n_ksubs):
                if j in used_blocks:
                    continue
                common = uncovered_mask & ksub_masks[j]
                gain = common.bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_j = j
                elif gain == best_gain and best_j != -1 and rng.random() < 0.1:
                    # Random tie-breaking to diversify
                    best_j = j
            
            if best_j == -1 or best_gain == 0:
                break
            
            blocks_indices.append(best_j)
            used_blocks.add(best_j)
            uncovered_mask &= ~ksub_masks[best_j]
            remaining -= best_gain
        
        return blocks_indices
    
    # Helper: remove redundant blocks
    def remove_redundants(indices, idx_set, cov):
        """Remove redundant blocks. Returns new list, set, and coverage."""
        changed = True
        while changed and elapsed() < deadline - 5.0:
            changed = False
            new_indices = []
            for idx in indices:
                mask = ksub_masks[idx]
                redundant = True
                m = mask
                while m:
                    lsb = m & (-m)
                    ts_idx = lsb.bit_length() - 1
                    if cov[ts_idx] == 1:
                        redundant = False
                        break
                    m ^= lsb
                
                if redundant:
                    idx_set.remove(idx)
                    m = mask
                    while m:
                        lsb = m & (-m)
                        ts_idx = lsb.bit_length() - 1
                        cov[ts_idx] -= 1
                        m ^= lsb
                    changed = True
                else:
                    new_indices.append(idx)
            indices = new_indices
        return indices, idx_set, cov
    
    # Helper: local search on a given solution
    def local_search(initial_indices, time_budget=30.0):
        """1-exchange local search. Returns improved solution."""
        current_indices = list(initial_indices)
        current_set = set(current_indices)
        
        # Initialize coverage counts
        coverage = [0] * n_tsubs
        for idx in current_indices:
            mask = ksub_masks[idx]
            m = mask
            while m:
                lsb = m & (-m)
                ts_idx = lsb.bit_length() - 1
                coverage[ts_idx] += 1
                m ^= lsb
        
        # Initial redundancy removal
        current_indices, current_set, coverage = remove_redundants(current_indices, current_set, coverage)
        
        local_start = time.time()
        improved = True
        steps = 0
        
        while improved and steps < 5000 and (time.time() - local_start) < time_budget and elapsed() < deadline - 5.0:
            improved = False
            steps += 1
            
            to_try = list(current_indices)
            random.shuffle(to_try)
            
            for b in to_try:
                if b not in current_set:
                    continue
                
                # Find critical t-subsets for b
                crit_ts = []
                mask_b = ksub_masks[b]
                m = mask_b
                while m:
                    lsb = m & (-m)
                    ts_idx = lsb.bit_length() - 1
                    if coverage[ts_idx] == 1:
                        crit_ts.append(ts_idx)
                    m ^= lsb
                
                if not crit_ts:
                    current_set.remove(b)
                    current_indices.remove(b)
                    m = mask_b
                    while m:
                        lsb = m & (-m)
                        ts_idx = lsb.bit_length() - 1
                        coverage[ts_idx] -= 1
                        m ^= lsb
                    improved = True
                    continue
                
                # Find candidates that cover all crit_ts
                candidates = None
                for ts in crit_ts:
                    blocks_covering = tsub_to_blocks[ts]
                    if candidates is None:
                        candidates = set(blocks_covering)
                    else:
                        candidates.intersection_update(blocks_covering)
                
                if not candidates:
                    continue
                
                candidates -= current_set
                
                if not candidates:
                    continue
                
                # Pick best candidate: maximize overlap with b
                best_cand = None
                best_score = -1
                cand_list = list(candidates)
                if len(cand_list) > 30:
                    cand_list = random.sample(cand_list, 30)
                
                for c in cand_list:
                    mask_c = ksub_masks[c]
                    common = mask_b & mask_c
                    score = common.bit_count()
                    if score > best_score:
                        best_score = score
                        best_cand = c
                
                if best_cand is None:
                    continue
                
                # Perform Swap
                old_size = len(current_indices)
                
                # Remove b
                current_set.remove(b)
                current_indices.remove(b)
                m = mask_b
                while m:
                    lsb = m & (-m)
                    ts_idx = lsb.bit_length() - 1
                    coverage[ts_idx] -= 1
                    m ^= lsb
                
                # Add c
                current_set.add(best_cand)
                current_indices.append(best_cand)
                mask_c = ksub_masks[best_cand]
                m = mask_c
                while m:
                    lsb = m & (-m)
                    ts_idx = lsb.bit_length() - 1
                    coverage[ts_idx] += 1
                    m ^= lsb
                
                # Run redundancy removal
                new_indices, new_set, new_cov = remove_redundants(current_indices, current_set, coverage)
                
                if len(new_indices) < old_size:
                    current_indices = new_indices
                    current_set = new_set
                    coverage = new_cov
                    improved = True
                else:
                    # Revert
                    current_set.remove(best_cand)
                    current_indices.remove(best_cand)
                    m = mask_c
                    while m:
                        lsb = m & (-m)
                        ts_idx = lsb.bit_length() - 1
                        coverage[ts_idx] -= 1
                        m ^= lsb
                    
                    current_set.add(b)
                    current_indices.append(b)
                    m = mask_b
                    while m:
                        lsb = m & (-m)
                        ts_idx = lsb.bit_length() - 1
                        coverage[ts_idx] += 1
                        m ^= lsb
        
        return current_indices
    
    # Multi-start: try several different greedy initializations
    best_solution = None
    best_size = float('inf')
    
    # First, do a deterministic greedy
    greedy_result = greedy_random(seed_offset=0)
    if greedy_result:
        ls_result = local_search(greedy_result, time_budget=20.0)
        if len(ls_result) < best_size:
            best_size = len(ls_result)
            best_solution = ls_result
    
    # Try a few randomized greedy starts if time permits
    n_restarts = 3
    for i in range(1, n_restarts + 1):
        if elapsed() > deadline - 25.0:
            break
        
        greedy_result = greedy_random(seed_offset=i * 1000)
        if greedy_result:
            ls_result = local_search(greedy_result, time_budget=15.0)
            if len(ls_result) < best_size:
                best_size = len(ls_result)
                best_solution = ls_result
    
    # If we still have time, do one more polish on the best solution
    if best_solution and elapsed() < deadline - 10.0:
        final_result = local_search(best_solution, time_budget=min(20.0, deadline - elapsed() - 5.0))
        if len(final_result) < best_size:
            best_solution = final_result
    
    if best_solution is None:
        # Fallback: return empty or minimal
        return []
    
    # Convert indices to blocks
    final_blocks = [list(all_ksubs[idx]) for idx in best_solution]
    return final_blocks
