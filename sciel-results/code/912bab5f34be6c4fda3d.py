import numpy as np
import itertools
import time
import random
import sys

def construct(v, k, t):
    points = range(v)
    all_subsets = list(itertools.combinations(points, t))
    num_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute block_to_subsets and subset_to_blocks
    block_to_subsets = [[] for _ in range(num_blocks)]
    subset_to_blocks = [[] for _ in range(num_subsets)]
    
    for b_idx, block in enumerate(all_blocks):
        for s in itertools.combinations(block, t):
            s_idx = subset_to_idx[s]
            block_to_subsets[b_idx].append(s_idx)
            subset_to_blocks[s_idx].append(b_idx)
    
    # Precompute as numpy arrays for vectorized ops
    block_to_subsets_arr = [np.array(s, dtype=np.int32) for s in block_to_subsets]
    subset_to_blocks_arr = [np.array(s, dtype=np.int32) for s in subset_to_blocks]
    
    def run_greedy_full(random_seed=None):
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
            
        uncovered = np.ones(num_subsets, dtype=bool)
        gains = np.array([len(subs) for subs in block_to_subsets], dtype=np.int32)
        remaining_uncovered = num_subsets
        
        selected_blocks = []
        
        while remaining_uncovered > 0:
            noise = rng.uniform(0, 0.5, size=num_blocks)
            scores = gains + noise
            best_idx = int(np.argmax(scores))
            
            if gains[best_idx] == 0:
                break
                
            selected_blocks.append(best_idx)
            
            b_subs = block_to_subsets_arr[best_idx]
            newly_covered_mask = uncovered[b_subs]
            if not np.any(newly_covered_mask):
                continue
            newly_covered = b_subs[newly_covered_mask]
            n_new = len(newly_covered)
            
            uncovered[newly_covered] = False
            remaining_uncovered -= n_new
            
            # Vectorized gain update
            for s_idx in newly_covered:
                gains[subset_to_blocks_arr[s_idx]] -= 1
                    
        return selected_blocks

    def run_greedy_partial(uncovered_mask, random_seed=None):
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
            
        uncovered_indices = np.where(uncovered_mask)[0]
        if len(uncovered_indices) == 0:
            return []
            
        gains = np.zeros(num_blocks, dtype=np.int32)
        for s_idx in uncovered_indices:
            gains[subset_to_blocks_arr[s_idx]] += 1
                
        selected_blocks = []
        current_uncovered = uncovered_mask.copy()
        remaining = len(uncovered_indices)
        
        while remaining > 0:
            noise = rng.uniform(0, 0.5, size=num_blocks)
            scores = gains + noise
            best_idx = int(np.argmax(scores))
            
            if gains[best_idx] == 0:
                break
                
            selected_blocks.append(best_idx)
            
            b_subs = block_to_subsets_arr[best_idx]
            newly_covered_mask = current_uncovered[b_subs]
            if not np.any(newly_covered_mask):
                continue
            newly_covered = b_subs[newly_covered_mask]
            n_new = len(newly_covered)
            
            current_uncovered[newly_covered] = False
            remaining -= n_new
            for s_idx in newly_covered:
                gains[subset_to_blocks_arr[s_idx]] -= 1
                    
        return selected_blocks

    def minimize(blocks_indices, random_seed=None):
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
            
        current_blocks = list(blocks_indices)
        
        if not current_blocks:
            return current_blocks
            
        cover_count = np.zeros(num_subsets, dtype=np.int32)
        for b_idx in current_blocks:
            for s_idx in block_to_subsets_arr[b_idx]:
                cover_count[s_idx] += 1
                
        def remove_redundant():
            changed = True
            while changed:
                changed = False
                redundant_blocks = []
                for i, b_idx in enumerate(current_blocks):
                    is_redundant = True
                    for s_idx in block_to_subsets_arr[b_idx]:
                        if cover_count[s_idx] == 1:
                            is_redundant = False
                            break
                    if is_redundant:
                        redundant_blocks.append(i)
                    
                if redundant_blocks:
                    remove_pos = redundant_blocks[rng.randint(0, len(redundant_blocks)-1)]
                    b_idx_to_remove = current_blocks[remove_pos]
                    for s_idx in block_to_subsets_arr[b_idx_to_remove]:
                        cover_count[s_idx] -= 1
                    current_blocks.pop(remove_pos)
                    changed = True

        # Phase 1: Remove redundant blocks
        remove_redundant()
        
        # Phase 2: Swap-based local search
        # Precompute bitmask for each block's t-subsets for fast containment checks
        block_subsets_sets = [set(s.tolist()) for s in block_to_subsets_arr]
        
        for _outer in range(30):
            if time.time() > deadline - 2.0:
                break
            if not current_blocks:
                break
            
            improved = False
            n_blocks = len(current_blocks)
            
            for _attempt in range(min(n_blocks, 15)):
                if time.time() > deadline - 2.0:
                    break
                i = rng.randint(0, len(current_blocks) - 1)
                b_idx = current_blocks[i]
                
                # Find subsets uniquely covered by this block
                unique_subs = []
                for s_idx in block_to_subsets_arr[b_idx]:
                    if cover_count[s_idx] == 1:
                        unique_subs.append(s_idx)
                
                if not unique_subs:
                    for s_idx in block_to_subsets_arr[b_idx]:
                        cover_count[s_idx] -= 1
                    current_blocks.pop(i)
                    improved = True
                    break
                
                # Find a block that covers all unique_subs and is not already in the set
                current_set = set(current_blocks)
                unique_subs_set = set(unique_subs)
                best_new = -1
                best_overlap = -1
                
                # Gather candidate blocks from the first unique subset
                candidates = subset_to_blocks_arr[unique_subs[0]]
                for cand in candidates:
                    if cand in current_set or cand == b_idx:
                        continue
                    # Check if cand covers all unique_subs using precomputed set
                    if unique_subs_set.issubset(block_subsets_sets[cand]):
                        overlap = np.sum(cover_count[block_to_subsets_arr[cand]] >= 1)
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_new = cand
                
                if best_new >= 0:
                    for s_idx in block_to_subsets_arr[b_idx]:
                        cover_count[s_idx] -= 1
                    current_blocks[i] = best_new
                    for s_idx in block_to_subsets_arr[best_new]:
                        cover_count[s_idx] += 1
                    improved = True
                    break
            
            if not improved:
                break
        
        # Phase 3: Final redundancy pass
        remove_redundant()
        
        return current_blocks

    best_blocks = None
    best_len = float('inf')
    
    start_time = time.time()
    deadline = start_time + 115.0
    
    trial = 0
    while time.time() < deadline - 3.0:
        # Strategy 1: Full Greedy + Minimize
        greedy_res = run_greedy_full(trial)
        min_res = minimize(greedy_res, trial)
        
        if len(min_res) < best_len:
            best_len = len(min_res)
            best_blocks = min_res
            
        # Strategy 2: Perturb Best + Repair + Minimize
        if best_blocks is not None and len(best_blocks) > 10 and time.time() < deadline - 10.0:
            perturbed = list(best_blocks)
            num_remove = min(3, len(perturbed) - 1)
            if num_remove > 0:
                remove_indices = random.sample(range(len(perturbed)), num_remove)
                blocks_to_remove = set(perturbed[i] for i in remove_indices)
                
                cover_count_temp = np.zeros(num_subsets, dtype=np.int32)
                for b_idx in perturbed:
                    if b_idx in blocks_to_remove:
                        continue
                    for s_idx in block_to_subsets_arr[b_idx]:
                        cover_count_temp[s_idx] += 1
                        
                uncovered_mask = (cover_count_temp == 0).astype(bool)
                
                repair_res = run_greedy_partial(uncovered_mask, trial + 1000)
                
                new_set = [b for b in perturbed if b not in blocks_to_remove]
                new_set.extend(repair_res)
                
                min_res2 = minimize(new_set, trial + 2000)
                
                if len(min_res2) < best_len:
                    best_len = len(min_res2)
                    best_blocks = min_res2

        trial += 1
        
    if best_blocks is None:
        greedy_res = run_greedy_full(0)
        best_blocks = minimize(greedy_res, 0)
        
    return [list(all_blocks[i]) for i in best_blocks]
