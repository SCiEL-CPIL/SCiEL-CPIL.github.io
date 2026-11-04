import numpy as np
import itertools
import time
import random
import scipy.sparse as sp

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
    
    rows = []
    cols = []
    data = []
    
    for b_idx, block in enumerate(all_blocks):
        for s in itertools.combinations(block, t):
            s_idx = subset_to_idx[s]
            block_to_subsets[b_idx].append(s_idx)
            subset_to_blocks[s_idx].append(b_idx)
            rows.append(b_idx)
            cols.append(s_idx)
            data.append(1)
            
    incidence_matrix = sp.csr_matrix((data, (rows, cols)), shape=(num_blocks, num_subsets))
            
    def run_greedy_full(random_seed=None):
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
            
        uncovered = np.ones(num_subsets, dtype=bool)
        gains = incidence_matrix.dot(uncovered).astype(np.int32)
        
        selected_blocks = []
        
        while True:
            if not np.any(uncovered):
                break
                
            noise = rng.uniform(0, 0.5, size=num_blocks)
            scores = gains + noise
            best_idx = np.argmax(scores)
            
            if gains[best_idx] == 0:
                break
                
            selected_blocks.append(best_idx)
            
            b_row = incidence_matrix[best_idx].toarray().ravel()
            newly_covered_mask = uncovered & b_row.astype(bool)
            
            if not np.any(newly_covered_mask):
                continue

            uncovered = uncovered & ~b_row.astype(bool)
            gains = incidence_matrix.dot(uncovered).astype(np.int32)
                    
        return selected_blocks

    def run_greedy_partial(uncovered_mask, random_seed=None):
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
            
        uncovered = uncovered_mask.copy()
        if not np.any(uncovered):
            return []
            
        gains = incidence_matrix.dot(uncovered).astype(np.int32)
        
        selected_blocks = []
        
        while True:
            if not np.any(uncovered):
                break
                
            noise = rng.uniform(0, 0.5, size=num_blocks)
            scores = gains + noise
            best_idx = np.argmax(scores)
            
            if gains[best_idx] == 0:
                break
                
            selected_blocks.append(best_idx)
            
            b_row = incidence_matrix[best_idx].toarray().ravel()
            newly_covered_mask = uncovered & b_row.astype(bool)
            
            if not np.any(newly_covered_mask):
                continue

            uncovered = uncovered & ~b_row.astype(bool)
            gains = incidence_matrix.dot(uncovered).astype(np.int32)
                    
        return selected_blocks

    def minimize(blocks_indices, random_seed=None):
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
            
        current_blocks = list(blocks_indices)
        
        if not current_blocks:
            return current_blocks
            
        selected_mask = np.zeros(num_blocks, dtype=np.int32)
        for b_idx in current_blocks:
            selected_mask[b_idx] = 1
        cover_count = incidence_matrix.T.dot(selected_mask).astype(np.int32)
        
        while True:
            if not current_blocks:
                break
                
            redundant_blocks = []
            
            for i, b_idx in enumerate(current_blocks):
                sub_indices = block_to_subsets[b_idx]
                if not np.any(cover_count[sub_indices] == 1):
                    redundant_blocks.append(i)
                    
            if not redundant_blocks:
                break
                
            remove_pos = redundant_blocks[rng.randint(0, len(redundant_blocks)-1)]
            b_idx_to_remove = current_blocks[remove_pos]
            
            cover_count -= incidence_matrix[b_idx_to_remove].toarray().ravel()
            
            current_blocks.pop(remove_pos)
            
        return current_blocks

    best_blocks = None
    best_len = float('inf')
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    trial = 0
    while time.time() < deadline:
        # Strategy 1: Full Greedy + Minimize
        greedy_res = run_greedy_full(trial)
        min_res = minimize(greedy_res, trial)
        
        if len(min_res) < best_len:
            best_len = len(min_res)
            best_blocks = min_res
            
        # Strategy 2: Perturb Best + Repair + Minimize
        if best_blocks is not None and len(best_blocks) > 10:
            perturbed = list(best_blocks)
            # Remove 3 random blocks
            num_remove = min(3, len(perturbed) - 1)
            if num_remove > 0:
                remove_indices = random.sample(range(len(perturbed)), num_remove)
                # Remove from end to start to keep indices valid, or use set
                # Easier: create a set of blocks to remove
                blocks_to_remove = set(perturbed[i] for i in remove_indices)
                
                # Build uncovered mask
                selected_mask = np.zeros(num_blocks, dtype=np.int32)
                for b_idx in perturbed:
                    selected_mask[b_idx] = 1
                cover_count_temp = incidence_matrix.T.dot(selected_mask).astype(np.int32)
                
                for b_idx in blocks_to_remove:
                    cover_count_temp -= incidence_matrix[b_idx].toarray().ravel()
                    
                uncovered_mask = (cover_count_temp == 0).astype(bool)
                
                # Repair
                repair_res = run_greedy_partial(uncovered_mask, trial + 1000)
                
                # Combine
                new_set = [b for b in perturbed if b not in blocks_to_remove]
                new_set.extend(repair_res)
                
                # Minimize
                min_res2 = minimize(new_set, trial + 2000)
                
                if len(min_res2) < best_len:
                    best_len = len(min_res2)
                    best_blocks = min_res2

        trial += 1
        
    if best_blocks is None:
        greedy_res = run_greedy_full(0)
        best_blocks = minimize(greedy_res, 0)
        
    return [list(all_blocks[i]) for i in best_blocks]
