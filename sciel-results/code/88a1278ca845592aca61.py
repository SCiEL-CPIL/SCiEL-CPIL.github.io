import numpy as np
import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # 1. Generate all t-subsets and map to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {sub: i for i, sub in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    # 2. Generate all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Precompute block_indices: shape (num_blocks, t_subsets_per_block)
    # t_subsets_per_block = C(k, t) = C(7, 3) = 35
    block_indices = np.empty((num_blocks, 35), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        for j, t_sub in enumerate(itertools.combinations(block, t)):
            block_indices[i, j] = t_subset_to_idx[t_sub]
            
    # Precompute set of triples for each block for fast subset checks
    block_triple_sets = [set(block_indices[i]) for i in range(num_blocks)]
    
    # Precompute frequency of each t-subset
    freq = np.bincount(block_indices.flatten(), minlength=num_t_subsets)
    inv_freq = 1.0 / (freq + 1e-9)
    
    uncovered = np.ones(num_t_subsets, dtype=bool)
    selected_blocks_idx = []
    
    # Greedy Phase
    while np.any(uncovered):
        if time.time() > deadline:
            break
            
        w = np.where(uncovered, inv_freq, 0.0)
        gains = np.sum(w[block_indices], axis=1)
        
        best_idx = np.argmax(gains)
        best_gain = gains[best_idx]
        
        if best_gain == 0:
            break
            
        selected_blocks_idx.append(best_idx)
        uncovered[block_indices[best_idx]] = False
        
    # Convert to list of indices for easier manipulation
    selected = set(selected_blocks_idx)
    
    # Maintain coverage counts for redundancy and local search
    coverage_count = np.zeros(num_t_subsets, dtype=np.int32)
    for idx in selected:
        coverage_count[block_indices[idx]] += 1
        
    # Phase 2: Redundancy Removal
    while True:
        if time.time() > deadline:
            break
        removed_any = False
        for idx in list(selected):
            coverage_count[block_indices[idx]] -= 1
            if np.min(coverage_count) > 0:
                selected.remove(idx)
                removed_any = True
            else:
                coverage_count[block_indices[idx]] += 1
                if time.time() > deadline:
                    break
        if not removed_any:
            break
            
    # Phase 3: 2-for-1 Local Search
    # Precompute for each triple, list of blocks covering it
    triple_to_blocks = [[] for _ in range(num_t_subsets)]
    for i in range(num_blocks):
        for t_idx in block_indices[i]:
            triple_to_blocks[t_idx].append(i)
            
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
            
        sel_list = list(selected)
        n_sel = len(sel_list)
        
        for i in range(n_sel):
            if time.time() > deadline:
                break
            for j in range(i + 1, n_sel):
                if time.time() > deadline:
                    break
                    
                idx_i = sel_list[i]
                idx_j = sel_list[j]
                
                set_i = block_triple_sets[idx_i]
                set_j = block_triple_sets[idx_j]
                union_ij = set_i.union(set_j)
                
                critical_triples = []
                for t in union_ij:
                    c = coverage_count[t]
                    n_cover = 0
                    if t in set_i: n_cover += 1
                    if t in set_j: n_cover += 1
                    
                    if c == n_cover:
                        critical_triples.append(t)
                        
                if not critical_triples:
                    # 2-for-0
                    selected.remove(idx_i)
                    selected.remove(idx_j)
                    coverage_count[block_indices[idx_i]] -= 1
                    coverage_count[block_indices[idx_j]] -= 1
                    improved = True
                    break
                else:
                    # Try to find a block k covering all critical_triples
                    first_t = critical_triples[0]
                    candidates = triple_to_blocks[first_t]
                    
                    found_replacement = False
                    for k in candidates:
                        if k in selected:
                            continue
                        set_k = block_triple_sets[k]
                        # Check if critical_triples is subset of set_k
                        is_subset = True
                        for t in critical_triples:
                            if t not in set_k:
                                is_subset = False
                                break
                        if is_subset:
                            # Replace i, j with k
                            selected.remove(idx_i)
                            selected.remove(idx_j)
                            selected.add(k)
                            coverage_count[block_indices[idx_i]] -= 1
                            coverage_count[block_indices[idx_j]] -= 1
                            coverage_count[block_indices[k]] += 1
                            improved = True
                            found_replacement = True
                            break
                            
                    if found_replacement:
                        break
                        
            if improved:
                break
                
    return [list(all_blocks[idx]) for idx in selected]
