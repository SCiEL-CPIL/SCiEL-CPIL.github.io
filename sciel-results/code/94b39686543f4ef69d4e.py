import numpy as np
import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 130  # 20s margin for safety
    
    # 1. Generate all t-subsets and map to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {sub: i for i, sub in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    # 2. Generate all k-subsets (blocks)
    blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(blocks)
    
    # Precompute block_indices: shape (num_blocks, t_subsets_per_block)
    # t_subsets_per_block = C(k, t) = C(7, 3) = 35
    t_subsets_per_block = int(np.math.comb(k, t)) if hasattr(np, 'math') else 35
    block_indices = np.empty((num_blocks, t_subsets_per_block), dtype=np.int32)
    for i, block in enumerate(blocks):
        for j, t_sub in enumerate(itertools.combinations(block, t)):
            block_indices[i, j] = t_subset_to_idx[t_sub]
            
    # Precompute frequency of each t-subset
    freq = np.bincount(block_indices.flatten(), minlength=num_t_subsets)
    
    # Inverse frequency for weighting
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
            
    # Precompute for each triple, list of blocks covering it
    triple_to_blocks = [[] for _ in range(num_t_subsets)]
    for i in range(num_blocks):
        for t_idx in block_indices[i]:
            triple_to_blocks[t_idx].append(i)
    
    # Precompute block triple sets for fast membership checking
    block_triple_sets = [set(block_indices[i]) for i in range(num_blocks)]
    
    # Phase 3: 2-for-1 Local Search
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
                
                t_set_i = block_triple_sets[idx_i]
                t_set_j = block_triple_sets[idx_j]
                t_union = t_set_i.union(t_set_j)
                
                critical_triples = []
                for t in t_union:
                    c = coverage_count[t]
                    in_i = t in t_set_i
                    in_j = t in t_set_j
                    n_cover_by_pair = (1 if in_i else 0) + (1 if in_j else 0)
                    
                    if c - n_cover_by_pair == 0:
                        critical_triples.append(t)
                        
                if not critical_triples:
                    selected.remove(idx_i)
                    selected.remove(idx_j)
                    coverage_count[block_indices[idx_i]] -= 1
                    coverage_count[block_indices[idx_j]] -= 1
                    improved = True
                    break
                else:
                    first_t = critical_triples[0]
                    candidates = triple_to_blocks[first_t]
                    
                    found = False
                    for k in candidates:
                        if k in selected:
                            continue
                        kt_set = block_triple_sets[k]
                        valid = True
                        for t in critical_triples:
                            if t not in kt_set:
                                valid = False
                                break
                        if valid:
                            # Remove i and j, add k
                            selected.remove(idx_i)
                            selected.remove(idx_j)
                            selected.add(k)
                            coverage_count[block_indices[idx_i]] -= 1
                            coverage_count[block_indices[idx_j]] -= 1
                            coverage_count[block_indices[k]] += 1
                            improved = True
                            found = True
                            break
                    if found:
                        break
                        
            if improved:
                break
                
    # Phase 4: 3-for-2 Local Search
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
                for l in range(j + 1, n_sel):
                    if time.time() > deadline:
                        break
                        
                    idx_i = sel_list[i]
                    idx_j = sel_list[j]
                    idx_l = sel_list[l]
                    
                    t_set_i = block_triple_sets[idx_i]
                    t_set_j = block_triple_sets[idx_j]
                    t_set_l = block_triple_sets[idx_l]
                    t_union = t_set_i.union(t_set_j).union(t_set_l)
                    
                    critical_triples = []
                    for t in t_union:
                        c = coverage_count[t]
                        n_cover = (1 if t in t_set_i else 0) + (1 if t in t_set_j else 0) + (1 if t in t_set_l else 0)
                        if c - n_cover == 0:
                            critical_triples.append(t)
                            
                    if not critical_triples:
                        # 3-for-0: just remove all three
                        selected.remove(idx_i)
                        selected.remove(idx_j)
                        selected.remove(idx_l)
                        coverage_count[block_indices[idx_i]] -= 1
                        coverage_count[block_indices[idx_j]] -= 1
                        coverage_count[block_indices[idx_l]] -= 1
                        improved = True
                        break
                    else:
                        # Try to find 2 blocks that cover all critical triples
                        first_t = critical_triples[0]
                        candidates = triple_to_blocks[first_t]
                        cand_list = [k for k in candidates if k not in selected]
                        
                        found = False
                        for a in range(len(cand_list)):
                            if time.time() > deadline:
                                break
                            for b in range(a + 1, len(cand_list)):
                                ka = cand_list[a]
                                kb = cand_list[b]
                                kat_set = block_triple_sets[ka]
                                kbt_set = block_triple_sets[kb]
                                valid = True
                                for t in critical_triples:
                                    if t not in kat_set and t not in kbt_set:
                                        valid = False
                                        break
                                if valid:
                                    selected.remove(idx_i)
                                    selected.remove(idx_j)
                                    selected.remove(idx_l)
                                    selected.add(ka)
                                    selected.add(kb)
                                    coverage_count[block_indices[idx_i]] -= 1
                                    coverage_count[block_indices[idx_j]] -= 1
                                    coverage_count[block_indices[idx_l]] -= 1
                                    coverage_count[block_indices[ka]] += 1
                                    coverage_count[block_indices[kb]] += 1
                                    improved = True
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break
                            
            if improved:
                break
                
    # Final redundancy removal pass
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
        if not removed_any:
            break
            
    return [list(blocks[idx]) for idx in selected]
