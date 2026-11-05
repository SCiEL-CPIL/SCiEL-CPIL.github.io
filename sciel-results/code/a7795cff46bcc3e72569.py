import itertools
import time
import numpy as np
import scipy.sparse
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0

    def time_ok():
        return time.time() < deadline

    # 1. Generate all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # 2. Generate all k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_k_blocks)

    # 3. Build Sparse Matrix M (n_blocks x n_t) and auxiliary structures
    row_indices = []
    col_indices = []
    data = []
    
    # block_to_subsets: list of lists of t-subset indices for each block
    block_to_subsets = [[] for _ in range(n_blocks)]
    # subset_to_blocks: list of lists of block indices for each t-subset
    subset_to_blocks = [[] for _ in range(n_t)]
    
    for b_idx, block in enumerate(all_k_blocks):
        for t_sub in itertools.combinations(block, t):
            s_idx = t_sub_to_idx.get(t_sub)
            if s_idx is not None:
                row_indices.append(b_idx)
                col_indices.append(s_idx)
                data.append(1)
                block_to_subsets[b_idx].append(s_idx)
                subset_to_blocks[s_idx].append(b_idx)
                
    M = scipy.sparse.csr_matrix((data, (row_indices, col_indices)), shape=(n_blocks, n_t), dtype=np.int32)
    
    # Precompute column degrees (how many blocks cover each t-subset)
    col_degrees = np.asarray(M.sum(axis=0)).flatten().astype(np.float64)
    # Weight for each t-subset: inverse of degree (subsets covered by fewer blocks are more valuable)
    subset_weights = 1.0 / np.maximum(col_degrees, 1.0)
    
    def greedy_construct(randomize=False, noise_scale=0.0, use_weights=True):
        uncovered = np.ones(n_t, dtype=np.int32)
        active_blocks = []
        
        # Weighted gain: sum of weights of uncovered subsets
        if use_weights:
            # Precompute weighted M
            # M_weighted = M with columns scaled by subset_weights
            weighted_uncovered = uncovered.astype(np.float64) * subset_weights
        else:
            weighted_uncovered = uncovered.astype(np.float64)
        
        while uncovered.sum() > 0:
            if not time_ok():
                break
                
            # Calculate gains
            if use_weights:
                gains = M @ weighted_uncovered
            else:
                gains = M @ uncovered
            
            if randomize and noise_scale > 0:
                noise = np.random.uniform(0, 1, n_blocks) * noise_scale
                gains = gains + noise
            
            # Pick best block
            best_b = int(np.argmax(gains))
            
            if gains[best_b] <= 0:
                break
            
            active_blocks.append(best_b)
            
            # Update uncovered
            start = M.indptr[best_b]
            end = M.indptr[best_b + 1]
            cols = M.indices[start:end]
            uncovered[cols] = 0
            
            if use_weights:
                weighted_uncovered[cols] = 0
            else:
                weighted_uncovered[cols] = 0
            
        return active_blocks

    def randomized_initialization(p=0.05, max_iter=100, use_weights=True):
        if not time_ok():
            return []
            
        mask = np.random.random(n_blocks) < p
        sampled_indices = np.where(mask)[0]
        
        uncovered = np.ones(n_t, dtype=np.int32)
        active_blocks = list(sampled_indices)
        
        if len(active_blocks) > 0:
            sub_M = M[active_blocks, :]
            covered_mask = (np.asarray(sub_M.sum(axis=0)).flatten() > 0)
            uncovered[covered_mask] = 0
        
        if use_weights:
            weighted_uncovered = uncovered.astype(np.float64) * subset_weights
        else:
            weighted_uncovered = uncovered.astype(np.float64)
        
        while uncovered.sum() > 0:
            if not time_ok():
                break
                
            if use_weights:
                gains = M @ weighted_uncovered
            else:
                gains = M @ uncovered
            best_b = int(np.argmax(gains))
            
            if gains[best_b] <= 0:
                break
            
            active_blocks.append(best_b)
            
            start = M.indptr[best_b]
            end = M.indptr[best_b + 1]
            cols = M.indices[start:end]
            uncovered[cols] = 0
            weighted_uncovered[cols] = 0
            
        return active_blocks

    def redundancy_removal(active_blocks, time_limit=5.0):
        if not active_blocks:
            return []
            
        active_arr = np.array(active_blocks, dtype=np.int32)
        
        end_time = time.time() + time_limit
        improved = True
        
        while improved and time.time() < end_time:
            if len(active_arr) == 0:
                break
            
            sub_M = M[active_arr, :]
            counts = np.asarray(sub_M.sum(axis=0)).flatten()
            
            remove_mask = np.zeros(len(active_arr), dtype=bool)
            
            for i in range(len(active_arr)):
                b_idx = active_arr[i]
                start = M.indptr[b_idx]
                end = M.indptr[b_idx + 1]
                if start == end:
                    remove_mask[i] = True
                    continue
                
                cols = M.indices[start:end]
                if np.min(counts[cols]) > 1:
                    remove_mask[i] = True
            
            if not remove_mask.any():
                break
            
            active_arr = active_arr[~remove_mask]
            improved = True
            
        return list(active_arr)

    def swap_local_search(active_blocks, time_limit=10.0):
        if not active_blocks:
            return []
            
        current_set = list(active_blocks)
        current_set_set = set(current_set)
        
        active_arr = np.array(current_set, dtype=np.int32)
        sub_M = M[active_arr, :]
        coverage_counts = np.asarray(sub_M.sum(axis=0)).flatten()
        
        best_set = list(current_set)
        best_len = len(current_set)
        
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            changed = False
            indices = list(range(len(current_set)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > end_time:
                    break
                
                old_block = current_set[i]
                
                # Find unique subsets covered by old_block
                unique_subs = []
                start = M.indptr[old_block]
                end = M.indptr[old_block + 1]
                if start != end:
                    cols = M.indices[start:end]
                    for s in cols:
                        if coverage_counts[s] == 1:
                            unique_subs.append(s)
                
                if not unique_subs:
                    # Block is redundant - try to remove it directly
                    old_cols = M.indices[M.indptr[old_block]:M.indptr[old_block + 1]]
                    coverage_counts[old_cols] -= 1
                    current_set_set.remove(old_block)
                    current_set.pop(i)
                    if len(current_set) < best_len:
                        best_len = len(current_set)
                        best_set = list(current_set)
                        changed = True
                    # Restart iteration since indices changed
                    indices = list(range(len(current_set)))
                    random.shuffle(indices)
                    break
                else:
                    # Block is critical for unique_subs
                    candidates = None
                    for s in unique_subs:
                        if candidates is None:
                            candidates = set(subset_to_blocks[s])
                        else:
                            candidates.intersection_update(subset_to_blocks[s])
                            if not candidates:
                                break
                                
                    if not candidates:
                        continue
                    
                    candidates.discard(old_block)
                    if not candidates:
                        continue
                    
                    # Score candidates by how many currently-covered subsets they add
                    cand_list = list(candidates)
                    random.shuffle(cand_list)
                    
                    new_block = None
                    best_score = -1
                    for cand in cand_list[:20]:  # Limit candidates for speed
                        if cand in current_set_set:
                            continue
                        # Score: number of subsets this block covers that are currently covered
                        # (to maintain coverage) minus number of new subsets (to avoid bloat)
                        cand_cols = M.indices[M.indptr[cand]:M.indptr[cand + 1]]
                        score = np.sum(coverage_counts[cand_cols] > 0)
                        if score > best_score:
                            best_score = score
                            new_block = cand
                            
                    if new_block is None:
                        continue
                    
                    # Perform swap
                    old_cols = M.indices[M.indptr[old_block]:M.indptr[old_block + 1]]
                    new_cols = M.indices[M.indptr[new_block]:M.indptr[new_block + 1]]
                    
                    coverage_counts[old_cols] -= 1
                    coverage_counts[new_cols] += 1
                    
                    current_set[i] = new_block
                    current_set_set.remove(old_block)
                    current_set_set.add(new_block)
            
            # Deletion pass after swaps
            if time.time() < end_time:
                active_arr = np.array(current_set, dtype=np.int32)
                sub_M = M[active_arr, :]
                coverage_counts = np.asarray(sub_M.sum(axis=0)).flatten()
                
                remove_mask = np.zeros(len(current_set), dtype=bool)
                for i in range(len(current_set)):
                    b_idx = current_set[i]
                    start = M.indptr[b_idx]
                    end = M.indptr[b_idx + 1]
                    if start == end:
                        remove_mask[i] = True
                        continue
                    cols = M.indices[start:end]
                    if np.min(coverage_counts[cols]) > 1:
                        remove_mask[i] = True
                        
                if remove_mask.any():
                    to_remove = np.where(remove_mask)[0]
                    for idx in to_remove:
                        b_idx = current_set[idx]
                        cols = M.indices[M.indptr[b_idx]:M.indptr[b_idx + 1]]
                        coverage_counts[cols] -= 1
                        current_set_set.remove(b_idx)
                    
                    new_current_set = []
                    for i, b in enumerate(current_set):
                        if not remove_mask[i]:
                            new_current_set.append(b)
                    current_set = new_current_set
                    
                    if len(current_set) < best_len:
                        best_len = len(current_set)
                        best_set = list(current_set)
                        changed = True
            
            if not changed and len(current_set) == best_len:
                break
            
        return best_set

    def two_opt_swap(active_blocks, time_limit=5.0):
        """Try to swap two blocks for two different blocks, reducing total count."""
        if not active_blocks or len(active_blocks) < 2:
            return list(active_blocks)
        
        current_set = list(active_blocks)
        current_set_set = set(current_set)
        
        active_arr = np.array(current_set, dtype=np.int32)
        sub_M = M[active_arr, :]
        coverage_counts = np.asarray(sub_M.sum(axis=0)).flatten()
        
        best_set = list(current_set)
        best_len = len(current_set)
        
        end_time = time.time() + time_limit
        n_set = len(current_set)
        
        while time.time() < end_time:
            improved = False
            # Try random pairs
            for _ in range(50):
                if time.time() > end_time:
                    break
                if n_set < 2:
                    break
                
                i, j = random.sample(range(n_set), 2)
                b1 = current_set[i]
                b2 = current_set[j]
                
                # Find unique subsets for b1 and b2
                unique_b1 = []
                unique_b2 = []
                cols_b1 = M.indices[M.indptr[b1]:M.indptr[b1 + 1]]
                cols_b2 = M.indices[M.indptr[b2]:M.indptr[b2 + 1]]
                
                for s in cols_b1:
                    if coverage_counts[s] == 1:
                        unique_b1.append(s)
                for s in cols_b2:
                    if coverage_counts[s] == 1:
                        unique_b2.append(s)
                
                # Combined unique subsets that must be covered
                unique_combined = set(unique_b1) | set(unique_b2)
                if not unique_combined:
                    # Both redundant? Remove one
                    continue
                
                # Find two blocks that cover all unique_combined
                # This is expensive; try a heuristic
                # For each candidate block, check if it covers most of unique_combined
                # Then find a second block to cover the rest
                
                # Simplified: just try random candidates
                for _ in range(10):
                    if time.time() > end_time:
                        break
                    c1 = random.randint(0, n_blocks - 1)
                    c2 = random.randint(0, n_blocks - 1)
                    if c1 in current_set_set or c2 in current_set_set or c1 == c2:
                        continue
                    
                    cols_c1 = M.indices[M.indptr[c1]:M.indptr[c1 + 1]]
                    cols_c2 = M.indices[M.indptr[c2]:M.indptr[c2 + 1]]
                    
                    # Check if c1 and c2 together cover all unique_combined
                    covered_by_new = set(cols_c1) | set(cols_c2)
                    if unique_combined.issubset(covered_by_new):
                        # Valid swap!
                        # Update counts
                        coverage_counts[cols_b1] -= 1
                        coverage_counts[cols_b2] -= 1
                        coverage_counts[cols_c1] += 1
                        coverage_counts[cols_c2] += 1
                        
                        current_set[i] = c1
                        current_set[j] = c2
                        current_set_set.remove(b1)
                        current_set_set.remove(b2)
                        current_set_set.add(c1)
                        current_set_set.add(c2)
                        
                        improved = True
                        # Now try to delete redundant blocks
                        active_arr = np.array(current_set, dtype=np.int32)
                        sub_M = M[active_arr, :]
                        coverage_counts = np.asarray(sub_M.sum(axis=0)).flatten()
                        
                        remove_mask = np.zeros(len(current_set), dtype=bool)
                        for idx in range(len(current_set)):
                            b_idx = current_set[idx]
                            start = M.indptr[b_idx]
                            end = M.indptr[b_idx + 1]
                            if start == end:
                                remove_mask[idx] = True
                                continue
                            cols = M.indices[start:end]
                            if np.min(coverage_counts[cols]) > 1:
                                remove_mask[idx] = True
                        
                        if remove_mask.any():
                            to_remove = np.where(remove_mask)[0]
                            for idx in to_remove:
                                b_idx = current_set[idx]
                                cols = M.indices[M.indptr[b_idx]:M.indptr[b_idx + 1]]
                                coverage_counts[cols] -= 1
                                current_set_set.remove(b_idx)
                            
                            new_current_set = []
                            for idx, b in enumerate(current_set):
                                if not remove_mask[idx]:
                                    new_current_set.append(b)
                            current_set = new_current_set
                            n_set = len(current_set)
                            
                            if len(current_set) < best_len:
                                best_len = len(current_set)
                                best_set = list(current_set)
                                improved = True
                        break
            
            if not improved:
                break
        
        return best_set

    # Strategy:
    # 1. Run weighted greedy
    # 2. Run multiple randomized variants
    # 3. Apply redundancy removal
    # 4. Apply swap local search
    # 5. Apply 2-opt swap
    # 6. Return best result

    best_result = []
    best_len = float('inf')

    def update_best(candidate):
        nonlocal best_result, best_len
        if candidate is None:
            return
        if len(candidate) < best_len:
            best_len = len(candidate)
            best_result = candidate

    # Phase 1: Weighted Greedy (Robust baseline)
    if time_ok():
        try:
            g1 = greedy_construct(randomize=False, use_weights=True)
            update_best(g1)
            if time_ok():
                r1 = redundancy_removal(g1, time_limit=5.0)
                update_best(r1)
        except Exception:
            pass

    # Phase 2: Multiple randomized variants
    iterations = 0
    while time.time() < deadline - 20.0:
        iterations += 1
        try:
            if iterations % 3 == 0:
                g = greedy_construct(randomize=True, noise_scale=0.3, use_weights=True)
            elif iterations % 3 == 1:
                p_val = 0.01 + 0.005 * (iterations % 8)
                g = randomized_initialization(p=p_val, use_weights=True)
            else:
                # Non-weighted greedy for diversity
                g = greedy_construct(randomize=True, noise_scale=0.5, use_weights=False)
            
            if not g:
                continue

            if time_ok():
                r = redundancy_removal(g, time_limit=2.0)
                update_best(r)
            else:
                r = g
            
            remaining = deadline - time.time()
            if remaining > 15.0:
                s = swap_local_search(r, time_limit=remaining - 15.0)
                update_best(s)
                
                # Try 2-opt if time permits
                remaining2 = deadline - time.time()
                if remaining2 > 10.0:
                    s2 = two_opt_swap(s, time_limit=min(remaining2 - 10.0, 5.0))
                    update_best(s2)
        except Exception:
            pass

    # Final redundancy check on best_result
    if best_result and time.time() < deadline - 5.0:
        try:
            final_r = redundancy_removal(best_result, time_limit=3.0)
            update_best(final_r)
        except Exception:
            pass

    # CRITICAL: If best_result is still empty, run a fast greedy to ensure valid output
    if not best_result:
        uncovered = np.ones(n_t, dtype=np.int32)
        active_blocks = []
        while uncovered.sum() > 0 and time.time() < deadline - 2.0:
            gains = M @ uncovered
            best_b = int(np.argmax(gains))
            if gains[best_b] == 0:
                break
            active_blocks.append(best_b)
            start = M.indptr[best_b]
            end = M.indptr[best_b + 1]
            cols = M.indices[start:end]
            uncovered[cols] = 0
        if active_blocks:
            best_result = active_blocks

    # Construct output
    result = [list(all_k_blocks[b]) for b in best_result]
    
    if not result:
         pass

    return result
