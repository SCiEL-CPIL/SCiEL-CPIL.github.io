import itertools
import time
import numpy as np
import scipy.sparse
import random

def construct(v, k, t):
    start_time = time.time()
    # Safety margin: 150s limit, aim for 140s to be safe
    deadline = start_time + 140.0 

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
    # Weight for each t-subset: inverse of degree
    subset_weights = 1.0 / np.maximum(col_degrees, 1.0)
    
    # Convert block_to_subsets to numpy arrays for faster indexing in local search
    block_to_subsets_np = [np.array(subs, dtype=np.int32) if subs else np.array([], dtype=np.int32) for subs in block_to_subsets]
    # Convert subset_to_blocks to sets for fast intersection in swap logic
    subset_to_blocks_sets = [set(blocks) for blocks in subset_to_blocks]

    def greedy_construct(randomize=False, noise_scale=0.0, use_weights=True):
        uncovered = np.ones(n_t, dtype=np.int32)
        active_blocks = []
        
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
            
            if randomize and noise_scale > 0:
                noise = np.random.uniform(0, 1, n_blocks) * noise_scale
                gains = gains + noise
            
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

    def randomized_initialization(p=0.05, use_weights=True):
        if not time_ok():
            return []
            
        mask = np.random.random(n_blocks) < p
        sampled_indices = np.where(mask)[0]
        
        uncovered = np.ones(n_t, dtype=np.int32)
        active_blocks = list(sampled_indices)
        
        if len(active_blocks) > 0:
            # Efficiently update uncovered for initial random set
            # Instead of summing, just mark them
            for b in active_blocks:
                cols = M.indices[M.indptr[b]:M.indptr[b + 1]]
                uncovered[cols] = 0
        
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
        """
        Fast redundancy removal using explicit coverage counts.
        """
        if not active_blocks:
            return []
            
        current_set = list(active_blocks)
        current_set_set = set(current_set)
        
        # Initialize coverage counts
        coverage_counts = np.zeros(n_t, dtype=np.int32)
        for b in current_set:
            cols = M.indices[M.indptr[b]:M.indptr[b + 1]]
            coverage_counts[cols] += 1
            
        end_time = time.time() + time_limit
        improved = True
        
        while improved and time.time() < end_time:
            if len(current_set) == 0:
                break
            
            improved = False
            # Shuffle indices for randomization
            indices = list(range(len(current_set)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > end_time:
                    break
                if i >= len(current_set):
                    continue
                
                b_idx = current_set[i]
                cols = M.indices[M.indptr[b_idx]:M.indptr[b_idx + 1]]
                
                # Check if block is redundant: all its subsets are covered by >1 block
                if len(cols) == 0 or np.min(coverage_counts[cols]) > 1:
                    # Remove block
                    coverage_counts[cols] -= 1
                    current_set_set.discard(b_idx)
                    current_set.pop(i)
                    improved = True
                    # Restart scan to be safe, or continue if we are careful with indices
                    # Restarting is safer for correctness
                    break
            
        return current_set

    def swap_local_search(active_blocks, time_limit=10.0):
        """
        Local search with swaps and deletions.
        Uses explicit coverage counts for O(1) updates.
        """
        if not active_blocks:
            return []
            
        current_set = list(active_blocks)
        current_set_set = set(current_set)
        
        coverage_counts = np.zeros(n_t, dtype=np.int32)
        for b in current_set:
            cols = M.indices[M.indptr[b]:M.indptr[b + 1]]
            coverage_counts[cols] += 1
            
        best_set = list(current_set)
        best_len = len(current_set)
        
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            changed = False
            
            # 1. Deletion Pass
            indices = list(range(len(current_set)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > end_time:
                    break
                if i >= len(current_set):
                    continue
                
                b_idx = current_set[i]
                cols = M.indices[M.indptr[b_idx]:M.indptr[b_idx + 1]]
                
                if len(cols) == 0 or np.min(coverage_counts[cols]) > 1:
                    # Remove
                    coverage_counts[cols] -= 1
                    current_set_set.discard(b_idx)
                    current_set.pop(i)
                    changed = True
                    if len(current_set) < best_len:
                        best_len = len(current_set)
                        best_set = list(current_set)
                    # Restart deletion scan
                    break
            
            if time.time() > end_time:
                break
            
            # 2. Swap Pass
            indices = list(range(len(current_set)))
            random.shuffle(indices)
            
            swaps_done = 0
            max_swaps = 50
            
            for i in indices:
                if time.time() > end_time:
                    break
                if swaps_done >= max_swaps:
                    break
                if i >= len(current_set):
                    continue
                    
                old_block = current_set[i]
                old_cols = M.indices[M.indptr[old_block]:M.indptr[old_block + 1]]
                
                # Find unique subsets covered by old_block
                unique_subs = []
                for s in old_cols:
                    if coverage_counts[s] == 1:
                        unique_subs.append(s)
                
                if not unique_subs:
                    # Block is not critical. Try swapping with a random block to diversify
                    # This helps escape local minima
                    new_block = random.randint(0, n_blocks - 1)
                    if new_block == old_block or new_block in current_set_set:
                        continue
                    
                    new_cols = M.indices[M.indptr[new_block]:M.indptr[new_block + 1]]
                    
                    # Perform swap
                    coverage_counts[old_cols] -= 1
                    coverage_counts[new_cols] += 1
                    
                    current_set_set.discard(old_block)
                    current_set_set.add(new_block)
                    current_set[i] = new_block
                    swaps_done += 1
                    changed = True
                else:
                    # Block is critical. Find candidates that cover all unique_subs
                    candidates = None
                    for s in unique_subs:
                        if candidates is None:
                            candidates = set(subset_to_blocks_sets[s])
                        else:
                            candidates.intersection_update(subset_to_blocks_sets[s])
                            if not candidates:
                                break
                                
                    if not candidates:
                        continue
                    
                    candidates.discard(old_block)
                    if not candidates:
                        continue
                    
                    cand_list = list(candidates)
                    random.shuffle(cand_list)
                    
                    new_block = None
                    best_score = -1
                    # Limit candidates for speed
                    for cand in cand_list[:20]:
                        if cand in current_set_set:
                            continue
                        # Score: number of subsets this block covers that are currently covered
                        # (to maintain coverage) 
                        # We want a block that covers the unique subsets AND potentially others
                        # but doesn't introduce too much new redundancy.
                        # Simple score: count of overlaps with current coverage
                        cand_cols = M.indices[M.indptr[cand]:M.indptr[cand + 1]]
                        # If it covers any unique sub, it's valid. 
                        # We prefer blocks that cover more already-covered subsets to keep the set tight?
                        # Or blocks that cover more *uncovered* subsets? 
                        # Since we are swapping, we must cover the unique ones.
                        # Let's pick the one with highest overlap with current set to minimize "spread"
                        score = np.sum(coverage_counts[cand_cols] > 0)
                        if score > best_score:
                            best_score = score
                            new_block = cand
                            
                    if new_block is None:
                        continue
                    
                    # Perform swap
                    new_cols = M.indices[M.indptr[new_block]:M.indptr[new_block + 1]]
                    
                    coverage_counts[old_cols] -= 1
                    coverage_counts[new_cols] += 1
                    
                    current_set_set.discard(old_block)
                    current_set_set.add(new_block)
                    current_set[i] = new_block
                    swaps_done += 1
                    changed = True
            
            if not changed:
                break
            
        return best_set

    def two_opt_swap(active_blocks, time_limit=5.0):
        """
        Try to swap two blocks for two different blocks, reducing total count.
        """
        if not active_blocks or len(active_blocks) < 2:
            return list(active_blocks)
        
        current_set = list(active_blocks)
        current_set_set = set(current_set)
        
        coverage_counts = np.zeros(n_t, dtype=np.int32)
        for b in current_set:
            cols = M.indices[M.indptr[b]:M.indptr[b + 1]]
            coverage_counts[cols] += 1
            
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
                    # We handle redundancy in the main loop, so skip here to avoid complexity
                    continue
                
                # Find two blocks that cover all unique_combined
                # Heuristic: Pick one candidate that covers many unique_combined, 
                # then find another to cover the rest.
                
                # Find candidates for b1 that cover some of unique_combined
                # This is complex, so we use a simpler random search limited by time
                
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
                    # Efficient check:
                    # Create a set of unique_combined
                    # Check if unique_combined is subset of (cols_c1 U cols_c2)
                    # Since unique_combined is small, we can iterate it
                    valid = True
                    for s in unique_combined:
                        # Check if s is in cols_c1 or cols_c2
                        # M is sparse, checking membership is O(1) if we had a set, but we have arrays.
                        # For small unique_combined, linear scan or set conversion is fine.
                        # Let's use a set for cols_c1 and cols_c2 for this check
                        # Optimization: precompute sets? No, too much memory.
                        # Just check:
                        if s not in cols_c1 and s not in cols_c2:
                            valid = False
                            break
                    
                    if valid:
                        # Valid swap!
                        # Update counts
                        coverage_counts[cols_b1] -= 1
                        coverage_counts[cols_b2] -= 1
                        coverage_counts[cols_c1] += 1
                        coverage_counts[cols_c2] += 1
                        
                        current_set[i] = c1
                        current_set[j] = c2
                        current_set_set.discard(b1)
                        current_set_set.discard(b2)
                        current_set_set.add(c1)
                        current_set_set.add(c2)
                        
                        improved = True
                        # Now try to delete redundant blocks
                        # Quick pass
                        indices = list(range(len(current_set)))
                        random.shuffle(indices)
                        for idx in indices:
                            if time.time() > end_time:
                                break
                            if idx >= len(current_set):
                                continue
                            b_idx = current_set[idx]
                            cols = M.indices[M.indptr[b_idx]:M.indptr[b_idx + 1]]
                            if len(cols) == 0 or np.min(coverage_counts[cols]) > 1:
                                coverage_counts[cols] -= 1
                                current_set_set.discard(b_idx)
                                current_set.pop(idx)
                                n_set = len(current_set)
                                if len(current_set) < best_len:
                                    best_len = len(current_set)
                                    best_set = list(current_set)
                                break # Restart deletion
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
