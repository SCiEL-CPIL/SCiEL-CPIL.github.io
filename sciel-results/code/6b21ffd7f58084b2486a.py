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
    
    def greedy_construct(randomize=False, noise_scale=0.0):
        uncovered = np.ones(n_t, dtype=np.int32)
        active_blocks = []
        
        # Pre-compute initial gains? No, gains change as we pick blocks.
        # But we can use the sparse matrix multiplication.
        
        while uncovered.sum() > 0:
            if not time_ok():
                break
                
            # Calculate gains: how many uncovered subsets each block covers
            gains = M @ uncovered
            
            if randomize and noise_scale > 0:
                # Add small noise to break ties and diversify
                noise = np.random.uniform(0, 1, n_blocks) * noise_scale
                gains = gains + noise
            
            # Pick best block
            best_b = np.argmax(gains)
            
            # If max gain is 0, we might have a bug or uncovered is stale?
            # With integer counts, if uncovered.sum() > 0, there must be a block covering it?
            # Yes, every t-subset is covered by some k-blocks.
            if gains[best_b] == 0:
                break # Should not happen unless uncovered is inconsistent
            
            active_blocks.append(best_b)
            
            # Update uncovered
            start = M.indptr[best_b]
            end = M.indptr[best_b + 1]
            cols = M.indices[start:end]
            uncovered[cols] = 0
            
        return active_blocks

    def randomized_initialization(p=0.05, max_iter=100):
        """
        Try to find a small covering by randomly sampling blocks with probability p,
        then filling gaps with greedy.
        This can sometimes find smaller solutions by avoiding the greedy trap.
        """
        if not time_ok():
            return []
            
        # Sample random blocks
        # We want to sample enough to cover most things, but not too many.
        # Expected number of blocks = p * n_blocks
        # Let's try a few different p values or just one.
        
        sampled_blocks = []
        # Use numpy random for speed
        mask = np.random.random(n_blocks) < p
        sampled_indices = np.where(mask)[0]
        
        # Build coverage for sampled blocks
        # We can check if they cover everything.
        # If not, we need to add more.
        
        # Instead of building full coverage, let's just use the greedy logic
        # but start with these blocks "already selected".
        
        uncovered = np.ones(n_t, dtype=np.int32)
        active_blocks = list(sampled_indices)
        
        # Update uncovered based on sampled blocks
        if len(active_blocks) > 0:
            sub_M = M[active_blocks, :]
            # If any column sum > 0, it's covered
            covered_mask = (np.asarray(sub_M.sum(axis=0)).flatten() > 0)
            uncovered[covered_mask] = 0
            
        # Now fill gaps with greedy
        while uncovered.sum() > 0:
            if not time_ok():
                break
                
            gains = M @ uncovered
            best_b = np.argmax(gains)
            
            if gains[best_b] == 0:
                break
            
            active_blocks.append(best_b)
            
            start = M.indptr[best_b]
            end = M.indptr[best_b + 1]
            cols = M.indices[start:end]
            uncovered[cols] = 0
            
        return active_blocks

    def redundancy_removal(active_blocks, time_limit=5.0):
        # active_blocks is a list of block indices
        if not active_blocks:
            return []
            
        active_arr = np.array(active_blocks, dtype=np.int32)
        
        # Compute coverage counts for the current set
        # counts[s] = number of blocks in active_arr covering t-subset s
        # We can do this by summing rows of M corresponding to active_arr
        # M[active_arr, :] is a sparse matrix.
        # sub_M.sum(axis=0) gives column sums.
        
        # To allow multiple passes within time limit
        end_time = time.time() + time_limit
        improved = True
        
        while improved and time.time() < end_time:
            if len(active_arr) == 0:
                break
            
            sub_M = M[active_arr, :]
            counts = np.asarray(sub_M.sum(axis=0)).flatten()
            
            remove_mask = np.zeros(len(active_arr), dtype=bool)
            
            # Iterate through blocks to check redundancy
            # We do this in a fixed order first, then maybe random?
            # For speed, just iterate.
            
            for i in range(len(active_arr)):
                b_idx = active_arr[i]
                start = M.indptr[b_idx]
                end = M.indptr[b_idx + 1]
                if start == end:
                    remove_mask[i] = True
                    continue
                
                cols = M.indices[start:end]
                # If min count > 1, block is redundant
                if np.min(counts[cols]) > 1:
                    remove_mask[i] = True
            
            if not remove_mask.any():
                break
            
            # Remove redundant blocks
            active_arr = active_arr[~remove_mask]
            improved = True
            
        return list(active_arr)

    def swap_local_search(active_blocks, time_limit=10.0):
        # active_blocks is a list of block indices
        if not active_blocks:
            return []
            
        current_set = list(active_blocks)
        current_set_set = set(current_set)
        
        # Compute coverage counts
        active_arr = np.array(current_set, dtype=np.int32)
        sub_M = M[active_arr, :]
        coverage_counts = np.asarray(sub_M.sum(axis=0)).flatten()
        
        best_set = list(current_set)
        best_len = len(current_set)
        
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            changed = False
            # Shuffle order for diversity
            indices = list(range(len(current_set)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > end_time:
                    break
                if len(current_set) <= best_len:
                    # If we haven't improved yet, we might still try swaps
                    pass
                
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
                    # Block is redundant?
                    # Actually, if no unique subs, it's redundant.
                    # We should have removed it in redundancy_removal.
                    # But maybe it became redundant due to swaps.
                    # Let's just try to remove it?
                    # No, swap logic replaces it.
                    # If it has no unique subs, we can replace it with anything?
                    # Or just remove it?
                    # The prompt asks for swap. Let's try to find a swap that might allow future deletions.
                    # If we remove it, coverage_counts decreases.
                    # If we replace it, coverage_counts stays same for unique subs (none),
                    # but changes for others.
                    
                    # Let's try to replace with a random block not in set
                    # This might help escape local minima.
                    # But we need to be careful not to increase size.
                    # Swap keeps size same.
                    
                    # Pick a random candidate
                    # We want a block that covers many subsets currently covered only once by other blocks?
                    # Or just random.
                    new_block = None
                    # Try a few random ones
                    for _ in range(5):
                        cand = random.randint(0, n_blocks - 1)
                        if cand not in current_set_set and cand != old_block:
                            new_block = cand
                            break
                            
                    if new_block is None:
                        continue
                    
                    # Perform swap
                    # Update counts
                    old_cols = M.indices[M.indptr[old_block]:M.indptr[old_block + 1]]
                    new_cols = M.indices[M.indptr[new_block]:M.indptr[new_block + 1]]
                    
                    coverage_counts[old_cols] -= 1
                    coverage_counts[new_cols] += 1
                    
                    current_set[i] = new_block
                    current_set_set.remove(old_block)
                    current_set_set.add(new_block)
                    
                    # Check if this swap made some other block redundant?
                    # We'll let the deletion pass handle it.
                    # Or we can check immediately.
                    
                else:
                    # Block is critical for unique_subs
                    # Find candidates that cover all unique_subs
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
                    
                    # Pick a candidate that is not in current_set
                    # And maybe has high "gain" for other subsets?
                    # For now, pick random from candidates
                    cand_list = list(candidates)
                    random.shuffle(cand_list)
                    
                    new_block = None
                    for cand in cand_list:
                        if cand not in current_set_set:
                            new_block = cand
                            break
                            
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
                # Recompute counts
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
                    # Remove
                    to_remove = np.where(remove_mask)[0]
                    for idx in to_remove:
                        b_idx = current_set[idx]
                        cols = M.indices[M.indptr[b_idx]:M.indptr[b_idx + 1]]
                        coverage_counts[cols] -= 1
                        current_set_set.remove(b_idx)
                    
                    # Rebuild list
                    new_current_set = []
                    for i, b in enumerate(current_set):
                        if not remove_mask[i]:
                            new_current_set.append(b)
                    current_set = new_current_set
                    
                    if len(current_set) < best_len:
                        best_len = len(current_set)
                        best_set = list(current_set)
                        changed = True
                        # If we removed, we should restart the loop to find more
                        # break inner loop?
            
            if not changed and len(current_set) == best_len:
                # If no improvement in this pass, maybe break?
                # Or continue to try more swaps?
                # Let's limit iterations.
                break
            
        return best_set

    # Strategy:
    # 1. Run standard greedy
    # 2. Run randomized greedy (if time permits)
    # 3. Apply redundancy removal
    # 4. Apply swap local search
    # 5. Return best result

    best_result = []
    best_len = float('inf')

    # Helper to validate and set best
    def update_best(candidate):
        nonlocal best_result, best_len
        if candidate is None:
            return
        if len(candidate) < best_len:
            best_len = len(candidate)
            best_result = candidate

    # Phase 1: Standard Greedy (Robust baseline)
    if time_ok():
        try:
            g1 = greedy_construct(randomize=False)
            update_best(g1)
            if time_ok():
                r1 = redundancy_removal(g1, time_limit=5.0)
                update_best(r1)
        except Exception:
            pass

    # Phase 2: Randomized Initializations + Greedy Fill + Local Search
    # We have ~90s left.
    # Let's do a few iterations of Randomized Init -> Redundancy -> Swap
    
    iterations = 0
    while time.time() < deadline - 15.0:
        iterations += 1
        try:
            if iterations % 2 == 0:
                g = greedy_construct(randomize=True, noise_scale=0.5)
            else:
                # Try a few different probabilities for random sampling
                p_val = 0.01 + 0.01 * (iterations % 5)
                g = randomized_initialization(p=p_val)
            
            if not g:
                continue

            # Redundancy removal
            if time_ok():
                r = redundancy_removal(g, time_limit=3.0)
                update_best(r)
            else:
                r = g
            
            # Swap local search
            remaining = deadline - time.time()
            if remaining > 10.0:
                s = swap_local_search(r, time_limit=remaining - 10.0)
                update_best(s)
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
        # Simple greedy without complex updates, just to get *a* solution
        uncovered = np.ones(n_t, dtype=np.int32)
        active_blocks = []
        while uncovered.sum() > 0 and time.time() < deadline - 2.0:
            gains = M @ uncovered
            best_b = np.argmax(gains)
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
    # Ensure we return a list of lists of ints
    result = [list(all_k_blocks[b]) for b in best_result]
    
    # If result is still empty (should not happen), return a single block? 
    # No, that's invalid. But the logic above should prevent this.
    # To be absolutely safe, if empty, return a known valid small set? 
    # Actually, if n_t is small, we can just pick blocks.
    if not result:
         # Fallback: pick first k-blocks until covered? 
         # This is very slow. But better than empty.
         # For v=19, k=8, t=5, we know C(19,5) is small enough that a few hundred blocks work.
         # Just return a list of 1 block? No, invalid.
         # The evaluator requires valid covering.
         # If we get here, we failed. But we shouldn't.
         pass

    return result
