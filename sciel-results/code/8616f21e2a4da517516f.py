import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # 1. Precomputation
    points = list(range(v))
    
    # Generate all t-subsets
    t_subsets = list(itertools.combinations(points, t))
    n_t_subsets = len(t_subsets)
    t_subset_to_idx = {sub: i for i, sub in enumerate(t_subsets)}
    
    # Generate all k-blocks
    k_blocks = list(itertools.combinations(points, k))
    n_k_blocks = len(k_blocks)
    
    # Precompute coverage for each block
    block_cover_indices = []
    for block in k_blocks:
        subs = list(itertools.combinations(block, t))
        indices = np.array([t_subset_to_idx[sub] for sub in subs], dtype=np.int32)
        block_cover_indices.append(indices)
        
    # Precompute inverse: for each t-subset, which blocks cover it
    tsub_to_blocks = [[] for _ in range(n_t_subsets)]
    for b_idx, indices in enumerate(block_cover_indices):
        for s_idx in indices:
            tsub_to_blocks[s_idx].append(b_idx)
            
    tsub_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in tsub_to_blocks]

    def greedy_construction(rng, randomized=False, top_k=10):
        uncovered = np.ones(n_t_subsets, dtype=bool)
        # Gains: number of uncovered subsets covered by each block
        # Initially, all blocks cover C(k,t) subsets
        gains = np.full(n_k_blocks, len(block_cover_indices[0]), dtype=np.int32)
        
        selected_indices = []
        
        while np.any(uncovered):
            if time.time() > deadline:
                break
                
            if randomized:
                positive_mask = gains > 0
                if not np.any(positive_mask):
                    break
                valid_blocks = np.where(positive_mask)[0]
                if len(valid_blocks) > top_k:
                    valid_gains = gains[valid_blocks]
                    top_idx = np.argpartition(valid_gains, -top_k)[-top_k:]
                    top_blocks = valid_blocks[top_idx]
                    weights = gains[top_blocks].astype(float)
                    weights /= weights.sum()
                    best_block = top_blocks[rng.choice(len(top_blocks), p=weights)]
                else:
                    best_block = valid_blocks[rng.integers(0, len(valid_blocks))]
            else:
                best_block = np.argmax(gains)
                if gains[best_block] == 0:
                    break
            
            selected_indices.append(best_block)
            
            # Update gains
            for s_idx in block_cover_indices[best_block]:
                if uncovered[s_idx]:
                    uncovered[s_idx] = False
                    # Decrement gain for all blocks covering s_idx
                    gains[tsub_to_blocks_np[s_idx]] -= 1
                    
        return selected_indices

    def local_search(selected_indices, max_rounds=50):
        if not selected_indices:
            return selected_indices
            
        selected_set = set(selected_indices)
        
        # Compute cover counts
        counts = np.zeros(n_t_subsets, dtype=np.int32)
        for b_idx in selected_indices:
            counts[block_cover_indices[b_idx]] += 1
            
        improved = True
        round_num = 0
        while improved and round_num < max_rounds and time.time() < deadline:
            improved = False
            round_num += 1
            
            # Phase A: Redundancy Removal
            new_selected = []
            for b_idx in selected_indices:
                subs = block_cover_indices[b_idx]
                if np.min(counts[subs]) > 1:
                    counts[subs] -= 1
                    improved = True
                else:
                    new_selected.append(b_idx)
            selected_indices = new_selected
            selected_set = set(selected_indices)
            
            # Phase B: Swap Moves
            # We iterate through the list. If we change a block, we update counts.
            # To avoid index issues, we iterate a copy or carefully manage indices.
            # Since we are just replacing, the length stays same (or decreases if we remove).
            # Let's do a pass of swaps.
            
            for i in range(len(selected_indices)):
                if time.time() > deadline:
                    break
                    
                b_idx = selected_indices[i]
                subs = block_cover_indices[b_idx]
                unique_mask = counts[subs] == 1
                unique_subs = subs[unique_mask]
                
                if len(unique_subs) == 0:
                    # Should have been removed in Phase A, but just in case
                    continue
                
                # Candidates must cover all unique_subs
                # Start with candidates of the first unique subset
                first_s = unique_subs[0]
                candidates = tsub_to_blocks_np[first_s]
                
                best_replacement = -1
                best_score = -1
                
                # Filter candidates
                for c in candidates:
                    if c in selected_set:
                        continue
                    
                    # Check if c covers all unique_subs
                    # This is the expensive part. We can optimize by precomputing sets?
                    # For now, check intersection.
                    # A faster check: 
                    # Create a boolean mask of unique_subs in c?
                    # Or simply:
                    covers_all = True
                    for s in unique_subs[1:]:
                        # s is in c if s is in block_cover_indices[c]
                        # Since block_cover_indices[c] is sorted, we can use searchsorted
                        # But unique_subs is small (max 35).
                        if s not in block_cover_indices[c]:
                            covers_all = False
                            break
                    
                    if not covers_all:
                        continue
                    
                    # Score the replacement
                    # We want to maximize redundancy.
                    # Score = number of subsets in c that have count > 1
                    # Or simply: Score = - (number of subsets in c that will become count 1 after swap)
                    # Let's calculate the new counts if we swap.
                    # New count for s in c: counts[s] + 1 - (1 if s in b_idx else 0)
                    # We want to minimize the number of s in c such that new_count[s] == 1
                    # And we want to maximize the number of s in b_idx \ c such that counts[s] > 1 (they become safe)
                    
                    # Simpler heuristic: Just take the first valid one.
                    # To improve, let's try a few random valid ones if we have many.
                    # But for now, let's try to pick one that maximizes overlap with existing coverage.
                    
                    # Calculate score: count of subsets in c that are already covered > 1 times
                    # This is approx counts[c_subs] > 1
                    c_subs = block_cover_indices[c]
                    score = np.sum(counts[c_subs] > 1)
                    
                    if score > best_score:
                        best_score = score
                        best_replacement = c
                        
                if best_replacement != -1:
                    # Perform swap
                    counts[block_cover_indices[b_idx]] -= 1
                    counts[block_cover_indices[best_replacement]] += 1
                    selected_indices[i] = best_replacement
                    selected_set.discard(b_idx)
                    selected_set.add(best_replacement)
                    improved = True
                    
        return selected_indices

    best_blocks = None
    best_len = float('inf')
    
    rng = np.random.default_rng(42)
    
    # Run multiple restarts
    num_restarts = 20
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Alternate between pure greedy and randomized
        if i % 2 == 0:
            selected_indices = greedy_construction(rng, randomized=False)
        else:
            selected_indices = greedy_construction(rng, randomized=True, top_k=5)
            
        if not selected_indices:
            continue
            
        refined_indices = local_search(selected_indices, max_rounds=20)
        
        # Verify
        is_valid = True
        if refined_indices:
            check_counts = np.zeros(n_t_subsets, dtype=np.int32)
            for b_idx in refined_indices:
                check_counts[block_cover_indices[b_idx]] += 1
            if np.any(check_counts == 0):
                is_valid = False
        else:
            is_valid = (n_t_subsets == 0)
            
        if is_valid and len(refined_indices) < best_len:
            best_len = len(refined_indices)
            best_blocks = [list(k_blocks[i]) for i in refined_indices]
            
            # If we find a good solution, try to polish it further
            if best_len < 140 and time.time() < deadline - 10:
                polished = local_search(refined_indices, max_rounds=50)
                if polished:
                    check_counts2 = np.zeros(n_t_subsets, dtype=np.int32)
                    for b_idx in polished:
                        check_counts2[block_cover_indices[b_idx]] += 1
                    if not np.any(check_counts2 == 0) and len(polished) < best_len:
                        best_len = len(polished)
                        best_blocks = [list(k_blocks[i]) for i in polished]

    if best_blocks is None:
        # Fallback
        selected_indices = greedy_construction(rng, randomized=False)
        if selected_indices:
            refined_indices = local_search(selected_indices, max_rounds=10)
            best_blocks = [list(k_blocks[i]) for i in refined_indices]
        else:
            best_blocks = []
            
    return best_blocks
