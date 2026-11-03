import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    timeout = 140  # Leave 10s margin for 150s limit
    
    # Pre-compute all t-subsets and map to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # Pre-compute for each t-subset, the list of k-blocks that contain it
    # A k-block containing a t-subset S is S union (k-t) points from the remaining v-t points
    # We can generate these on the fly or pre-compute.
    # Pre-computing: for each t-subset, list of candidate k-blocks (as tuples)
    # Number of candidates per t-subset is C(v-t, k-t)
    # For v=19, t=5, k=8: C(14,3) = 364.
    # Total storage: 11628 * 364 ~ 4.2M entries. This is fine.
    
    # However, generating all k-blocks for each t-subset might be slow in Python.
    # Let's generate them on the fly in the inner loop? 
    # Or pre-compute the "extension" candidates.
    
    # Better: Pre-compute a list of all k-blocks? No, C(19,8) is 75k.
    # For each k-block, we can pre-compute the list of t-subset indices it covers.
    # Then, for a given uncovered t-subset S, we need to find k-blocks that cover S.
    # This requires an inverted index: t_subset_index -> list of k_block_indices.
    
    all_k_blocks = list(itertools.combinations(range(v), k))
    k_block_to_t_indices = []
    t_to_k_blocks = [[] for _ in range(num_t_subsets)]
    
    for kb in all_k_blocks:
        t_indices = []
        for ts in itertools.combinations(kb, t):
            idx = t_subset_to_idx[ts]
            t_indices.append(idx)
            t_to_k_blocks[idx].append(len(k_block_to_t_indices))
        k_block_to_t_indices.append(t_indices)
        
    best_blocks = None
    best_len = float('inf')
    
    # Run greedy multiple times with different random seeds/orders if time allows
    # But deterministic greedy is usually consistent. Let's try a few variations.
    # Variation 1: Standard Greedy (pick first uncovered, then best block)
    # Variation 2: Weighted Greedy (pick uncovered with highest "deficit")
    
    for seed in range(10): # Try up to 10 runs if fast enough
        if time.time() - start_time > timeout:
            break
            
        random.seed(seed)
        uncovered = set(range(num_t_subsets))
        blocks = []
        
        while uncovered:
            if time.time() - start_time > timeout:
                # If we timeout, we can't guarantee validity.
                # We must finish the current run or abort.
                # If we abort, best_blocks might be None or invalid.
                # We must ensure we return a valid solution.
                # So, if we are about to timeout, we should finish the current block addition
                # and then check if valid. If not, we have a problem.
                # Given the speed, we are unlikely to timeout in the middle of a run
                # unless the greedy is extremely slow.
                pass
            
            # Pick a target t-subset
            # Heuristic: Pick the one with the fewest candidate k-blocks that are "good"?
            # Or just pick a random one from uncovered to break symmetry?
            # Picking the one with the smallest number of available covering blocks
            # might be harder to cover later, so cover it now.
            
            # For simplicity and speed, pick a random one from uncovered
            target_idx = random.choice(list(uncovered))
            
            # Find best k-block covering target_idx
            candidate_k_indices = t_to_k_blocks[target_idx]
            
            best_gain = -1
            best_k_idx = -1
            
            for kb_idx in candidate_k_indices:
                gain = 0
                for ts_idx in k_block_to_t_indices[kb_idx]:
                    if ts_idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_k_idx = kb_idx
            
            # Add the block
            blocks.append(all_k_blocks[best_k_idx])
            
            # Update uncovered
            for ts_idx in k_block_to_t_indices[best_k_idx]:
                uncovered.discard(ts_idx)
                
        # End while uncovered
        
        # Check if this solution is better
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
        # If we found a solution, and time is running out, break
        if time.time() - start_time > timeout - 20:
            break

    if best_blocks is None:
        # Fallback: if we timed out before completing any run, 
        # we must return something. But the loop above only adds to best_blocks 
        # when a run completes. 
        # If the first run doesn't complete, best_blocks is None.
        # We need a safer fallback.
        # Let's re-run a single deterministic greedy without random seed if best_blocks is None
        # and ensure it finishes.
        
        uncovered = set(range(num_t_subsets))
        blocks = []
        while uncovered:
            # Deterministic: pick min index
            target_idx = min(uncovered)
            candidate_k_indices = t_to_k_blocks[target_idx]
            
            best_gain = -1
            best_k_idx = -1
            
            for kb_idx in candidate_k_indices:
                gain = 0
                for ts_idx in k_block_to_t_indices[kb_idx]:
                    if ts_idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_k_idx = kb_idx
            
            blocks.append(all_k_blocks[best_k_idx])
            for ts_idx in k_block_to_t_indices[best_k_idx]:
                uncovered.discard(ts_idx)
                
        best_blocks = blocks

    return [list(b) for b in best_blocks]
