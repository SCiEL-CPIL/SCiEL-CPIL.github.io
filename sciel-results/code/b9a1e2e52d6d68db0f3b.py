import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115
    
    # Precompute all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    
    # Precompute all possible blocks (k-subsets)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Build mapping from t-subset to index
    ts_to_idx = {ts: i for i, ts in enumerate(all_t_subs)}
    
    # For each block, store the list of t-subset indices it covers
    block_cover_indices = []
    for block in all_blocks:
        indices = []
        for ts in itertools.combinations(block, t):
            indices.append(ts_to_idx[ts])
        block_cover_indices.append(indices)
    
    # Build numpy matrix: M[block, t_subset] = 1 if block covers t_subset
    M = np.zeros((n_blocks, n_t), dtype=np.uint8)
    for bi, indices in enumerate(block_cover_indices):
        for idx in indices:
            M[bi, idx] = 1
    
    # Greedy set cover with multiple restarts + local deletion refinement
    best_solution = None
    best_count = float('inf')
    
    def local_deletion(blocks_indices, M, n_t, deadline, max_iter=50):
        """Try to remove redundant blocks from the solution."""
        if not blocks_indices:
            return blocks_indices
        # Build coverage count: for each t-subset, how many blocks cover it
        cover_count = np.zeros(n_t, dtype=np.int32)
        for bi in blocks_indices:
            cover_count[M[bi]] += 1
        
        current = list(blocks_indices)
        improved = True
        iters = 0
        while improved and iters < max_iter and time.time() < deadline - 2:
            improved = False
            iters += 1
            for i in range(len(current)):
                if time.time() > deadline - 2:
                    break
                bi = current[i]
                # Check if removing block bi leaves any t-subset uncovered
                # A t-subset becomes uncovered if cover_count[ts] == 1 and block bi covers it
                # i.e., if there exists ts where M[bi, ts] == 1 and cover_count[ts] == 1
                if not np.any(M[bi] & (cover_count == 1)):
                    # Block bi is redundant
                    cover_count[M[bi]] -= 1
                    current.pop(i)
                    improved = True
                    break  # Restart scan from beginning
        return current
    
    # Number of greedy restarts to try
    max_restarts = 30
    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            break
        
        if restart == 0:
            rng = None
        else:
            rng = random.Random(restart * 7919 + 12345)
        
        # Track remaining uncovered count for each t-subset
        rem = np.ones(n_t, dtype=np.int32)
        
        selected_blocks = []
        
        while True:
            remaining = int(rem.sum())
            if remaining == 0:
                break
            
            if time.time() > deadline - 5:
                break
            
            # Compute gain for each block: number of currently uncovered t-subsets it covers
            # gain[bi] = sum(rem * M[bi, :])
            gains = M @ rem  # shape (n_blocks,)
            
            best_gain = int(gains.max())
            if best_gain <= 0:
                break
            
            # Find blocks with best gain
            best_mask = (gains == best_gain)
            best_indices = np.where(best_mask)[0]
            
            if rng is None or best_gain <= 2 or len(best_indices) <= 3:
                chosen_block = int(best_indices[0])
            else:
                # Pick randomly from top candidates, biased towards better
                # Use weighted random selection
                weights = np.ones(len(best_indices), dtype=np.float64)
                r = rng.random()
                # Simple: pick uniformly from best for diversity
                chosen_block = int(best_indices[int(r * len(best_indices))])
            
            selected_blocks.append(chosen_block)
            
            # Update remaining counts
            rem -= M[chosen_block]
            rem = np.maximum(rem, 0)
        
        if len(selected_blocks) == 0:
            continue
            
        # Apply local deletion to refine
        refined = local_deletion(selected_blocks, M, n_t, deadline, max_iter=30)
        
        if len(refined) < best_count:
            best_count = len(refined)
            best_solution = refined.copy()
        
        # Check if we have a valid solution
        if int(rem.sum()) == 0:
            # Valid solution found - do one more refinement pass
            refined2 = local_deletion(refined, M, n_t, deadline, max_iter=50)
            if len(refined2) < best_count:
                best_count = len(refined2)
                best_solution = refined2.copy()
            break
    
    # If we still have time, do additional refinement on best solution
    if best_solution is not None and time.time() < deadline - 3:
        final_refined = local_deletion(best_solution, M, n_t, deadline, max_iter=100)
        if len(final_refined) < best_count:
            best_count = len(final_refined)
            best_solution = final_refined
    
    # Fallback if no solution found
    if best_solution is None:
        # Simple greedy as fallback
        rem = np.ones(n_t, dtype=np.int32)
        selected_blocks = []
        while int(rem.sum()) > 0 and time.time() < deadline - 1:
            gains = M @ rem
            best_gain = int(gains.max())
            if best_gain <= 0:
                break
            chosen = int(np.argmax(gains))
            selected_blocks.append(chosen)
            rem -= M[chosen]
            rem = np.maximum(rem, 0)
        best_solution = selected_blocks
        best_count = len(selected_blocks)
    
    # Return the blocks as lists
    result = [list(all_blocks[bi]) for bi in best_solution]
    return result
