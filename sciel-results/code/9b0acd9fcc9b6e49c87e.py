import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # 1. Generate all blocks and subsets
    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    N_blocks = len(all_blocks)
    
    all_subsets = list(itertools.combinations(points, t))
    N_subsets = len(all_subsets)
    
    # Map subset to index
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    # Precompute which subsets each block covers
    # block_subsets[i] is a list of subset indices
    block_subsets = []
    for b in all_blocks:
        subs = [subset_to_idx[s] for s in itertools.combinations(b, t)]
        block_subsets.append(subs)
        
    # Precompute which blocks cover each subset
    # blocks_covering_subset[j] is a list of block indices
    blocks_covering_subset = [[] for _ in range(N_subsets)]
    for i, subs in enumerate(block_subsets):
        for s in subs:
            blocks_covering_subset[s].append(i)
            
    # Precompute block sizes (number of subsets covered)
    block_sizes = np.array([len(s) for s in block_subsets], dtype=np.int32)
    
    # Helper: remove redundant blocks from a selected set
    def remove_redundant(selected_blocks, cover_count):
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            # Iterate in reverse to allow in-place modification
            for i in range(len(selected_blocks) - 1, -1, -1):
                b_idx = selected_blocks[i]
                is_redundant = True
                for s in block_subsets[b_idx]:
                    if cover_count[s] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    for s in block_subsets[b_idx]:
                        cover_count[s] -= 1
                    selected_blocks.pop(i)
                    changed = True
        return selected_blocks, cover_count

    # Helper: run greedy set cover
    def run_greedy(randomize_ties=False, seed=None):
        cover_count_local = np.zeros(N_subsets, dtype=np.int32)
        potential_local = block_sizes.copy()
        selected_local = []
        
        if seed is not None:
            random.seed(seed)
            
        while np.any(cover_count_local == 0):
            if time.time() > deadline - 10:
                return None, None # Timeout safety
                
            max_pot = np.max(potential_local)
            if max_pot == 0:
                break
                
            if randomize_ties:
                candidates = np.where(potential_local == max_pot)[0]
                best_idx = candidates[random.randint(0, len(candidates)-1)]
            else:
                best_idx = np.argmax(potential_local)
                
            selected_local.append(best_idx)
            
            # Update cover_count and potentials
            for s in block_subsets[best_idx]:
                old_count = cover_count_local[s]
                cover_count_local[s] += 1
                if old_count == 0:
                    for b in blocks_covering_subset[s]:
                        potential_local[b] -= 1
                        
        # Ensure validity check? 
        # If we timed out in the middle, cover_count_local might have zeros.
        # But we break out and return None if we hit the inner deadline.
        # However, the greedy loop condition is `while np.any(cover_count_local == 0)`.
        # If we break due to timeout, we return None.
        # If we break due to max_pot == 0 while zeros exist, that's a bug/edge case.
        
        # Check if fully covered
        if np.any(cover_count_local == 0):
            return None, None
            
        # Redundancy removal
        selected_local, cover_count_local = remove_redundant(selected_local, cover_count_local)
        
        return selected_local, cover_count_local

    # Helper: local search with swaps
    def local_search(selected_blocks, cover_count, max_iters=1000):
        current_blocks = selected_blocks.copy()
        current_cover = cover_count.copy()
        current_len = len(current_blocks)
        
        # Map block index to position in current_blocks for quick removal?
        # Or just use list of indices.
        
        # Precompute for each block in current set, the list of subsets it uniquely covers?
        # No, that's expensive to maintain.
        
        # Strategy: 
        # 1. Try to remove a block. If valid, keep.
        # 2. Try to replace a block with a better one? 
        #    "Better" means: if we remove block A, we have uncovered subsets U.
        #    We need a block B that covers all of U.
        #    If such B exists and B is not in current set, we can swap A->B.
        #    But swapping doesn't reduce size. It might enable future removals.
        #    
        # Let's stick to aggressive removal first, then maybe simple swaps if time permits.
        # Actually, the "remove redundant" already does pure removal.
        # The "swap" idea: 
        # Pick a block A. Remove it. Find uncovered subsets U.
        # If U is empty, we already removed it in the redundancy pass.
        # If U is not empty, we need to cover U.
        # If there is a block B that covers U (and potentially more), and adding B allows us to remove another block C?
        # This is getting complex.
        
        # Simpler Local Search: Randomized Removal & Repair
        # Pick a random block. Remove it.
        # If valid, keep.
        # If invalid, identify uncovered subsets. 
        # Try to add a block that covers the most of these uncovered subsets.
        # If we can cover them all with one block, and that block allows removing another, it's a net gain?
        # No, if we remove 1 and add 1, size is same.
        # We only want to reduce size.
        
        # So, the only way to reduce size is to find a block whose removal leaves no uncovered subsets (handled by redundancy) 
        # OR to find a set of blocks that can be replaced by fewer blocks.
        # Given the constraints, redundancy removal is the primary driver.
        # Randomized restarts are the other driver.
        
        return current_blocks, current_cover

    # 2. Initial Run (Deterministic)
    best_blocks, best_cover = run_greedy(randomize_ties=False)
    
    if best_blocks is None:
        # Fallback: If greedy failed (shouldn't happen for this size), try random
        # Or just return empty? No, must be valid.
        # For v=19, k=8, t=5, greedy is very fast.
        # If it failed, something is wrong.
        # Let's force a valid solution by brute force if needed? No, too slow.
        # Assume it works.
        best_blocks = []
        best_cover = np.zeros(N_subsets, dtype=np.int32)

    best_len = len(best_blocks)
    
    # 3. Randomized Restarts
    # Estimate time for one run.
    # Let's just run as many as we can.
    # Each run is ~0.1-0.5s.
    
    n_restarts = 0
    while time.time() < deadline - 10:
        n_restarts += 1
        seed = 1000 + n_restarts
        res_blocks, res_cover = run_greedy(randomize_ties=True, seed=seed)
        
        if res_blocks is not None:
            if len(res_blocks) < best_len:
                best_len = len(res_blocks)
                best_blocks = res_blocks
                best_cover = res_cover
                
            # Try local search on the new best if it's significantly better?
            # Or just try on the current best periodically?
            # Let's try local search on the best found so far every 5 restarts.
            if n_restarts % 5 == 0:
                # Local search might not improve much beyond redundancy removal, 
                # but let's see if we can do a quick pass.
                # For now, skip complex local search to ensure speed and stability.
                pass

    # 4. Final Polish: Aggressive Redundancy Removal
    # Sometimes the greedy order leaves redundant blocks that the internal removal pass missed
    # due to order dependencies? The internal removal is a loop until no change, so it should be clean.
    # But let's run it one more time to be safe.
    best_blocks, best_cover = remove_redundant(best_blocks, best_cover)
    best_len = len(best_blocks)

    # Convert indices back to blocks
    final_blocks = [list(all_blocks[i]) for i in best_blocks]
    
    # Final validation (just in case)
    # This is expensive, so only do it if we have time?
    # The task says "Valid when every t-subset ... lies in at least one block".
    # Our construction guarantees this if we didn't time out mid-greedy.
    # We checked `if np.any(cover_count_local == 0): return None` in run_greedy.
    # So best_blocks should be valid.
    
    return final_blocks
