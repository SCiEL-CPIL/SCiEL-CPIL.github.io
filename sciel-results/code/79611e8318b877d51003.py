import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    # 1. Precomputation
    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    N_blocks = len(all_blocks)
    
    all_subsets = list(itertools.combinations(points, t))
    N_subsets = len(all_subsets)
    
    # Map subset to index
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    # Precompute for each block:
    # 1. List of subset indices it covers
    # 2. Bitmask of subset indices it covers (Python int)
    block_subsets = []
    block_bitmasks = []
    
    # For fast candidate lookup: for each subset, which blocks cover it?
    # This is a list of lists. blocks_covering_subset[s] = [b1, b2, ...]
    blocks_covering_subset = [[] for _ in range(N_subsets)]
    
    for i, b in enumerate(all_blocks):
        subs = [subset_to_idx[s] for s in itertools.combinations(b, t)]
        block_subsets.append(subs)
        
        mask = 0
        for s in subs:
            mask |= (1 << s)
            blocks_covering_subset[s].append(i)
        block_bitmasks.append(mask)
        
    # Convert block_subsets to numpy arrays for faster indexing if needed, 
    # but lists are fine for the local search logic if we use indices.
    # For greedy, we need vectorized gains.
    
    # Create a sparse-like structure for greedy gains calculation
    # Actually, for N_blocks=75582 and N_subsets=11628, a dense boolean matrix 
    # is 75582 * 11628 * 1 byte = ~878 MB. Too large.
    # We must use the list structure or a sparse matrix.
    # Let's stick to the list structure for gains but update it smartly.
    # Or, use the fact that we only need to update gains for blocks that 
    # cover the newly covered subsets.
    
    # Alternative for Greedy:
    # Maintain `uncovered` as a boolean numpy array of size N_subsets.
    # To get gains, we need to count how many uncovered subsets each block covers.
    # This is a matrix-vector product if we had the matrix.
    # Since we don't have the dense matrix, we can maintain `potential` array 
    # as in the seed, but update it efficiently.
    
    # Let's use the potential array approach from the seed, which is O(1) update per new subset.
    
    def run_greedy(seed=0):
        rng = random.Random(seed)
        cover_count = np.zeros(N_subsets, dtype=np.int32)
        potential = np.array([len(s) for s in block_subsets], dtype=np.int32)
        selected_blocks = []
        selected_set = set()
        
        # We need to know which subsets are uncovered to check if we are done.
        # We can just check if any potential > 0? No, potential is total uncovered.
        # If potential > 0, there is at least one block covering at least one uncovered subset.
        # But we need to stop when NO uncovered subsets remain.
        # We can track `uncovered_count` or just check `cover_count` min.
        # Checking `np.any(cover_count == 0)` is O(N_subsets). N_subsets=11628. This is fast.
        
        while np.any(cover_count == 0):
            if time.time() > deadline - 5.0:
                return None, None
                
            max_pot = np.max(potential)
            if max_pot == 0:
                # This should not happen if cover_count has 0s, unless logic error
                break
                
            if seed == 0:
                best_idx = np.argmax(potential)
            else:
                candidates = np.where(potential == max_pot)[0]
                best_idx = candidates[rng.randint(0, len(candidates)-1)]
                
            selected_blocks.append(best_idx)
            selected_set.add(best_idx)
            
            # Update cover_count and potential
            for s in block_subsets[best_idx]:
                old_count = cover_count[s]
                cover_count[s] += 1
                if old_count == 0:
                    # This subset was just covered. Decrease potential of all blocks covering it.
                    for b in blocks_covering_subset[s]:
                        potential[b] -= 1
                        
        return selected_blocks, cover_count

    def local_search(selected_blocks, cover_count):
        # Make copies to avoid modifying original
        selected_blocks = list(selected_blocks)
        cover_count = cover_count.copy()
        selected_set = set(selected_blocks)
        
        best_len = len(selected_blocks)
        best_blocks = list(selected_blocks)
        
        steps = 0
        max_steps = 100000 # High limit, time check will stop it
        no_improve_count = 0
        max_no_improve = 5000
        
        while steps < max_steps:
            steps += 1
            if time.time() > deadline - 2.0:
                break
            if not selected_blocks:
                break
                
            # 1. Try to remove redundant blocks
            # We can do a full pass or random check. Full pass is safer for finding all redundancies.
            # But it's O(N_selected * C(k,t)). 
            # Let's do a random deletion attempt first, it's faster.
            # Actually, the seed did a full redundancy removal loop. 
            # Let's stick to random swap/deletion for speed, but ensure we catch obvious redundancies.
            
            # Randomly pick a block to try to remove/swap
            b_out_idx = random.randint(0, len(selected_blocks) - 1)
            b_out = selected_blocks[b_out_idx]
            
            # Find subsets covered ONLY by b_out
            critical_subsets = []
            for s in block_subsets[b_out]:
                if cover_count[s] == 1:
                    critical_subsets.append(s)
            
            if not critical_subsets:
                # Redundant, just remove
                for s in block_subsets[b_out]:
                    cover_count[s] -= 1
                selected_blocks.pop(b_out_idx)
                selected_set.remove(b_out)
                
                if len(selected_blocks) < best_len:
                    best_len = len(selected_blocks)
                    best_blocks = list(selected_blocks)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # Need to find a block that covers all critical_subsets
                # Use bitset check
                critical_mask = 0
                for s in critical_subsets:
                    critical_mask |= (1 << s)
                
                # Candidate blocks must cover the first critical subset
                first_crit = critical_subsets[0]
                candidates = blocks_covering_subset[first_crit]
                
                found_swap = False
                # To avoid bias, we can shuffle a subset or just iterate.
                # Iterating all candidates might be slow if many.
                # Let's check a random sample or all if small.
                # For efficiency, let's just iterate. The list size is manageable.
                
                for b_in in candidates:
                    if b_in in selected_set:
                        continue
                    
                    # Check if b_in covers all critical subsets
                    if (block_bitmasks[b_in] & critical_mask) == critical_mask:
                        # Perform swap
                        # Remove b_out
                        for s in block_subsets[b_out]:
                            cover_count[s] -= 1
                        selected_blocks.pop(b_out_idx)
                        selected_set.remove(b_out)
                        
                        # Add b_in
                        selected_blocks.append(b_in)
                        selected_set.add(b_in)
                        for s in block_subsets[b_in]:
                            cover_count[s] += 1
                        
                        found_swap = True
                        break
                
                if found_swap:
                    # Try to remove redundant blocks after swap
                    # This is crucial to gain from the swap
                    changed = True
                    while changed:
                        changed = False
                        # Iterate backwards to safely pop
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
                                selected_set.remove(b_idx)
                                changed = True
                                if len(selected_blocks) < best_len:
                                    best_len = len(selected_blocks)
                                    best_blocks = list(selected_blocks)
                                    no_improve_count = 0
                            
                    if len(selected_blocks) >= best_len:
                        no_improve_count += 1
                    else:
                        no_improve_count = 0
                else:
                    no_improve_count += 1
                    
            if no_improve_count > max_no_improve:
                break
                
        return best_blocks, best_len

    # Main execution
    best_blocks_global = None
    best_len_global = float('inf')
    
    # 1. Deterministic greedy
    blocks, counts = run_greedy(0)
    if blocks:
        best_blocks_local, best_len_local = local_search(blocks, counts)
        if best_len_local < best_len_global:
            best_len_global = best_len_local
            best_blocks_global = best_blocks_local
            
    # 2. Randomized greedy restarts
    # We have ~145s. Greedy is fast. Local search is the bottleneck.
    # Let's run as many restarts as we can, with a time limit on local search per restart.
    # Or, run greedy, then local search with a shared deadline.
    
    for seed in range(1, 20): # Try up to 20 restarts
        if time.time() > deadline - 10.0:
            break
            
        blocks, counts = run_greedy(seed)
        if not blocks:
            continue
            
        # Run local search. It will stop if time is low.
        best_blocks_local, best_len_local = local_search(blocks, counts)
        
        if best_len_local < best_len_global:
            best_len_global = best_len_local
            best_blocks_global = best_blocks_local

    # If we still have time, do a final polish on the best global solution
    if best_blocks_global and time.time() < deadline - 5.0:
        # Reconstruct cover_count for the best global blocks
        cover_count_final = np.zeros(N_subsets, dtype=np.int32)
        for b in best_blocks_global:
            for s in block_subsets[b]:
                cover_count_final[s] += 1
        
        # Run local search again, potentially finding more improvements
        best_blocks_final, best_len_final = local_search(list(best_blocks_global), cover_count_final)
        if best_len_final < best_len_global:
            best_len_global = best_len_final
            best_blocks_global = best_blocks_final

    if not best_blocks_global:
        # Fallback: return the first greedy result if something went wrong
        blocks, _ = run_greedy(0)
        best_blocks_global = blocks

    return [list(all_blocks[i]) for i in best_blocks_global]
