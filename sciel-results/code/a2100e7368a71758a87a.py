import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    # Safety margin: The evaluator kills at 150s. We should return well before.
    # Let's aim for 115s to be safe.
    deadline = start_time + 115.0 

    points = list(range(v))
    all_kblocks = list(itertools.combinations(points, k))
    all_tsubs = list(itertools.combinations(points, t))
    
    n_blocks = len(all_kblocks)
    n_tsubs = len(all_tsubs)
    
    # Map t-subsets to indices
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Precompute for each block, the list of t-subsets it covers
    # This is C(k, t) per block. C(7,4) = 35.
    # Total entries: n_blocks * 35. 50k * 35 = 1.75M entries. Manageable.
    block_tsubs = []
    for B in all_kblocks:
        covered = []
        for ts in itertools.combinations(B, t):
            covered.append(tsub_to_idx[ts])
        block_tsubs.append(covered)
        
    # Precompute for each t-subset, the list of blocks that cover it
    # Inverse of the above.
    tsub_blocks = [[] for _ in range(n_tsubs)]
    for b_idx, tsubs in enumerate(block_tsubs):
        for s_idx in tsubs:
            tsub_blocks[s_idx].append(b_idx)
            
    # Convert tsub_blocks to numpy arrays for faster indexing if needed, 
    # but lists are fine for iteration.
    
    best_solution = None
    best_score = float('inf')
    
    # Multi-restart strategy
    # Each restart: Greedy -> Local Search
    # Number of restarts depends on time available.
    # Greedy is fast. Local search is the bottleneck.
    
    num_restarts = 10 # Start with a few, increase if time permits
    
    for restart in range(num_restarts):
        if time.time() > deadline - 20.0:
            break
            
        # Phase 1: Greedy Construction with Randomization
        uncovered = np.ones(n_tsubs, dtype=bool)
        # gains[b] = number of uncovered t-subsets covered by block b
        # Initially, all blocks cover C(k,t) t-subsets.
        gains = np.full(n_blocks, len(block_tsubs[0]), dtype=np.int32)
        
        selected_indices = []
        
        while np.any(uncovered):
            if time.time() > deadline - 15.0:
                break
                
            # Randomly pick from top few to break symmetry
            # Find argmax
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            # Get all blocks with max gain
            # This can be expensive if done naively. 
            # Since gains change slowly, we can just pick argmax.
            # To add randomness, we can pick randomly from blocks with gain >= max_gain - 1
            # Or just use argmax for speed, as greedy is deterministic unless we shuffle input.
            # Shuffling input order is a common trick.
            
            # Let's use a randomized greedy: pick randomly from top 5% of gains?
            # Simpler: just pick argmax. The diversity comes from different random seeds 
            # if we were to shuffle the order of processing or breaking ties.
            # For now, let's use argmax but with a small random perturbation in gains?
            # No, that breaks the set cover logic.
            # Let's stick to argmax but rely on the local search to diversify.
            
            b_idx = np.argmax(gains)
            
            selected_indices.append(b_idx)
            
            # Update uncovered and gains
            for s_idx in block_tsubs[b_idx]:
                if uncovered[s_idx]:
                    uncovered[s_idx] = False
                    # Decrease gain for all blocks covering this t-subset
                    # This is O(n_blocks_covering_s). 
                    # Average number of blocks covering a t-subset is C(v-t, k-t) = C(15,3) = 455.
                    # So 455 updates per t-subset.
                    # 35 t-subsets per block. 35 * 455 = ~16k updates per greedy step.
                    # Greedy takes ~150 steps. 150 * 16k = 2.4M operations. Very fast.
                    for b2 in tsub_blocks[s_idx]:
                        gains[b2] -= 1
                        
        if np.any(uncovered):
            # Greedy failed to cover? Unlikely for this size.
            # If it happens, skip this restart.
            continue
            
        # Phase 2: Local Search Optimization
        # Convert to set for O(1) lookup
        selected_set = set(selected_indices)
        selected_list = list(selected_indices)
        
        # Calculate cover counts
        cover_counts = np.zeros(n_tsubs, dtype=np.int32)
        for b in selected_list:
            for s in block_tsubs[b]:
                cover_counts[s] += 1
                
        # Local Search Loop
        improved = True
        local_search_time_limit = deadline - 10.0
        
        while improved and time.time() < local_search_time_limit:
            improved = False
            
            # Step 2a: Remove Redundant Blocks
            # A block is redundant if it doesn't cover any t-subset with count 1.
            new_selected_list = []
            for b in selected_list:
                is_essential = False
                for s in block_tsubs[b]:
                    if cover_counts[s] == 1:
                        is_essential = True
                        break
                if is_essential:
                    new_selected_list.append(b)
                else:
                    # Remove
                    for s in block_tsubs[b]:
                        cover_counts[s] -= 1
                    selected_set.remove(b)
                    improved = True
                    
            selected_list = new_selected_list
            
            if not improved:
                # Step 2b: Try Swaps
                # For each block, find its unique t-subsets.
                # If we can find a block not in selected_set that covers all unique t-subsets 
                # of the current block, and adding it doesn't increase the total count 
                # (which is 1-for-1, so count stays same), we might enable further reductions.
                # Actually, a 1-for-1 swap keeps the number of blocks the same.
                # The goal is to change the structure to allow redundancy removal later.
                
                # Identify candidates for swap: blocks with minimal unique t-subsets.
                # If a block has 0 unique t-subsets, it's redundant (handled above).
                # If it has 1 unique t-subset, it's a good candidate to swap.
                
                candidates_out = []
                unique_map = {}
                
                for b in selected_list:
                    u_list = []
                    for s in block_tsubs[b]:
                        if cover_counts[s] == 1:
                            u_list.append(s)
                    if len(u_list) > 0:
                        candidates_out.append(b)
                        unique_map[b] = u_list
                        
                if not candidates_out:
                    break
                    
                # Sort candidates by number of unique t-subsets (ascending)
                candidates_out.sort(key=lambda x: len(unique_map[x]))
                
                # Try to swap the first few candidates
                swap_made = False
                for b_out in candidates_out[:5]: # Limit attempts
                    u_list = unique_map[b_out]
                    if not u_list:
                        continue
                        
                    # Find a replacement block b_in
                    # b_in must cover all s in u_list.
                    # Candidates for b_in are blocks covering u_list[0].
                    # Then filter for those covering all others.
                    
                    candidates_in = tsub_blocks[u_list[0]]
                    
                    found_in = None
                    for b_in in candidates_in:
                        if b_in in selected_set:
                            continue
                        # Check if b_in covers all u_list
                        valid = True
                        # Optimization: check the most constrained ones first?
                        # u_list is small (usually 1 or 2).
                        for s in u_list:
                            if s not in block_tsubs[b_in]:
                                valid = False
                                break
                        if valid:
                            found_in = b_in
                            break
                            
                    if found_in is not None:
                        # Perform swap
                        selected_list.remove(b_out)
                        selected_set.remove(b_out)
                        
                        for s in block_tsubs[b_out]:
                            cover_counts[s] -= 1
                            
                        selected_list.append(found_in)
                        selected_set.add(found_in)
                        
                        for s in block_tsubs[found_in]:
                            cover_counts[s] += 1
                            
                        improved = True
                        swap_made = True
                        break # Restart loop to re-evaluate redundancies
                        
                if not swap_made:
                    # Try a random perturbation to escape local minimum?
                    # Pick a random block, remove it, add a random block that covers its unique t-subsets?
                    # This is complex. Let's just break if no swap found.
                    break
                    
        # Update best solution
        current_size = len(selected_list)
        if current_size < best_score:
            best_score = current_size
            best_solution = selected_list[:]
            
        # If we found a very good solution, we might stop early?
        # Lower bound is 131. If we hit 131, we can stop.
        if best_score <= 131:
            break

    if best_solution is None:
        # Fallback: return a simple greedy solution if all else fails
        # This should not happen.
        uncovered = np.ones(n_tsubs, dtype=bool)
        gains = np.full(n_blocks, len(block_tsubs[0]), dtype=np.int32)
        selected_indices = []
        while np.any(uncovered):
            b_idx = np.argmax(gains)
            if gains[b_idx] == 0: break
            selected_indices.append(b_idx)
            for s_idx in block_tsubs[b_idx]:
                if uncovered[s_idx]:
                    uncovered[s_idx] = False
                    for b2 in tsub_blocks[s_idx]:
                        gains[b2] -= 1
        best_solution = selected_indices

    # Convert indices back to blocks
    blocks = [all_kblocks[i] for i in best_solution]
    
    return blocks
