import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Precomputation
    # 1. All triples
    triples = list(itertools.combinations(range(v), t))
    num_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    # 2. All blocks and their masks
    blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(blocks)
    
    # Precompute masks for all blocks
    # This is the most expensive part.
    # Optimization: Generate blocks and compute mask on the fly?
    # No, we need to scan them multiple times in Greedy.
    # Let's precompute.
    
    # To speed up precomputation, we can use a helper to get mask of a block
    # But doing it in a loop for 346k blocks might take ~5-10s.
    
    block_masks = [0] * num_blocks
    # We can parallelize? No, standard library only.
    # We can optimize the inner loop.
    
    # For each block, find indices of its t-subsets
    # Precompute t-subsets for all possible k-subsets? No.
    
    # Let's just loop.
    # 346,104 blocks.
    # Inner loop: 35 triples per block.
    # 12 million iterations.
    
    for i, block in enumerate(blocks):
        mask = 0
        for triple in itertools.combinations(block, t):
            idx = triple_to_idx[tuple(triple)]
            mask |= (1 << idx)
        block_masks[i] = mask
        
    # Check time
    if time.time() > deadline - 10:
        # Fallback to simple greedy if precomp took too long
        pass

    # Greedy Construction with Randomization
    def greedy_construct(seed):
        random.seed(seed)
        uncovered_mask = (1 << num_triples) - 1
        selected_indices = []
        
        while uncovered_mask:
            if time.time() > deadline - 5:
                break
                
            best_score = -1
            candidates = []
            
            # Find all blocks with max gain
            # To allow randomization, we collect all max-score blocks
            # and pick one randomly.
            
            for i in range(num_blocks):
                m = block_masks[i]
                if m & uncovered_mask:
                    score = (m & uncovered_mask).bit_count()
                    if score > best_score:
                        best_score = score
                        candidates = [i]
                    elif score == best_score:
                        candidates.append(i)
            
            if not candidates:
                break
                
            # Randomly pick from candidates
            # If many candidates, this list can be large.
            # To save memory/time, we can just pick the first one if we don't want randomness,
            # but randomness helps escape local optima.
            # However, building a list of all max-score blocks is O(N).
            # We are already doing O(N) for the scan.
            
            chosen_idx = random.choice(candidates)
            selected_indices.append(chosen_idx)
            uncovered_mask &= ~block_masks[chosen_idx]
            
        return selected_indices

    def delete_redundant(indices):
        if not indices:
            return indices
            
        # Calculate union of all
        total_mask = 0
        for idx in indices:
            total_mask |= block_masks[idx]
            
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            new_indices = []
            for idx in indices:
                # Check if redundant
                # other_mask = total_mask ^ block_masks[idx] ? No.
                # We need to know if block_masks[idx] is covered by the rest.
                # Since total_mask is the union, if we remove idx, the new union is
                # total_mask & ~block_masks[idx] ONLY IF no other block covers any bit in block_masks[idx] that is unique to it?
                # Actually, simpler:
                # Compute union of all others.
                # This is O(N^2) if done naively.
                # N is ~70. 70^2 = 4900. Very fast.
                
                other_mask = 0
                for j in indices:
                    if j != idx:
                        other_mask |= block_masks[j]
                        
                if (other_mask & block_masks[idx]) == block_masks[idx]:
                    changed = True
                    continue # Remove
                else:
                    new_indices.append(idx)
            
            indices = new_indices
        return indices

    def local_search_swap(indices, max_passes=3):
        # Try to swap blocks to enable further deletion
        # This is expensive.
        # For each block b in indices:
        #   Find unique triples covered by b.
        #   If none, delete (handled by delete_redundant).
        #   If some, find a block b' not in indices that covers them.
        #   If found, swap b with b'.
        #   Run delete_redundant.
        #   If size reduced, keep. Else revert.
        
        current_indices = list(indices)
        current_size = len(current_indices)
        
        for _ in range(max_passes):
            if time.time() > deadline - 2:
                break
                
            improved = False
            new_indices = list(current_indices)
            
            # Iterate over blocks to replace
            for i, idx in enumerate(current_indices):
                if time.time() > deadline - 2:
                    break
                    
                mask_idx = block_masks[idx]
                
                # Find unique coverage of this block
                # Union of all others
                other_mask = 0
                for j, jdx in enumerate(current_indices):
                    if j != i:
                        other_mask |= block_masks[jdx]
                
                unique_mask = mask_idx & ~other_mask
                
                if unique_mask == 0:
                    # Redundant, delete_redundant should have caught it, but just in case
                    new_indices.pop(i)
                    improved = True
                    break # Restart loop with new set
                else:
                    # Need to find a block b' that covers unique_mask
                    # b' must not be in current_indices
                    # To speed up, we can check all blocks
                    # This is O(N_blocks) per block in set.
                    # 70 * 346k = 24M checks.
                    # Might be too slow if done many times.
                    # Let's limit this.
                    
                    # Optimization: Only try if we have time.
                    if time.time() > deadline - 10:
                        break
                        
                    found_swap = -1
                    # We can break early if we find one
                    for k_idx in range(num_blocks):
                        if k_idx in current_indices:
                            continue
                        if (block_masks[k_idx] & unique_mask) == unique_mask:
                            found_swap = k_idx
                            break
                            
                    if found_swap != -1:
                        # Perform swap
                        new_indices[i] = found_swap
                        # Try to delete redundants
                        new_indices = delete_redundant(new_indices)
                        if len(new_indices) < current_size:
                            current_indices = new_indices
                            current_size = len(current_indices)
                            improved = True
                            break # Restart
            if not improved:
                break
                
        return current_indices

    # Main Loop
    best_solution = None
    best_size = float('inf')
    
    # Run multiple restarts
    restart_count = 0
    while time.time() < deadline - 15:
        seed = restart_count
        sol = greedy_construct(seed)
        sol = delete_redundant(sol)
        sol = local_search_swap(sol)
        
        if len(sol) < best_size:
            best_size = len(sol)
            best_solution = sol
            # If we hit the known lower bound or best known, we can stop early?
            # Best known is 76. Lower bound 69.
            if best_size <= 76:
                # We found a competitive solution.
                # We can try to optimize further or break.
                # Let's break to be safe and return this.
                break
                
        restart_count += 1
        
    if best_solution is None:
        # Fallback: Simple greedy
        sol = greedy_construct(0)
        sol = delete_redundant(sol)
        best_solution = sol

    # Convert indices to lists
    result = [list(blocks[i]) for i in best_solution]
    
    return result
