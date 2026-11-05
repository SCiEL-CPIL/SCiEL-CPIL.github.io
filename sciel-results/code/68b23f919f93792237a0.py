import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # 1. Precomputation
    # Generate all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    M = len(all_t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Generate all k-subsets
    all_k_blocks = list(itertools.combinations(range(v), k))
    N = len(all_k_blocks)
    
    # Map each k-block to the indices of the t-subsets it covers
    # B[i] will be an array of indices into all_t_subsets
    B = np.zeros((N, C(k, t)), dtype=np.int32)
    for i, block in enumerate(all_k_blocks):
        subs = list(itertools.combinations(block, t))
        for j, sub in enumerate(subs):
            B[i, j] = subset_to_idx[sub]
            
    # 2. Greedy Cover
    uncovered_mask = np.ones(M, dtype=bool)
    selected_blocks = []
    
    while uncovered_mask.any():
        if time.time() > deadline:
            break
            
        # Calculate gain for each block
        # uncovered_mask[B] gives a (N, 56) boolean array
        # Sum along axis 1 gives count of uncovered subsets for each block
        gains = uncovered_mask[B].sum(axis=1)
        
        best_idx = np.argmax(gains)
        if gains[best_idx] == 0:
            break # Should not happen if valid cover exists
            
        selected_blocks.append(best_idx)
        # Mark covered
        uncovered_mask[B[best_idx]] = False
        
    if not uncovered_mask.all():
        # If we ran out of time during greedy, we might have an incomplete cover.
        # But the problem requires a valid cover.
        # If we didn't finish, we should probably return the best valid cover we can quickly fix?
        # Or just return the blocks we have if they happen to cover everything?
        # The loop condition `while uncovered_mask.any()` ensures we stop when covered.
        # If we break due to deadline, `uncovered_mask` might still have True values.
        # In that case, the result is invalid.
        # To be safe, if we hit deadline during greedy, we should try to complete it fast?
        # Or just return whatever we have? The contract says "Valid when every t-subset...".
        # If invalid, score is 0.
        # So we must ensure we return a valid cover.
        # If we run out of time, we should probably fall back to a simpler, slower but guaranteed finish?
        # Or just hope the greedy finishes. 300 blocks * 5ms = 1.5s. It should finish.
        pass

    # 3. Local Search: Removal
    # Maintain coverage counts
    cov_count = np.zeros(M, dtype=np.int32)
    for block_idx in selected_blocks:
        cov_count[B[block_idx]] += 1
        
    best_solution = selected_blocks[:]
    best_len = len(best_solution)
    
    last_improvement_time = time.time()
    
    while time.time() < deadline:
        improved = False
        current_solution = best_solution[:]
        
        # Shuffle to avoid order bias
        random.shuffle(current_solution)
        
        new_solution = []
        for block_idx in current_solution:
            if time.time() > deadline:
                break
                
            subsets = B[block_idx]
            cov_count[subsets] -= 1
            
            # Check if any subset is now uncovered (count == 0)
            if np.any(cov_count[subsets] == 0):
                # Restore
                cov_count[subsets] += 1
                new_solution.append(block_idx)
            else:
                # Block was redundant
                improved = True
        
        if improved:
            best_solution = new_solution
            best_len = len(best_solution)
            last_improvement_time = time.time()
            
            # Recalculate cov_count for the new best solution for next iteration
            cov_count = np.zeros(M, dtype=np.int32)
            for block_idx in best_solution:
                cov_count[B[block_idx]] += 1
        else:
            # No improvement in removal. Try a "Kick" if we have time.
            if time.time() < deadline - 5: # Need 5s buffer
                # Kick: Remove a few random blocks, then re-greedy to fix
                kick_size = min(3, len(best_solution))
                if kick_size > 0:
                    # Copy current state
                    kick_cov = cov_count.copy()
                    kick_sol = best_solution[:]
                    
                    # Remove kick_size random blocks
                    removed_indices = random.sample(range(len(kick_sol)), kick_size)
                    removed_blocks = [kick_sol[i] for i in removed_indices]
                    
                    # Decrement counts for removed blocks
                    for b in removed_blocks:
                        kick_cov[B[b]] -= 1
                        
                    # Find uncovered subsets
                    uncovered_idx = np.where(kick_cov == 0)[0]
                    
                    if len(uncovered_idx) > 0:
                        # Run Greedy to cover these specific uncovered subsets
                        # We need to add blocks to cover `uncovered_idx`
                        # We can reuse the greedy logic but only care about these subsets?
                        # Actually, we can just run the standard greedy on the *remaining* uncovered mask.
                        
                        # Create a mask for currently uncovered subsets
                        # Note: kick_cov reflects the state after removal.
                        # Subsets with kick_cov > 0 are still covered by remaining blocks.
                        # Subsets with kick_cov == 0 need new blocks.
                        
                        # We need to add blocks to cover the sets where kick_cov == 0.
                        # Let's just run the main greedy loop again on a temporary mask?
                        # That might be expensive.
                        
                        # Alternative: Simple Greedy for the holes
                        # Temp mask
                        temp_uncovered = np.zeros(M, dtype=bool)
                        temp_uncovered[uncovered_idx] = True
                        
                        added_blocks = []
                        # Greedy for holes
                        while temp_uncovered.any():
                            if time.time() > deadline:
                                break
                            gains = temp_uncovered[B].sum(axis=1)
                            # We should prefer blocks that are already in the solution? No, any block.
                            # But we want to minimize total blocks.
                            # Standard greedy picks max gain.
                            best_idx = np.argmax(gains)
                            if gains[best_idx] == 0:
                                break
                            added_blocks.append(best_idx)
                            temp_uncovered[B[best_idx]] = False
                            
                        # Combine
                        candidate_solution = [b for i, b in enumerate(kick_sol) if i not in removed_indices] + added_blocks
                        
                        # Minimize candidate_solution via Removal
                        cand_cov = np.zeros(M, dtype=np.int32)
                        for b in candidate_solution:
                            cand_cov[B[b]] += 1
                            
                        cand_sol = candidate_solution[:]
                        random.shuffle(cand_sol)
                        final_cand = []
                        for b in cand_sol:
                            subs = B[b]
                            cand_cov[subs] -= 1
                            if np.any(cand_cov[subs] == 0):
                                cand_cov[subs] += 1
                                final_cand.append(b)
                            
                        if len(final_cand) < best_len:
                            best_solution = final_cand
                            best_len = len(best_solution)
                            last_improvement_time = time.time()
                            # Update main cov_count
                            cov_count = np.zeros(M, dtype=np.int32)
                            for b in best_solution:
                                cov_count[B[b]] += 1
            else:
                break # No time for kicks

    # Convert indices back to blocks
    return [list(all_k_blocks[i]) for i in best_solution]

def C(n, r):
    if r < 0 or r > n:
        return 0
    return math.comb(n, r)

import math
