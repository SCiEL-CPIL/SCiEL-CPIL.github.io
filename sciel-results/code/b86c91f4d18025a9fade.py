import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    # Safety margin: evaluator kills at 150s. We aim for 140s limit.
    timeout = 140.0
    
    # 1. Precomputation
    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    N_blocks = len(all_blocks)
    
    all_subsets = list(itertools.combinations(points, t))
    N_subsets = len(all_subsets)
    
    # Map subset to index
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    # Precompute which subsets each block covers
    block_subsets = []
    for b in all_blocks:
        subs = [subset_to_idx[s] for s in itertools.combinations(b, t)]
        block_subsets.append(subs)
        
    # Precompute which blocks cover each subset
    blocks_covering_subset = [[] for _ in range(N_subsets)]
    for i, subs in enumerate(block_subsets):
        for s in subs:
            blocks_covering_subset[s].append(i)
            
    # Helper to check time
    def time_exceeded():
        return time.time() - start_time > timeout

    # 2. Greedy Construction
    def run_greedy(randomize_ties=False):
        if time_exceeded():
            return None
            
        cover_count = np.zeros(N_subsets, dtype=np.int32)
        # potential[i] = number of uncovered subsets covered by block i
        potential = np.array([len(s) for s in block_subsets], dtype=np.int32)
        
        selected_blocks = []
        
        # We need to track uncovered count to know when to stop
        uncovered_count = N_subsets
        
        while uncovered_count > 0:
            if time_exceeded():
                return None
                
            # Find block with max potential
            max_pot = np.max(potential)
            if max_pot == 0:
                break # Should not happen if uncovered_count > 0 and problem is solvable
                
            if randomize_ties:
                candidates = np.where(potential == max_pot)[0]
                best_idx = candidates[random.randint(0, len(candidates)-1)]
            else:
                best_idx = np.argmax(potential)
                
            selected_blocks.append(best_idx)
            
            # Update cover_count and potentials
            for s in block_subsets[best_idx]:
                old_count = cover_count[s]
                cover_count[s] += 1
                if old_count == 0:
                    uncovered_count -= 1
                    # Decrease potential for all blocks that cover this subset
                    # Only if they are not the one just selected? 
                    # Actually, potential is "uncovered subsets covered by b".
                    # If s was uncovered, it contributed 1 to potential of all b covering s.
                    # Now s is covered, so it contributes 0.
                    # So we decrement potential for all b in blocks_covering_subset[s].
                    for b in blocks_covering_subset[s]:
                        potential[b] -= 1
                        
        return selected_blocks

    # 3. Local Search: Reduction and Swap
    def local_search(selected_blocks):
        if time_exceeded():
            return selected_blocks
            
        cover_count = np.zeros(N_subsets, dtype=np.int32)
        for b_idx in selected_blocks:
            for s in block_subsets[b_idx]:
                cover_count[s] += 1
                
        selected_set = set(selected_blocks)
        
        # Phase 1: Reduction
        changed = True
        while changed and not time_exceeded():
            changed = False
            # Iterate in reverse to allow safe removal
            for i in range(len(selected_blocks) - 1, -1, -1):
                b_idx = selected_blocks[i]
                is_redundant = True
                for s in block_subsets[b_idx]:
                    if cover_count[s] == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    # Remove block
                    for s in block_subsets[b_idx]:
                        cover_count[s] -= 1
                    selected_blocks.pop(i)
                    selected_set.remove(b_idx)
                    changed = True
                    
        # Phase 2: Swap Search
        # Try to swap blocks to enable further reductions
        # We iterate multiple times as swaps can cascade
        for _ in range(50): # Limit swap iterations to prevent infinite loops
            if time_exceeded():
                break
                
            improved = False
            # Randomize order of blocks to try
            current_list = list(selected_blocks)
            random.shuffle(current_list)
            
            for b_idx in current_list:
                if b_idx not in selected_set:
                    continue # Block was removed in a previous step of this pass? 
                             # Actually, we are iterating a snapshot. 
                             # If it was removed by reduction, it's not in selected_set.
                             # But we are in Swap phase. Reduction is done.
                             # So b_idx should be in selected_set.
                             
                # Tentatively remove b_idx
                # Find subsets that become uncovered
                uncovered_subs = []
                for s in block_subsets[b_idx]:
                    if cover_count[s] == 1:
                        uncovered_subs.append(s)
                        
                if not uncovered_subs:
                    # It was redundant, but reduction should have caught it.
                    # This can happen if we entered swap phase and counts changed?
                    # No, counts are static during the swap scan for a specific b_idx.
                    # If reduction didn't remove it, it's not redundant.
                    # So this case shouldn't happen if reduction was thorough.
                    # But just in case, remove it.
                    for s in block_subsets[b_idx]:
                        cover_count[s] -= 1
                    selected_blocks.remove(b_idx)
                    selected_set.remove(b_idx)
                    improved = True
                    break # Restart reduction
                    
                # Find a block j that covers all uncovered_subs
                # Intersection of blocks_covering_subset[s] for s in uncovered_subs
                # Start with the list for the first subset
                candidate_blocks = set(blocks_covering_subset[uncovered_subs[0]])
                for s in uncovered_subs[1:]:
                    candidate_blocks &= set(blocks_covering_subset[s])
                    
                # Filter out blocks already in selected_set
                candidate_blocks -= selected_set
                
                if candidate_blocks:
                    # Pick a candidate
                    # Heuristic: Pick one that covers the most *other* subsets?
                    # Or just random.
                    j_idx = random.choice(list(candidate_blocks))
                    
                    # Perform swap
                    # Remove b_idx
                    for s in block_subsets[b_idx]:
                        cover_count[s] -= 1
                    selected_blocks.remove(b_idx)
                    selected_set.remove(b_idx)
                    
                    # Add j_idx
                    for s in block_subsets[j_idx]:
                        cover_count[s] += 1
                    selected_blocks.append(j_idx)
                    selected_set.add(j_idx)
                    
                    improved = True
                    break # Restart reduction to see if new block makes others redundant
                    
            if not improved:
                break # No more swaps found in this pass
                
            # If improved, run reduction again immediately
            # The loop structure above breaks out of the b_idx loop, 
            # but we need to re-run reduction.
            # So we structure the outer loop to re-run reduction if improved.
            
            # Re-run reduction
            changed = True
            while changed and not time_exceeded():
                changed = False
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
                        
        return selected_blocks

    # 4. Main Execution
    best_blocks = None
    best_len = float('inf')
    
    # Run a few deterministic/randomized greedy instances
    # The first one is deterministic
    res = run_greedy(randomize_ties=False)
    if res:
        res = local_search(res)
        if res:
            best_blocks = res
            best_len = len(res)
            
    # Run randomized restarts
    # Estimate time taken so far
    t_elapsed = time.time() - start_time
    if t_elapsed < timeout - 5:
        # Do restarts
        # Each restart takes ~1s greedy + ~1-2s local search?
        # Let's try up to 20 restarts or until time runs out
        for i in range(20):
            if time.time() - start_time > timeout - 2:
                break
            res = run_greedy(randomize_ties=True)
            if res:
                res = local_search(res)
                if res and len(res) < best_len:
                    best_len = len(res)
                    best_blocks = res

    if best_blocks is None:
        # Fallback: return empty or invalid? 
        # Problem says "Valid when every t-subset...". 
        # If we fail, we should probably return the best effort, but it might be invalid.
        # However, greedy should always finish for these parameters.
        return []

    final_blocks = [list(all_blocks[i]) for i in best_blocks]
    return final_blocks
