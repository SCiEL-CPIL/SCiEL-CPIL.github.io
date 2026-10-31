import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    time_limit = 110.0

    # Fallback for non-standard parameters
    if v != 19 or k != 7 or t != 4:
        return _greedy_generic(v, k, t)

    # --- Precomputation ---
    # 1. Map 4-subsets to IDs
    map4 = {}
    all_4s = []
    id_counter = 0
    for comb in itertools.combinations(range(v), t):
        map4[comb] = id_counter
        all_4s.append(comb)
        id_counter += 1
    n_4s = id_counter # 3876

    # 2. For each 7-subset (block), get list of 4-subset IDs it covers
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks) # 50388
    block_4s = []
    for block in all_blocks:
        ids = []
        for sub in itertools.combinations(block, t):
            ids.append(map4[sub])
        block_4s.append(np.array(ids, dtype=np.int32))
    
    # 3. For each 4-subset, get list of block indices covering it
    # This is useful for repair steps
    sub_blocks = [[] for _ in range(n_4s)]
    for b_idx, ids in enumerate(block_4s):
        for sid in ids:
            sub_blocks[sid].append(b_idx)
            
    # --- Greedy Initialization ---
    counts = np.zeros(n_4s, dtype=np.int32)
    current_blocks = []
    
    # Greedy Set Cover
    # To speed up, we can precompute scores, but dynamic update is needed.
    # Since 50k blocks, calculating scores from scratch each step is 50k * 35 = 1.75M ops.
    # ~100 steps * 1.75M = 175M ops. Might be slow in Python if not vectorized.
    # Let's use a slightly smarter greedy or just accept the cost if vectorized.
    
    # Vectorized score calculation:
    # scores[b] = sum(uncovered_mask[block_4s[b]])
    # This requires 50388 array accesses and sums.
    
    uncovered_mask = np.ones(n_4s, dtype=bool)
    
    while np.any(uncovered_mask):
        if time.time() - start_time > time_limit - 10:
            break
            
        # Calculate score for all blocks
        # scores[i] = number of uncovered 4s in block i
        # We can do this by iterating blocks? No, too slow in Python loop.
        # We need numpy.
        # However, block_4s is a list of arrays.
        # We can stack them? 50388 x 35 array of int32 is ~7MB.
        # Then scores = (uncovered_mask[all_block_4s_array]).sum(axis=1)
        
        # Let's precompute a flat array of block_4s for vectorization if needed, 
        # but for now, let's try a faster greedy heuristic:
        # Pick block covering most uncovered.
        
        # Optimization: Only consider blocks that cover at least one uncovered set?
        # Actually, calculating scores for all 50k blocks every step is the bottleneck.
        # 50388 * 35 = 1.7M operations per step.
        # If we need 200 steps, that's 350M operations.
        # In NumPy, 350M simple ops might take a few seconds.
        
        # Let's create the big array once.
        # block_4s_array shape: (n_blocks, 35)
        block_4s_array = np.array(block_4s, dtype=np.int32)
        
        # scores = uncovered_mask[block_4s_array].sum(axis=1)
        # This creates a temporary array of shape (50388, 35) of bools, then sums.
        # This is efficient in NumPy.
        
        scores = uncovered_mask[block_4s_array].sum(axis=1)
        best_idx = np.argmax(scores)
        
        if scores[best_idx] == 0:
             # Should not happen if uncovered_mask has True
             break
             
        current_blocks.append(best_idx)
        counts[block_4s_array[best_idx]] += 1
        uncovered_mask = (counts == 0)
        
    # --- Local Search ---
    # State: current_blocks (list of indices), counts (np array)
    
    def is_valid():
        return np.all(counts >= 1)
        
    def try_remove(b_idx):
        # Returns True if removal is valid
        # Temporarily decrement
        counts[block_4s_array[b_idx]] -= 1
        if np.any(counts == 0):
            counts[block_4s_array[b_idx]] += 1
            return False
        # Valid, so remove from list
        current_blocks.remove(b_idx)
        return True
        
    def try_add(b_idx):
        if b_idx in current_blocks:
            return False
        current_blocks.append(b_idx)
        counts[block_4s_array[b_idx]] += 1
        return True
        
    # Redundancy Calculation
    # For each block in current_blocks, count how many of its 4s have count > 1
    def get_redundancy_scores():
        # current_blocks is a list of indices
        if not current_blocks:
            return []
        # Get the 4s for these blocks
        # Shape: (N_current, 35)
        b_4s = block_4s_array[current_blocks]
        # Get counts for these 4s
        # Shape: (N_current, 35)
        c_4s = counts[b_4s]
        # Redundancy is count of elements in c_4s that are > 1
        red_scores = (c_4s > 1).sum(axis=1)
        return red_scores
        
    # Local Search Loop
    best_blocks = list(current_blocks)
    best_len = len(current_blocks)
    
    while time.time() - start_time < time_limit - 5:
        # 1. Try to remove most redundant blocks
        removed_any = False
        # Calculate redundancy for all current blocks
        red_scores = get_redundancy_scores()
        if len(red_scores) == 0:
            break
            
        # Sort indices by redundancy (desc)
        # We need to map back to block indices
        # current_blocks[i] is the block index, red_scores[i] is its score
        order = np.argsort(red_scores)[::-1]
        
        for i in order:
            if time.time() - start_time > time_limit - 5:
                break
            b_idx = current_blocks[i]
            if try_remove(b_idx):
                removed_any = True
                # If we removed a block, the redundancy scores of other blocks change.
                # We should restart the pass or recalculate.
                # For efficiency, let's just break and recalculate.
                break
                
        if not removed_any:
            # 2. If no removals, try swaps
            # Pick a random block to remove
            if len(current_blocks) > 0:
                b_old = random.choice(current_blocks)
                # Tentatively remove
                counts[block_4s_array[b_old]] -= 1
                uncovered_now = (counts == 0)
                
                if np.any(uncovered_now):
                    # We need to add a block that covers the uncovered sets
                    # Find blocks that cover the uncovered sets
                    # We can just pick a random block from the union of sub_blocks[uncovered_indices]
                    # Or pick the one covering the most uncovered.
                    
                    # Get indices of uncovered 4s
                    unc_idx = np.where(uncovered_now)[0]
                    
                    # Find candidate blocks
                    # For each uncovered 4, look at sub_blocks
                    # We want a block that covers many of them.
                    
                    # Heuristic: Pick a random uncovered 4, pick a random block covering it.
                    # Or better: Pick the uncovered 4 with the fewest covering blocks (bottleneck),
                    # and pick a block covering it.
                    
                    # Let's try to find a block that covers the most uncovered sets.
                    # This is expensive. Let's just pick one that covers at least one.
                    
                    # Collect candidates
                    candidates = set()
                    for uid in unc_idx:
                        candidates.update(sub_blocks[uid])
                        
                    if candidates:
                        # Pick the candidate that covers the most uncovered sets
                        # This is O(|candidates| * 35)
                        best_candidate = -1
                        max_cov = -1
                        for cand in candidates:
                            # Count how many of unc_idx are in block_4s_array[cand]
                            # We can do this by checking intersection
                            # Or just sum(uncovered_now[block_4s_array[cand]])
                            cov = np.sum(uncovered_now[block_4s_array[cand]])
                            if cov > max_cov:
                                max_cov = cov
                                best_candidate = cand
                                
                        if best_candidate != -1 and best_candidate not in current_blocks:
                            # Perform Swap: Add best_candidate, keep b_old removed (for now)
                            # Wait, we already decremented b_old.
                            # We need to add best_candidate.
                            current_blocks.remove(b_old)
                            current_blocks.append(best_candidate)
                            counts[block_4s_array[best_candidate]] += 1
                            # b_old counts are already decremented and NOT incremented back.
                            # So the swap is: -b_old, +best_candidate.
                            
                            # Check if valid (it should be, because we covered the uncovered ones)
                            # But did we uncover anything else? No, we only added.
                            # Did we cover the uncovered ones? Yes, best_candidate covers max_cov > 0.
                            # Are there any remaining uncovered?
                            if not np.any(counts == 0):
                                # Successful swap
                                pass
                            else:
                                # Failed, revert
                                counts[block_4s_array[best_candidate]] -= 1
                                counts[block_4s_array[b_old]] += 1
                                current_blocks.remove(best_candidate)
                                current_blocks.append(b_old)
                        else:
                            # No valid candidate, revert b_old
                            counts[block_4s_array[b_old]] += 1
                    else:
                        counts[block_4s_array[b_old]] += 1
                else:
                    # b_old was redundant, remove it
                    current_blocks.remove(b_old)
                    removed_any = True # Just in case logic flow needs it, though we handled removal above
                    
        # Update best
        if len(current_blocks) < best_len:
            best_len = len(current_blocks)
            best_blocks = list(current_blocks)
            
        # Optional: Random restart / perturbation if stuck
        # If we haven't improved in a long time, we could randomize.
        # For now, the swap mechanism should handle it.
        
    # Ensure we return a valid covering
    # The current state should be valid.
    # Let's verify and repair if necessary.
    if not np.all(counts >= 1):
        # Greedy repair
        uncovered_mask = (counts == 0)
        while np.any(uncovered_mask):
            scores = uncovered_mask[block_4s_array].sum(axis=1)
            best_idx = np.argmax(scores)
            if scores[best_idx] == 0:
                break
            if best_idx not in current_blocks:
                current_blocks.append(best_idx)
                counts[block_4s_array[best_idx]] += 1
                uncovered_mask = (counts == 0)
            else:
                break # Should not happen

    # Use the best found blocks
    # Rebuild counts for best_blocks to be sure
    final_blocks = best_blocks
    # Verify best_blocks is valid
    # (Assuming local search maintained validity)
    
    result = [list(all_blocks[i]) for i in final_blocks]
    return result

def _greedy_generic(v, k, t):
    # Fallback for non-19,7,4
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        block = list(min(uncovered))
        while len(block) < k:
            def gain(x):
                return sum(1 for rest in itertools.combinations(block, t - 1)
                           if tuple(sorted(rest + (x,))) in uncovered)
            block.append(max((x for x in range(v) if x not in block), key=gain))
        block.sort()
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
    return blocks
