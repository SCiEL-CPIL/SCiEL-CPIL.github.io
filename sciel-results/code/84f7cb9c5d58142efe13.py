import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s
    
    # Precomputation
    # 1. Enumerate all t-subsets and assign indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    # 2. Enumerate all k-subsets and compute their coverage as bitmasks
    k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(k_subsets)
    
    # coverage[i] is an integer bitmask of t-subsets covered by k_subset i
    coverage = [0] * num_k_subsets
    # Also store the list of t-subset indices for each block to update counts efficiently
    coverage_lists = [[] for _ in range(num_k_subsets)]
    
    for i, ks in enumerate(k_subsets):
        mask = 0
        indices = []
        for ts in itertools.combinations(ks, t):
            idx = t_subset_index[ts]
            mask |= (1 << idx)
            indices.append(idx)
        coverage[i] = mask
        coverage_lists[i] = indices

    best_blocks = None
    best_count = float('inf')
    
    # Run multiple random restarts
    num_restarts = 50000  # We will limit by time
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # 1. Greedy Phase
        uncovered_mask = (1 << num_t_subsets) - 1
        current_blocks = []
        used_blocks = set()
        
        greedy_success = True
        while uncovered_mask:
            if time.time() > deadline:
                greedy_success = False
                break
                
            max_gain = 0
            candidates = []
            
            for i in range(num_k_subsets):
                if i in used_blocks:
                    continue
                # Gain is number of new t-subsets covered
                # Use bit_count for speed
                gain = (coverage[i] & uncovered_mask).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    candidates = [i]
                elif gain == max_gain and gain > 0:
                    candidates.append(i)
            
            if max_gain == 0:
                break
                
            # Pick a random candidate to add diversity
            chosen_idx = random.choice(candidates)
            current_blocks.append(chosen_idx)
            used_blocks.add(chosen_idx)
            uncovered_mask &= ~coverage[chosen_idx]
            
        if not greedy_success or uncovered_mask:
            continue
            
        current_count = len(current_blocks)
        
        # 2. Local Search Phase: Only 1-swap (removal) for speed and reliability
        counts = [0] * num_t_subsets
        for b_idx in current_blocks:
            for t_idx in coverage_lists[b_idx]:
                counts[t_idx] += 1
        
        improved = True
        # Limit iterations to avoid getting stuck
        max_local_iters = 200
        iter_count = 0
        
        while improved and time.time() < deadline and iter_count < max_local_iters:
            improved = False
            iter_count += 1
            
            # Try 1-swap (remove redundant block)
            # A block is redundant if for all t in it, counts[t] > 1
            # Iterate in random order to find different removals
            indices_to_check = list(range(len(current_blocks)))
            random.shuffle(indices_to_check)
            
            for pos in indices_to_check:
                if time.time() > deadline:
                    break
                b_idx = current_blocks[pos]
                redundant = True
                for t_idx in coverage_lists[b_idx]:
                    if counts[t_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove b_idx
                    # Using index for removal is O(N), but N is small (~300)
                    current_blocks.pop(pos)
                    for t_idx in coverage_lists[b_idx]:
                        counts[t_idx] -= 1
                    improved = True
                    break
            
        # 3. Pairwise Swap Local Search
        # Try to swap one block with another to reduce total size or improve coverage
        # This is a more aggressive local search step
        if time.time() < deadline:
            # We will do a limited number of pairwise swap attempts
            # For each block, try to replace it with a different block that covers
            # the same unique t-subsets but perhaps allows other blocks to be removed
            # Or simply try to find a better block to replace a current one if it leads
            # to a smaller solution after re-running removal.
            
            # A simpler heuristic: Try swapping a block with another if the new block
            # covers all the t-subsets that the old block uniquely covered (count==1).
            # Then re-check for removals.
            
            swap_improved = True
            max_swap_iters = 50
            swap_iter_count = 0
            
            while swap_improved and time.time() < deadline and swap_iter_count < max_swap_iters:
                swap_improved = False
                swap_iter_count += 1
                
                # Get a random block to try swapping
                pos = random.randrange(len(current_blocks))
                old_idx = current_blocks[pos]
                
                # Find t-subsets that are uniquely covered by old_idx
                unique_ts = [t_idx for t_idx in coverage_lists[old_idx] if counts[t_idx] == 1]
                
                if not unique_ts:
                    # If no unique coverage, we can already remove it in the removal phase.
                    # But if we are here, it means removal phase didn't find it? 
                    # Or maybe it was found and removed. So unique_ts should be non-empty 
                    # if we are in a stable state where removals have been exhausted.
                    # Actually, if unique_ts is empty, we can just remove it.
                    # But the removal phase should have caught that.
                    # So if we are in swap phase, unique_ts is likely non-empty for most blocks.
                    continue

                # Find a candidate block that covers all unique_ts
                # And is not already in current_blocks
                # And ideally has a "small" footprint or high overlap with existing coverage
                # to minimize new unique t-subsets introduced.
                
                best_swap_candidate = -1
                min_new_unique = float('inf')
                
                # We don't want to scan all 75k blocks for every swap attempt if it's slow.
                # But 75k * 50 swaps * 50000 restarts is too much.
                # We should limit this.
                
                # Let's just do a random sample of candidates to keep it fast.
                # Or break out if time is tight.
                
                if time.time() > deadline - 0.5:
                    break

                # Sample a few random blocks
                candidates_to_check = random.sample(range(num_k_subsets), min(100, num_k_subsets))
                
                for cand_idx in candidates_to_check:
                    if cand_idx in used_blocks:
                        continue
                    
                    # Check if cand_idx covers all unique_ts
                    covers_all = True
                    for t_idx in unique_ts:
                        if not (coverage[cand_idx] & (1 << t_idx)):
                            covers_all = False
                            break
                    
                    if not covers_all:
                        continue
                    
                    # If it covers all unique_ts, calculate how many NEW unique t-subsets it introduces
                    # A t-subset is "new unique" if it is covered by cand_idx 
                    # and its count is currently 0 (if we remove old_idx) or 1 (if we keep others).
                    # This is complex to calculate exactly without simulating the swap.
                    
                    # Simpler: Just check if cand_idx is "better" in some heuristic sense.
                    # E.g., it covers more t-subsets in total? No, that's not right.
                    # E.g., it has less overlap with the rest of the solution?
                    
                    # Let's just accept the first valid candidate found to keep it fast.
                    # This is a greedy swap.
                    best_swap_candidate = cand_idx
                    break
                
                if best_swap_candidate != -1:
                    # Perform the swap
                    # 1. Remove old_idx
                    for t_idx in coverage_lists[old_idx]:
                        counts[t_idx] -= 1
                    used_blocks.remove(old_idx)
                    
                    # 2. Add best_swap_candidate
                    current_blocks[pos] = best_swap_candidate
                    used_blocks.add(best_swap_candidate)
                    for t_idx in coverage_lists[best_swap_candidate]:
                        counts[t_idx] += 1
                    
                    swap_improved = True
                    
                    # After a swap, re-run removal phase to clean up
                    removal_improved = True
                    max_removal_iters_after_swap = 20
                    rem_iter = 0
                    while removal_improved and rem_iter < max_removal_iters_after_swap:
                        removal_improved = False
                        rem_iter += 1
                        indices_to_check = list(range(len(current_blocks)))
                        random.shuffle(indices_to_check)
                        for rpos in indices_to_check:
                            rb_idx = current_blocks[rpos]
                            redundant = True
                            for t_idx in coverage_lists[rb_idx]:
                                if counts[t_idx] == 1:
                                    redundant = False
                                    break
                            if redundant:
                                current_blocks.pop(rpos)
                                used_blocks.remove(rb_idx)
                                for t_idx in coverage_lists[rb_idx]:
                                    counts[t_idx] -= 1
                                removal_improved = True
                                break

        if len(current_blocks) < best_count:
            best_count = len(current_blocks)
            best_blocks = [list(k_subsets[b]) for b in current_blocks]
            
            # If we've reached a very good count, we might want to break early or continue
            # But continuing allows us to find even better ones.

    if best_blocks is None:
        # Fallback: If we failed to find any valid covering (should not happen),
        # we must return a valid one. We can't easily construct one on the fly in 0s.
        # However, the first greedy run is very likely to succeed.
        # If it didn't, we return an empty list which scores 0, but this is a failure case.
        # To be safe, we could run a non-randomized greedy one last time if time permits.
        if time.time() < deadline:
             uncovered_mask = (1 << num_t_subsets) - 1
             current_blocks = []
             used_blocks = set()
             while uncovered_mask:
                 max_gain = 0
                 best_i = -1
                 for i in range(num_k_subsets):
                     if i in used_blocks: continue
                     gain = (coverage[i] & uncovered_mask).bit_count()
                     if gain > max_gain:
                         max_gain = gain
                         best_i = i
                 if max_gain == 0: break
                 current_blocks.append(best_i)
                 used_blocks.add(best_i)
                 uncovered_mask &= ~coverage[best_i]
             if not uncovered_mask:
                 best_blocks = [list(k_subsets[b]) for b in current_blocks]
                 best_count = len(current_blocks)

    return best_blocks
