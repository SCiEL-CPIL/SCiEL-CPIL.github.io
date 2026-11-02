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
