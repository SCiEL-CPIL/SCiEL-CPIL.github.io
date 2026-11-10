# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_START = time.time()
_DEADLINE = _START + 580.0  # safety margin before 600s limit


def _compute_forbidden_bitmask(m):
    """
    Returns a list of bitmasks. forbidden_mask[i] is a bitmask where bit j is 1
    if (j - i) % m is a forbidden difference.
    """
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    masks = [0] * m
    for i in range(m):
        mask = 0
        for j in range(m):
            if i == j:
                continue
            d = (j - i) % m
            if d in forbidden:
                mask |= (1 << j)
        masks[i] = mask
    return masks


def _greedy_weighted(m, forbidden_masks, rng):
    """
    Greedy initialization that prioritizes nodes with fewer conflicts.
    """
    # Calculate initial degree (number of forbidden neighbors) for each node
    degrees = [bin(mask).count("1") for mask in forbidden_masks]
    
    # We want to pick nodes with low degree first (fewer conflicts)
    # Create a list of (degree, node) and sort by degree
    candidates = sorted(range(m), key=lambda x: degrees[x])
    # Shuffle ties to introduce randomness
    # A simple way to handle ties randomly is to shuffle the list, but that breaks the sort.
    # Instead, we can bucket by degree or just shuffle and then stable sort.
    # Let's just shuffle the candidates first, then stable sort by degree.
    rng.shuffle(candidates)
    candidates.sort(key=lambda x: degrees[x])
    
    selected_mask = 0
    selected_set = set()
    
    # We need to dynamically update degrees or just check compatibility on the fly.
    # Checking compatibility on the fly is O(k) where k is selected size.
    # With bitmask, we can check if (forbidden_masks[cand] & selected_mask) == 0.
    
    for cand in candidates:
        # Check if cand is compatible with all selected
        if (forbidden_masks[cand] & selected_mask) == 0:
            selected_mask |= (1 << cand)
            selected_set.add(cand)
            
    return selected_set, selected_mask


def _local_search_2swap(m, forbidden_masks, selected_set, selected_mask, rng, max_iterations=50):
    """
    Local search using 2-swap moves: remove two elements, add two different elements.
    Returns improved set and mask.
    """
    improved = True
    iterations = 0
    best_size = len(selected_set)
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        # Randomly pick two elements to remove
        sel_list = list(selected_set)
        if len(sel_list) < 2:
            break
            
        rng.shuffle(sel_list)
        rem1, rem2 = sel_list[0], sel_list[1]
        
        new_mask = selected_mask & ~(1 << rem1) & ~(1 << rem2)
        new_set = selected_set - {rem1, rem2}
        
        # Find two candidates to add
        # We want to add two nodes that are compatible with each other and the new_set
        # To speed this up, we can pre-filter candidates that are compatible with new_set
        # Then check pairs among them.
        
        # Get list of all nodes not in new_set
        # Since m=205, we can just iterate
        candidates = []
        for i in range(m):
            if i in new_set:
                continue
            # Check compatibility with new_set
            if (forbidden_masks[i] & new_mask) == 0:
                candidates.append(i)
                
        if len(candidates) < 2:
            continue
            
        # Try to find a pair (c1, c2) in candidates such that they are compatible with each other
        # c1 and c2 are compatible if (forbidden_masks[c1] & (1 << c2)) == 0
        # Note: forbidden_masks[c1] has bits set for all j where (j-c1)%m is forbidden.
        # So we check if bit c2 is set in forbidden_masks[c1].
        
        found_pair = False
        # Limit pair checks to avoid O(N^2) explosion if candidates list is large
        # But m=205 is small.
        for i in range(len(candidates)):
            c1 = candidates[i]
            for j in range(i + 1, len(candidates)):
                c2 = candidates[j]
                # Check if c1 and c2 are compatible
                if (forbidden_masks[c1] & (1 << c2)) == 0:
                    # They are compatible
                    new_set.add(c1)
                    new_set.add(c2)
                    new_mask |= (1 << c1) | (1 << c2)
                    improved = True
                    found_pair = True
                    break
            if found_pair:
                break
                
        if improved:
            # Update state
            selected_set = new_set
            selected_mask = new_mask
            if len(selected_set) > best_size:
                best_size = len(selected_set)
                
    return selected_set, selected_mask


def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    forbidden_masks = _compute_forbidden_bitmask(m)

    rng = random.Random(12345)
    best_set = set()
    best_mask = 0

    # Phase 1: Weighted Greedy + 1-Swap (Fast)
    # We will do several restarts of weighted greedy.
    # Then apply 1-swap (add only) to fill gaps.
    
    num_restarts = 100
    for r in range(num_restarts):
        if time.time() > _DEADLINE - 10:
            break
            
        sel_set, sel_mask = _greedy_weighted(m, forbidden_masks, rng)
        
        # Quick 1-add local search
        improved = True
        while improved:
            improved = False
            for cand in range(m):
                if cand in sel_set:
                    continue
                if (forbidden_masks[cand] & sel_mask) == 0:
                    sel_set.add(cand)
                    sel_mask |= (1 << cand)
                    improved = True
                    break
        
        if len(sel_set) > len(best_set):
            best_set = sel_set
            best_mask = sel_mask

    # Phase 2: 2-Swap Local Search on Best Candidates
    # Apply 2-swap to the current best and potentially other good candidates
    # We keep a list of top candidates
    
    top_candidates = [(len(best_set), best_set, best_mask)]
    
    # Let's run a few more greedy restarts just to get more diverse starting points for 2-swap
    for r in range(50):
        if time.time() > _DEADLINE - 5:
            break
        sel_set, sel_mask = _greedy_weighted(m, forbidden_masks, rng)
        
        # Quick 1-add
        improved = True
        while improved:
            improved = False
            for cand in range(m):
                if cand in sel_set:
                    continue
                if (forbidden_masks[cand] & sel_mask) == 0:
                    sel_set.add(cand)
                    sel_mask |= (1 << cand)
                    improved = True
                    break
        
        if len(sel_set) > len(best_set):
            best_set = sel_set
            best_mask = sel_mask
            top_candidates = [(len(best_set), best_set, best_mask)]
        else:
            # Keep it if it's close to best
            if len(sel_set) >= len(best_set) - 2:
                top_candidates.append((len(sel_set), sel_set, sel_mask))
                
    # Sort candidates by size descending
    top_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Try 2-swaps on the top candidates
    # Limit number of candidates to process to stay within time
    max_candidates_to_process = 5
    
    for i in range(min(max_candidates_to_process, len(top_candidates))):
        if time.time() > _DEADLINE - 2:
            break
            
        size, curr_set, curr_mask = top_candidates[i]
        
        # Run 2-swap local search
        # We might want to run multiple rounds of 2-swap
        improved = True
        while improved and time.time() < _DEADLINE - 2:
            improved = False
            # Run a batch of 2-swaps
            for _ in range(20):
                if time.time() > _DEADLINE - 2:
                    break
                new_set, new_mask = _local_search_2swap(m, forbidden_masks, curr_set, curr_mask, rng, max_iterations=1)
                if len(new_set) > len(curr_set):
                    curr_set = new_set
                    curr_mask = new_mask
                    improved = True
                    if len(curr_set) > len(best_set):
                        best_set = curr_set
                        best_mask = curr_mask
        
    return np.array(sorted(best_set), dtype=np.int64)


# EVOLVE-BLOCK-END
