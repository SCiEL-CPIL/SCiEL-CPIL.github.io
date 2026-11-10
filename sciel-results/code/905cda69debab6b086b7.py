import time
import random

def construct():
    N = 512
    start_time = time.time()
    time_limit = 58.0  # Safety margin
    
    # 1. Precompute all 4-term arithmetic progressions
    # AP: (a, a+d, a+2d, a+3d)
    # Store for each element x, the list of APs it belongs to.
    # To make checking fast, store the other 3 elements.
    # ap_neighbors[x] = list of (y, z, w) such that {x,y,z,w} is an AP.
    # Actually, for a swap (i, j), we check APs containing j.
    # For each AP P containing j, we check if P\{j} is subset of S\{i}.
    
    aps_containing = [[] for _ in range(N + 1)]
    
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            # AP is (a, b, c, e)
            aps_containing[a].append((b, c, e))
            aps_containing[b].append((a, c, e))
            aps_containing[c].append((a, b, e))
            aps_containing[e].append((a, b, c))

    def is_valid_add(S, x):
        """Check if adding x to S creates a 4-AP."""
        for (y, z, w) in aps_containing[x]:
            if y in S and z in S and w in S:
                return False
        return True

    def is_valid_swap(S, i, j):
        """Check if swapping i (in S) with j (not in S) keeps S valid."""
        # i is removed, j is added.
        # New set S' = S - {i} + {j}.
        # A 4-AP is formed in S' only if it contains j.
        # Let P be an AP containing j.
        # P is a subset of S' iff P\{j} is subset of S\{i}.
        # P\{j} is subset of S\{i} iff (P\{j} is subset of S) AND (i is not in P\{j}).
        # If i is in P, then i is in P\{j} (since i!=j). So the condition "i is not in P\{j}" fails.
        # Thus, if i is in P, P is NOT a subset of S'.
        # So we only need to check APs containing j but NOT i.
        
        for (y, z, w) in aps_containing[j]:
            # Check if i is one of y, z, w
            if i == y or i == z or i == w:
                continue
            # If i is not in P, we must ensure P\{j} is NOT a subset of S.
            # i.e., at least one of y, z, w is NOT in S.
            if y in S and z in S and w in S:
                return False
        return True

    def local_search(S):
        """Perform local search to maximize size of S."""
        S = set(S)
        while time.time() - start_time < time_limit:
            added = False
            # Try to add elements
            # Randomize order to escape symmetries
            candidates = [x for x in range(1, N + 1) if x not in S]
            random.shuffle(candidates)
            
            for x in candidates:
                if is_valid_add(S, x):
                    S.add(x)
                    added = True
                    # After adding, we might be able to add more, so restart the candidate loop?
                    # Or just continue. Restarting is safer to catch new opportunities immediately.
                    # But checking all candidates is O(N). 
                    # Let's just break and restart the outer while loop to re-evaluate all candidates.
                    break
            
            if added:
                continue
                
            # If no element could be added, try swaps
            swapped = False
            elements_in_S = list(S)
            random.shuffle(elements_in_S)
            
            for i in elements_in_S:
                if time.time() - start_time > time_limit:
                    break
                candidates_out = [x for x in range(1, N + 1) if x not in S]
                random.shuffle(candidates_out)
                
                for j in candidates_out:
                    if is_valid_swap(S, i, j):
                        S.remove(i)
                        S.add(j)
                        swapped = True
                        break
                if swapped:
                    break
            
            if not swapped:
                # Local optimum reached
                break
                
        return S

    # Strategy: Multiple restarts with randomization
    best_S = set()
    
    # Initial greedy to get a decent start
    S_greedy = set()
    # Randomize the order of addition for the initial greedy to find different basins
    # But we want to be fast.
    
    # Run a few iterations of the local search starting from different initial states
    # Start 1: Greedy
    S_curr = set()
    for x in range(1, N + 1):
        if is_valid_add(S_curr, x):
            S_curr.add(x)
    best_S = local_search(S_curr)
    
    # Start 2: Randomized Greedy
    if time.time() - start_time < time_limit - 10:
        S_curr = set()
        order = list(range(1, N + 1))
        random.shuffle(order)
        for x in order:
            if is_valid_add(S_curr, x):
                S_curr.add(x)
        S_opt = local_search(S_curr)
        if len(S_opt) > len(best_S):
            best_S = S_opt

    # Start 3: Another Randomized Greedy
    if time.time() - start_time < time_limit - 10:
        S_curr = set()
        order = list(range(1, N + 1))
        random.shuffle(order)
        for x in order:
            if is_valid_add(S_curr, x):
                S_curr.add(x)
        S_opt = local_search(S_curr)
        if len(S_opt) > len(best_S):
            best_S = S_opt
            
    # If time remains, continue local search on the best found so far
    # Or run more restarts?
    # local_search already runs until limit or stuck.
    # If it got stuck early, we can try another restart.
    
    if time.time() - start_time < time_limit - 5:
        S_curr = set()
        order = list(range(1, N + 1))
        random.shuffle(order)
        for x in order:
            if is_valid_add(S_curr, x):
                S_curr.add(x)
        S_opt = local_search(S_curr)
        if len(S_opt) > len(best_S):
            best_S = S_opt

    return sorted(best_S)

# EVOLVE-BLOCK-END
