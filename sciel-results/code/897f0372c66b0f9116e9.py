import random
import time

def construct():
    N = 512
    start_time = time.time()
    time_limit = 85.0  # Safety margin before 90s kill

    # 1. Precompute 4-APs
    # ap_list: list of tuples (a, b, c, d_val)
    # aps_containing: list of lists, aps_containing[i] = list of indices of APs containing i
    ap_list = []
    aps_containing = [[] for _ in range(N + 1)]
    
    # Generate all 4-APs
    # a, a+d, a+2d, a+3d <= N
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            d_val = a + 3 * d
            ap_idx = len(ap_list)
            ap_list.append((a, b, c, d_val))
            for v in (a, b, c, d_val):
                aps_containing[v].append(ap_idx)
    
    num_aps = len(ap_list)
    
    # 2. Greedy Construction
    def is_valid_addition(ap_count, x):
        # Check if adding x would complete any 4-AP
        for ap_idx in aps_containing[x]:
            if ap_count[ap_idx] == 3:
                return False
        return True
        
    def greedy_construct(order):
        S = set()
        ap_count = [0] * num_aps
        for x in order:
            if is_valid_addition(ap_count, x):
                S.add(x)
                for ap_idx in aps_containing[x]:
                    ap_count[ap_idx] += 1
        return S, ap_count

    # 3. Local Search with Swaps and Additions
    def local_search(S, ap_count):
        # S is a set, ap_count is a list
        best_S = S.copy()
        best_ap_count = ap_count.copy()
        best_size = len(S)
        
        all_elements = list(range(1, N + 1))
        
        # We will iterate until time runs out or no improvements are found
        # To avoid getting stuck in a local minimum, we use a mix of random restarts within the search
        # but primarily rely on the random nature of the moves.
        
        while time.time() - start_time < time_limit:
            improved = False
            
            # Try to add a random element
            # This is a cheap check.
            for _ in range(100):
                x = random.choice(all_elements)
                if x in S:
                    continue
                if is_valid_addition(ap_count, x):
                    S.add(x)
                    for ap_idx in aps_containing[x]:
                        ap_count[ap_idx] += 1
                    improved = True
                    if len(S) > best_size:
                        best_S = S.copy()
                        best_ap_count = ap_count.copy()
                        best_size = len(S)
                    break
            
            if not improved:
                # Try Swap: remove y, add x
                # We need to be careful with the count updates.
                swapped = False
                for _ in range(50):
                    x = random.choice(all_elements)
                    if x in S:
                        continue
                    
                    # Find a y to remove such that x becomes valid.
                    # x is blocked if there exists an AP P containing x with count(P) == 3.
                    # If we remove y, count(P) becomes count(P) - 1 if y in P.
                    # So we need count(P) - (1 if y in P else 0) < 3.
                    # If count(P) == 3, we MUST remove an element from P.
                    # If count(P) < 3, we don't care about P for x's validity (unless y in P and count(P)==2? No, 2-1=1 < 3).
                    
                    # So, let's identify the "blocking" APs for x.
                    # These are APs P containing x such that ap_count[P] == 3.
                    blocking_aps = []
                    for ap_idx in aps_containing[x]:
                        if ap_count[ap_idx] == 3:
                            blocking_aps.append(ap_idx)
                    
                    if not blocking_aps:
                        # x is actually valid? We checked is_valid_addition in the loop above.
                        # If we are here, it means x was not added, so is_valid_addition returned False.
                        # So there must be blocking APs.
                        # Wait, is_valid_addition checks if ANY ap has count 3.
                        # So if we didn't add x, there is at least one blocking AP.
                        pass
                    
                    # We need to pick a y that is in the intersection of all blocking APs?
                    # No, we just need to pick a y such that for ALL blocking APs P, y is in P.
                    # If we pick y in P1 but not P2, and P2 is also blocking (count 3), then count(P2) remains 3, so x is still blocked.
                    # So y must be in the intersection of all blocking APs.
                    
                    if len(blocking_aps) == 0:
                        # This shouldn't happen if is_valid_addition failed
                        continue
                        
                    # Find intersection of elements in all blocking APs
                    # The elements are the other 3 terms in the AP (since x is in S? No, x is NOT in S).
                    # The AP contains x and 3 other elements.
                    # Those 3 other elements are in S (since count is 3).
                    
                    common_candidates = None
                    for ap_idx in blocking_aps:
                        a, b, c, d_val = ap_list[ap_idx]
                        # The other 3 elements
                        others = [v for v in (a, b, c, d_val) if v != x]
                        if common_candidates is None:
                            common_candidates = set(others)
                        else:
                            common_candidates &= set(others)
                            
                    if not common_candidates:
                        # No single removal can fix all blocking APs.
                        # We might need to remove multiple elements? 
                        # The current local search only does 1-for-1 swaps.
                        # So we skip this x.
                        continue
                        
                    # Pick a random y from common_candidates
                    y = random.choice(list(common_candidates))
                    
                    # Perform swap
                    # Remove y
                    S.remove(y)
                    for ap_idx in aps_containing[y]:
                        ap_count[ap_idx] -= 1
                    
                    # Add x
                    S.add(x)
                    for ap_idx in aps_containing[x]:
                        ap_count[ap_idx] += 1
                    
                    swapped = True
                    if len(S) > best_size:
                        best_S = S.copy()
                        best_ap_count = ap_count.copy()
                        best_size = len(S)
                    break
                
                if not swapped:
                    # Try a "kick" move: remove a random y, then greedily refill.
                    # This allows escaping local minima where no single swap works.
                    if S:
                        y = random.choice(list(S))
                        S_old = S.copy()
                        ap_count_old = ap_count.copy()
                        
                        S.remove(y)
                        for ap_idx in aps_containing[y]:
                            ap_count[ap_idx] -= 1
                        
                        # Greedy refill
                        candidates = [z for z in all_elements if z not in S]
                        random.shuffle(candidates)
                        
                        for z in candidates:
                            if is_valid_addition(ap_count, z):
                                S.add(z)
                                for ap_idx in aps_containing[z]:
                                    ap_count[ap_idx] += 1
                            
                        if len(S) > len(S_old):
                            best_S = S.copy()
                            best_ap_count = ap_count.copy()
                            best_size = len(S)
                            improved = True
                        else:
                            # Revert if no improvement
                            S = S_old
                            ap_count = ap_count_old
    
        return best_S, best_ap_count

    best_S = set()
    best_size = 0
    
    # Run multiple greedy restarts
    # Each greedy run is fast.
    # We will interleave greedy restarts and local searches.
    
    # First, do a few quick greedy restarts to get a good starting point
    for i in range(50):
        if time.time() - start_time > time_limit - 10:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        S, ap_count = greedy_construct(order)
        if len(S) > best_size:
            best_S = S
            best_size = len(S)
            
    # Now, take the best S and run local search
    if best_S:
        best_ap_count = [0] * num_aps
        for x in best_S:
            for ap_idx in aps_containing[x]:
                best_ap_count[ap_idx] += 1
        
        S_ls, ap_count_ls = local_search(best_S, best_ap_count)
        if len(S_ls) > best_size:
            best_S = S_ls
            best_size = len(S_ls)
    
    # If time remains, do more greedy restarts and local search on them
    while time.time() - start_time < time_limit - 2:
        order = list(range(1, N + 1))
        random.shuffle(order)
        S, ap_count = greedy_construct(order)
        
        # Quick local search for this new candidate
        # We limit the local search time to a small fraction to allow more restarts
        start_ls = time.time()
        ls_limit = 2.0 # 2 seconds per local search
        
        # We need a modified local search that respects a local time limit
        # Or just use the main one but break early if total time is close.
        # The main local_search loop checks `time.time() - start_time < time_limit`.
        # This is fine, but it might consume all remaining time.
        # Let's just run it. If it finishes, we get a result.
        
        S_ls, ap_count_ls = local_search(S, ap_count)
        
        if len(S_ls) > best_size:
            best_S = S_ls
            best_size = len(S_ls)
            
    return sorted(best_S)

