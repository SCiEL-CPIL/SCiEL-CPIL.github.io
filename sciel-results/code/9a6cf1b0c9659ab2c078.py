import random
import time

N = 512

# Precompute all 4-APs in [1, N]
all_4aps = []
aps_containing = {i: [] for i in range(1, N + 1)}

for d in range(1, (N - 1) // 3 + 1):
    for a in range(1, N - 3 * d + 1):
        ap = (a, a + d, a + 2 * d, a + 3 * d)
        all_4aps.append(ap)
        for x in ap:
            aps_containing[x].append(ap)

# For each element x, precompute the list of (ap_index, others_tuple)
# where others_tuple = the other 3 elements in that AP
element_aps = {i: [] for i in range(1, N + 1)}
for ap_idx, ap in enumerate(all_4aps):
    for x in ap:
        others = tuple(y for y in ap if y != x)
        element_aps[x].append((ap_idx, others))

def greedy_random():
    S = set()
    order = list(range(1, N + 1))
    random.shuffle(order)
    for x in order:
        can_add = True
        for ap_idx, others in element_aps[x]:
            if all(y in S for y in others):
                can_add = False
                break
        if can_add:
            S.add(x)
    return S

def pair_swap_local_search(S, deadline):
    """Try to improve S by swapping one element out and one in."""
    S = set(S)
    n = len(S)
    
    # Build conflict count: for each AP, how many of its elements are in S
    ap_count = [0] * len(all_4aps)
    for x in S:
        for ap_idx, others in element_aps[x]:
            ap_count[ap_idx] += 1
    
    # An AP is "satisfied" (violated) if ap_count[ap_idx] == 4
    # We want to maintain no violated APs.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try all pairs (x_out in S, x_in not in S)
        # To make this tractable, only try promising swaps
        S_list = list(S)
        not_in_S = [x for x in range(1, N + 1) if x not in S]
        
        random.shuffle(S_list)
        random.shuffle(not_in_S)
        
        for x_out in S_list:
            if time.time() >= deadline:
                break
            # Compute delta for removing x_out
            # Removing x_out decreases ap_count for all APs containing x_out
            remove_deltas = {}  # ap_idx -> delta
            for ap_idx, others in element_aps[x_out]:
                remove_deltas[ap_idx] = -1
            
            # After removal, check which APs become non-violated
            # (i.e., were at 4, now at 3)
            
            for x_in in not_in_S:
                if time.time() >= deadline:
                    break
                # Compute delta for adding x_in
                # Adding x_in increases ap_count for all APs containing x_in
                # But we must ensure no AP reaches 4 after both changes
                
                # Check if x_in can be added given x_out is removed
                can_add = True
                for ap_idx, others in element_aps[x_in]:
                    # Current count of this AP in S
                    cur = ap_count[ap_idx]
                    # If x_out is in this AP, cur will be decreased by 1
                    if ap_idx in remove_deltas:
                        cur += remove_deltas[ap_idx]
                    # After adding x_in, count becomes cur + 1
                    if cur + 1 > 3:
                        can_add = False
                        break
                
                if can_add:
                    # Perform the swap
                    S.discard(x_out)
                    S.add(x_in)
                    
                    # Update ap_count
                    for ap_idx, others in element_aps[x_out]:
                        ap_count[ap_idx] -= 1
                    for ap_idx, others in element_aps[x_in]:
                        ap_count[ap_idx] += 1
                    
                    improved = True
                    # Restart the outer loop since S changed
                    S_list = list(S)
                    not_in_S = [x for x in range(1, N + 1) if x not in S]
                    random.shuffle(S_list)
                    random.shuffle(not_in_S)
                    break
    
    return S

def construct():
    best_set = set()
    start_time = time.time()
    deadline = start_time + 80.0  # 10s margin for safety
    
    # Initial run
    current_best = greedy_random()
    best_set = current_best
    
    # Loop with restarts and local search
    while time.time() < deadline:
        candidate = greedy_random()
        
        # Apply local search if we have time
        remaining = deadline - time.time()
        if remaining > 5.0:
            candidate = pair_swap_local_search(candidate, deadline - 2.0)
        
        if len(candidate) > len(best_set):
            best_set = candidate
            
            # Try local search on the new best
            remaining = deadline - time.time()
            if remaining > 5.0:
                best_set = pair_swap_local_search(best_set, deadline - 2.0)
    
    return sorted(best_set)
