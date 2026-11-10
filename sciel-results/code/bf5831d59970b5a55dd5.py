import time
import random

N = 512
DEADLINE = 55.0

def construct():
    start_time = time.time()
    
    # Precompute all 4-APs
    ap_list = []
    aps_containing = [[] for _ in range(N + 1)]
    
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
    
    def is_valid_addition(ap_count, x):
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
    
    def local_search(S, ap_count, time_limit):
        best_S = S.copy()
        best_ap_count = ap_count.copy()
        best_size = len(S)
        
        while time.time() < time_limit:
            # Try to add any element
            added = False
            for x in range(1, N + 1):
                if x not in S and is_valid_addition(ap_count, x):
                    S.add(x)
                    for ap_idx in aps_containing[x]:
                        ap_count[ap_idx] += 1
                    added = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = S.copy()
                        best_ap_count = ap_count.copy()
                    break
            
            if added:
                continue
            
            # Try swaps: remove y, add x
            swapped = False
            for _ in range(200):
                x = random.randint(1, N)
                if x in S:
                    continue
                y = random.choice(list(S))
                
                # Check if S \ {y} + {x} is valid
                # For each AP containing x: count_in_S_minus_y < 3
                valid = True
                for ap_idx in aps_containing[x]:
                    current_count = ap_count[ap_idx]
                    y_in_ap = 0
                    ap = ap_list[ap_idx]
                    if ap[0] == y or ap[1] == y or ap[2] == y or ap[3] == y:
                        y_in_ap = 1
                    if current_count - y_in_ap >= 3:
                        valid = False
                        break
                
                if valid:
                    # Perform swap
                    S.remove(y)
                    for ap_idx in aps_containing[y]:
                        ap_count[ap_idx] -= 1
                    S.add(x)
                    for ap_idx in aps_containing[x]:
                        ap_count[ap_idx] += 1
                    swapped = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = S.copy()
                        best_ap_count = ap_count.copy()
                    break
            
            if not swapped:
                # Try remove-and-refill: remove one element, try to add multiple
                removed = False
                for _ in range(50):
                    y = random.choice(list(S))
                    S_old = S.copy()
                    ap_count_old = ap_count.copy()
                    
                    S.remove(y)
                    for ap_idx in aps_containing[y]:
                        ap_count[ap_idx] -= 1
                    
                    # Try to add elements
                    added_count = 0
                    for z in range(1, N + 1):
                        if z not in S and is_valid_addition(ap_count, z):
                            S.add(z)
                            for ap_idx in aps_containing[z]:
                                ap_count[ap_idx] += 1
                            added_count += 1
                    
                    if len(S) > len(S_old):
                        removed = True
                        if len(S) > best_size:
                            best_size = len(S)
                            best_S = S.copy()
                            best_ap_count = ap_count.copy()
                    else:
                        S = S_old
                        ap_count = ap_count_old
                    if removed:
                        break
                if not removed:
                    break
        
        return best_S, best_ap_count
    
    best_S = set()
    best_size = 0
    
    # Multi-start: run many greedy restarts, then local search on best candidates
    num_restarts = 300
    
    for i in range(num_restarts):
        if time.time() > DEADLINE - 5:
            break
        
        order = list(range(1, N + 1))
        random.shuffle(order)
        S, ap_count = greedy_construct(order)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = S.copy()
        
        # Local search on this candidate if we have time
        if time.time() < DEADLINE - 10:
            S_ls, ap_count_ls = local_search(S, ap_count, DEADLINE - 1)
            if len(S_ls) > best_size:
                best_size = len(S_ls)
                best_S = S_ls.copy()
    
    return sorted(best_S)
