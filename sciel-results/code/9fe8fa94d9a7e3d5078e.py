import time
import random
import math

N = 256

def construct():
    # Precompute all 4-APs as (a, b, c, d) tuples
    four_aps = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            four_aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Precompute for each element x, the list of 4-AP indices that contain x
    # And for each 4-AP, the three other elements
    four_ap_by_elem = [[] for _ in range(N + 1)]
    four_ap_others = []
    for idx, (a, b, c, d_val) in enumerate(four_aps):
        four_ap_others.append((b, c, d_val))
        four_ap_by_elem[a].append(idx)
        four_ap_others.append((a, c, d_val))
        four_ap_by_elem[b].append(idx)
        four_ap_others.append((a, b, d_val))
        four_ap_by_elem[c].append(idx)
        four_ap_others.append((a, b, c))
        four_ap_by_elem[d_val].append(idx)
    
    def can_add_elem(A_mask, x):
        """Check if adding x to A_mask creates any 4-AP."""
        for idx in four_ap_by_elem[x]:
            o1, o2, o3 = four_ap_others[idx]
            if ((A_mask >> (o1 - 1)) & 1) and ((A_mask >> (o2 - 1)) & 1) and ((A_mask >> (o3 - 1)) & 1):
                return False
        return True
    
    def count_3aps_involving(x, A_mask):
        """Count 3-APs that include x and other elements in A_mask."""
        count = 0
        # x is middle: (x-d, x, x+d)
        for d in range(1, min(x - 1, N - x) + 1):
            a = x - d
            c = x + d
            if ((A_mask >> (a - 1)) & 1) and ((A_mask >> (c - 1)) & 1):
                count += 1
        # x is first: (x, x+d, x+2d)
        for d in range(1, (N - x) // 2 + 1):
            b = x + d
            c = x + 2 * d
            if b > N or c > N:
                break
            if ((A_mask >> (b - 1)) & 1) and ((A_mask >> (c - 1)) & 1):
                count += 1
        # x is third: (x-2d, x-d, x)
        for d in range(1, (x - 1) // 2 + 1):
            a = x - 2 * d
            b = x - d
            if a < 1:
                break
            if ((A_mask >> (a - 1)) & 1) and ((A_mask >> (b - 1)) & 1):
                count += 1
        return count
    
    def total_3aps(A_mask):
        """Count all distinct 3-APs in A_mask."""
        total = 0
        for x in range(1, N + 1):
            if (A_mask >> (x - 1)) & 1:
                total += count_3aps_involving(x, A_mask)
        return total // 3
    
    # Greedy initialization: add elements in order if possible
    A_mask = 0
    for x in range(1, N + 1):
        if can_add_elem(A_mask, x):
            A_mask |= (1 << (x - 1))
    
    current_score = total_3aps(A_mask)
    best_mask = A_mask
    best_score = current_score
    
    start_time = time.time()
    timeout = 55.0
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.9995
    min_T = 0.005
    
    while T > min_T and (time.time() - start_time) < timeout:
        # Pick a random element to remove
        in_set = [i + 1 for i in range(N) if (A_mask >> i) & 1]
        if not in_set:
            break
        x_remove = random.choice(in_set)
        
        # Pick a random element to add (not in set)
        not_in_set = [i + 1 for i in range(N) if not (A_mask >> i) & 1]
        if not not_in_set:
            break
        x_add = random.choice(not_in_set)
        
        # Perform move: Remove x_remove, Add x_add
        temp_mask = A_mask & ~(1 << (x_remove - 1))
        
        if not can_add_elem(temp_mask, x_add):
            T *= cooling
            continue
        
        new_mask = temp_mask | (1 << (x_add - 1))
        
        # Calculate score change
        removed_count = count_3aps_involving(x_remove, A_mask)
        added_count = count_3aps_involving(x_add, temp_mask)
        
        new_score = current_score - removed_count + added_count
        delta = new_score - current_score
        
        if delta > 0 or (delta == 0 and random.random() < 0.3) or random.random() < math.exp(delta / T):
            A_mask = new_mask
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_mask = A_mask
        
        T *= cooling
    
    return sorted([i + 1 for i in range(N) if (best_mask >> i) & 1])
