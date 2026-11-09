import random
import time
import itertools

def is_sidon(subset):
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def count_sidon_5_subsets(s):
    count = 0
    for combo in itertools.combinations(s, 5):
        if is_sidon(combo):
            count += 1
    return count

def largest_sidon_subset_size(s):
    # Check sizes 6, 5, 4
    for k in range(6, 3, -1):
        for combo in itertools.combinations(s, k):
            if is_sidon(combo):
                return k
    return 3

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Initial candidates
    # 1. AP
    cand_ap = list(range(12))
    # 2. Random
    random.seed(42)
    cand_rand = sorted(random.sample(range(129), 12))
    # 3. Union of two APs
    cand_union = sorted(list(range(0, 6, 1)) + list(range(20, 26, 1)))
    
    candidates = [cand_ap, cand_rand, cand_union]
    
    best_set = None
    best_count = float('inf')
    
    for c in candidates:
        cnt = count_sidon_5_subsets(c)
        if cnt < best_count:
            best_count = cnt
            best_set = c[:]
            
    # If we already found a set with 0 Sidon 5-subsets, we are done
    if best_count == 0:
        return best_set
        
    # Local search with simulated annealing
    current_set = best_set[:]
    current_count = best_count
    
    T = 10.0
    T_min = 0.1
    cooling = 0.999
    
    while time.time() < deadline:
        if best_count == 0:
            break
            
        idx = random.randint(0, 11)
        old_val = current_set[idx]
        new_val = random.randint(0, 128)
        if new_val in current_set and new_val != old_val:
            continue
            
        new_set = current_set[:]
        new_set[idx] = new_val
        new_count = count_sidon_5_subsets(new_set)
        
        delta = new_count - current_count
        
        if delta < 0 or random.random() < pow(2.718281828, -delta / T):
            current_set = new_set
            current_count = new_count
            
            if current_count < best_count:
                best_count = current_count
                best_set = new_set[:]
                
        T = max(T * cooling, T_min)
        
        # Occasionally restart
        if random.random() < 0.01:
            current_set = sorted(random.sample(range(129), 12))
            current_count = count_sidon_5_subsets(current_set)
            if current_count < best_count:
                best_count = current_count
                best_set = current_set[:]
            T = 5.0
            
    # Verify the best set
    # If best_count is 0, the largest Sidon subset is at most 4.
    # We know it's at least 4, so it's 4.
    # If best_count > 0, we need to find the actual largest Sidon subset size.
    if best_count == 0:
        return best_set
    else:
        # Fallback: find the actual largest Sidon subset size
        # and return the set with the minimum such size
        # But we already minimized the count of 5-subsets, which is a good proxy.
        # Let's just return the best set found.
        return best_set
