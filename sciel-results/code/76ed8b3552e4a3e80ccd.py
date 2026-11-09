import time
import random

# EVOLVE-BLOCK-START

def construct():
    N = 512
    start_time = time.time()
    time_limit = 55.0  # Leave 5s margin for evaluator overhead

    # Precompute AP index
    ap_by_elem = [[] for _ in range(N)]
    
    for a in range(N):
        for d in range(1, N // 3 + 1):
            a3 = a + 3 * d
            if a3 >= N:
                break
            p0, p1, p2, p3 = a, a + d, a + 2 * d, a + 3 * d
            ap_by_elem[p0].append((p1, p2, p3))
            ap_by_elem[p1].append((p0, p2, p3))
            ap_by_elem[p2].append((p0, p1, p3))
            ap_by_elem[p3].append((p0, p1, p2))
            
    # Precompute degrees for sorting
    degrees = [len(lst) for lst in ap_by_elem]

    def can_add(mask, x):
        for (o1, o2, o3) in ap_by_elem[x]:
            if (mask >> o1) & 1 and (mask >> o2) & 1 and (mask >> o3) & 1:
                return False
        return True
    
    def greedy_with_order(order):
        mask = 0
        for x in order:
            if can_add(mask, x):
                mask |= (1 << x)
        return mask
    
    def count_bits(mask):
        return bin(mask).count('1')
        
    best_mask = 0
    best_size = 0
    
    # Strategy 1: Natural order
    mask1 = greedy_with_order(list(range(N)))
    best_mask, best_size = mask1, count_bits(mask1)
    
    # Strategy 2: Reverse order
    mask2 = greedy_with_order(list(range(N - 1, -1, -1)))
    if count_bits(mask2) > best_size:
        best_mask, best_size = mask2, count_bits(mask2)
        
    # Strategy 3: Degree order (lowest degree first)
    degree_order = sorted(range(N), key=lambda x: degrees[x])
    mask3 = greedy_with_order(degree_order)
    if count_bits(mask3) > best_size:
        best_mask, best_size = mask3, count_bits(mask3)
        
    # Strategy 4: Random orders
    random.seed(42)
    for _ in range(20):
        if time.time() - start_time > time_limit:
            break
        order = list(range(N))
        random.shuffle(order)
        m = greedy_with_order(order)
        sz = count_bits(m)
        if sz > best_size:
            best_size = sz
            best_mask = m
            
    # Local Search
    current_mask = best_mask
    current_size = best_size
    
    stuck_counter = 0
    max_stuck = 50
    
    while time.time() - start_time < time_limit:
        made_progress = False
        
        # 1. Try to add any element
        order = list(range(N))
        random.shuffle(order)
        
        for x in order:
            if (current_mask >> x) & 1:
                continue
            if can_add(current_mask, x):
                current_mask |= (1 << x)
                current_size += 1
                made_progress = True
                if current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask
        if made_progress:
            stuck_counter = 0
            continue
            
        # 2. Try swaps
        in_set = [x for x in range(N) if (current_mask >> x) & 1]
        if not in_set:
            break
            
        swap_success = False
        for _ in range(10):
            if time.time() - start_time > time_limit:
                break
            y = random.choice(in_set)
            new_mask = current_mask & ~(1 << y)
            
            added_count = 0
            for x in range(N):
                if (new_mask >> x) & 1:
                    continue
                if can_add(new_mask, x):
                    new_mask |= (1 << x)
                    added_count += 1
            
            if added_count >= 1:
                current_mask = new_mask
                current_size += added_count - 1
                if current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask
                swap_success = True
                break
                
        if swap_success:
            stuck_counter = 0
            continue
            
        stuck_counter += 1
        if stuck_counter > max_stuck:
            break
            
    # Final polish
    for x in range(N):
        if not (current_mask >> x) & 1:
            if can_add(current_mask, x):
                current_mask |= (1 << x)
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask
                    
    result = [x + 1 for x in range(N) if (best_mask >> x) & 1]
    return sorted(result)

# EVOLVE-BLOCK-END
