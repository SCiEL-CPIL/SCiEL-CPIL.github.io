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
        
    # Strategy 4: Random orders with weighted sampling
    random.seed(42)
    for _ in range(50):
        if time.time() - start_time > time_limit:
            break
        # Use degree-weighted random order
        weights = [1.0 / (degrees[x] + 1) for x in range(N)]
        order = []
        remaining = list(range(N))
        while remaining:
            total_w = sum(weights[i] for i in remaining)
            r = random.random() * total_w
            cum = 0
            for idx, x in enumerate(remaining):
                cum += weights[x]
                if r <= cum:
                    order.append(x)
                    remaining.remove(x)
                    break
        m = greedy_with_order(order)
        sz = count_bits(m)
        if sz > best_size:
            best_size = sz
            best_mask = m
            
    # Local Search with multiple strategies
    current_mask = best_mask
    current_size = best_size
    
    stuck_counter = 0
    max_stuck = 100
    no_improve_count = 0
    
    while time.time() - start_time < time_limit:
        made_progress = False
        
        # 1. Try to add any element (systematic scan)
        for x in range(N):
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
            no_improve_count = 0
            continue
            
        # 2. Try single swaps (remove one, add many)
        in_set = [x for x in range(N) if (current_mask >> x) & 1]
        if not in_set:
            break
            
        # Priority: try elements with high degree first (they block more)
        in_set_sorted = sorted(in_set, key=lambda x: degrees[x], reverse=True)
        
        swap_success = False
        for y in in_set_sorted:
            if time.time() - start_time > time_limit:
                break
            new_mask = current_mask & ~(1 << y)
            
            added_count = 0
            for x in range(N):
                if (new_mask >> x) & 1:
                    continue
                if can_add(new_mask, x):
                    new_mask |= (1 << x)
                    added_count += 1
            
            if added_count >= 2:  # Need to gain at least 1 net
                current_mask = new_mask
                current_size += added_count - 1
                if current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask
                swap_success = True
                break
                
        if swap_success:
            stuck_counter = 0
            no_improve_count = 0
            continue
            
        # 3. Try pair swaps (remove two, add three or more)
        pair_swap_success = False
        if len(in_set) >= 2:
            # Sample some pairs to try
            num_pairs_to_try = min(50, len(in_set) * (len(in_set) - 1) // 2)
            random.shuffle(in_set)
            for i in range(min(num_pairs_to_try, len(in_set))):
                for j in range(i + 1, min(i + 5, len(in_set))):
                    if time.time() - start_time > time_limit:
                        break
                    y1, y2 = in_set[i], in_set[j]
                    new_mask = current_mask & ~(1 << y1) & ~(1 << y2)
                    
                    added_count = 0
                    for x in range(N):
                        if (new_mask >> x) & 1:
                            continue
                        if can_add(new_mask, x):
                            new_mask |= (1 << x)
                            added_count += 1
                    
                    if added_count >= 3:  # Net gain of at least 1
                        current_mask = new_mask
                        current_size += added_count - 2
                        if current_size > best_size:
                            best_size = current_size
                            best_mask = current_mask
                        pair_swap_success = True
                        break
                if pair_swap_success or time.time() - start_time > time_limit:
                    break
                    
        if pair_swap_success:
            stuck_counter = 0
            no_improve_count = 0
            continue
            
        # 4. Random perturbation: remove 1-2 random elements, then greedily re-add
        if stuck_counter > 20:
            perturbation_success = False
            for _ in range(5):
                if time.time() - start_time > time_limit:
                    break
                # Remove 1-2 random elements
                num_remove = random.randint(1, 2)
                to_remove = random.sample(in_set, min(num_remove, len(in_set)))
                perturbed_mask = current_mask
                for y in to_remove:
                    perturbed_mask &= ~(1 << y)
                
                # Greedily add back
                for x in range(N):
                    if (perturbed_mask >> x) & 1:
                        continue
                    if can_add(perturbed_mask, x):
                        perturbed_mask |= (1 << x)
                
                perturbed_size = count_bits(perturbed_mask)
                if perturbed_size > current_size:
                    current_mask = perturbed_mask
                    current_size = perturbed_size
                    if current_size > best_size:
                        best_size = current_size
                        best_mask = current_mask
                    perturbation_success = True
                    break
                # Accept with some probability if equal (to escape local optima)
                elif perturbed_size == current_size and random.random() < 0.1:
                    current_mask = perturbed_mask
                    perturbation_success = True
                    break
                    
            if perturbation_success:
                stuck_counter = 0
                no_improve_count = 0
                continue
        
        stuck_counter += 1
        no_improve_count += 1
        if stuck_counter > max_stuck:
            break
            
    # Final polish: multiple passes
    for _ in range(5):
        changed = False
        for x in range(N):
            if not (current_mask >> x) & 1:
                if can_add(current_mask, x):
                    current_mask |= (1 << x)
                    current_size += 1
                    changed = True
                    if current_size > best_size:
                        best_size = current_size
                        best_mask = current_mask
        if not changed:
            break
                    
    result = [x + 1 for x in range(N) if (best_mask >> x) & 1]
    return sorted(result)

# EVOLVE-BLOCK-END
