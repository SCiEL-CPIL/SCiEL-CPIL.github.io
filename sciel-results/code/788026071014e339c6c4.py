"""Program combining incremental checking with diverse ordering strategies for B_3[1] sets."""
# EVOLVE-BLOCK-START
import time
import random

def _check_valid(selected):
    """Return True if all unordered triple sums (with replacement) are unique."""
    n = len(selected)
    seen = set()
    for a in range(n):
        sa = selected[a]
        for b in range(a, n):
            sb = selected[b]
            sa_b = sa + sb
            for c in range(b, n):
                s = sa_b + selected[c]
                if s in seen:
                    return False
                seen.add(s)
    return True

def _greedy_build_order(order, limit, deadline):
    """Greedy construction following a given order, using incremental sum tracking."""
    vals = []
    existing_sums = set()
    
    for x in order:
        if time.time() > deadline:
            break
        if len(vals) >= 80:
            break
        if x > limit:
            break
        
        k = len(vals)
        conflict = False
        new_sums = []
        
        # Check (x,x,x)
        s = 3 * x
        if s in existing_sums:
            conflict = True
        else:
            new_sums.append(s)
        
        # Check (v, x, x) for v in vals
        if not conflict:
            for v in vals:
                s = v + 2 * x
                if s in existing_sums or s in new_sums:
                    conflict = True
                    break
                new_sums.append(s)
        
        # Check (v1, v2, x) for v1 <= v2 in vals
        if not conflict:
            for i in range(k):
                v1 = vals[i]
                for j in range(i, k):
                    v2 = vals[j]
                    s = v1 + v2 + x
                    if s in existing_sums or s in new_sums:
                        conflict = True
                        break
                    new_sums.append(s)
                if conflict:
                    break
        
        if not conflict:
            vals.append(x)
            existing_sums.update(new_sums)
    
    return vals

def _greedy_build_order_fast(order, limit, deadline):
    """Faster greedy using list-based sum tracking with early exit."""
    vals = []
    existing_sums = set()
    max_sum = 3 * limit
    
    for x in order:
        if time.time() > deadline:
            break
        if len(vals) >= 80:
            break
        if x > limit:
            break
        
        k = len(vals)
        conflict = False
        new_sums = []
        
        # Check (x,x,x)
        s = 3 * x
        if s in existing_sums:
            conflict = True
        else:
            new_sums.append(s)
        
        # Check (v, x, x) for v in vals
        if not conflict:
            for v in vals:
                s = v + 2 * x
                if s in existing_sums or s in new_sums:
                    conflict = True
                    break
                new_sums.append(s)
        
        # Check (v1, v2, x) for v1 <= v2 in vals
        if not conflict:
            for i in range(k):
                v1 = vals[i]
                for j in range(i, k):
                    v2 = vals[j]
                    s = v1 + v2 + x
                    if s in existing_sums or s in new_sums:
                        conflict = True
                        break
                    new_sums.append(s)
                if conflict:
                    break
        
        if not conflict:
            vals.append(x)
            existing_sums.update(new_sums)
    
    return vals

def _greedy_from_start(start_list, limit, deadline):
    """Greedy construction starting from a given list, adding smallest valid numbers."""
    base = sorted(start_list)
    if not base:
        base = [1]
    
    # Compute initial sums
    sums = set()
    n = len(base)
    for a in range(n):
        sa = base[a]
        for b in range(a, n):
            sb = base[b]
            sa_b = sa + sb
            for c in range(b, n):
                s = sa_b + base[c]
                if s in sums:
                    return base  # Invalid start
                sums.add(s)
    
    while len(base) < 80:
        if time.time() > deadline:
            break
        next_min = max(base) + 1
        found = False
        for x in range(next_min, limit + 1):
            if x > limit:
                break
            k = len(base)
            conflict = False
            new_sums = []
            
            s = 3 * x
            if s in sums:
                conflict = True
            else:
                new_sums.append(s)
            
            if not conflict:
                for v in base:
                    s = v + 2 * x
                    if s in sums or s in new_sums:
                        conflict = True
                        break
                    new_sums.append(s)
            
            if not conflict:
                for i in range(k):
                    v1 = base[i]
                    for j in range(i, k):
                        v2 = base[j]
                        s = v1 + v2 + x
                        if s in sums or s in new_sums:
                            conflict = True
                            break
                        new_sums.append(s)
                    if conflict:
                        break
            
            if not conflict:
                base.append(x)
                sums.update(new_sums)
                found = True
                break
        if not found:
            break
    return base

def _try_add_element(base, x, sums):
    """Try to add element x to base, returning (success, new_sums) or (False, None)."""
    k = len(base)
    conflict = False
    new_sums = []
    
    s = 3 * x
    if s in sums:
        return False, None
    new_sums.append(s)
    
    for v in base:
        s = v + 2 * x
        if s in sums or s in new_sums:
            return False, None
        new_sums.append(s)
    
    for i in range(k):
        v1 = base[i]
        for j in range(i, k):
            v2 = base[j]
            s = v1 + v2 + x
            if s in sums or s in new_sums:
                return False, None
            new_sums.append(s)
    
    return True, new_sums

def _greedy_complete(base, limit, deadline):
    """Complete a base set by greedily adding the smallest valid numbers."""
    base = sorted(base)
    if not base:
        return base
    
    # Compute initial sums
    sums = set()
    n = len(base)
    for a in range(n):
        sa = base[a]
        for b in range(a, n):
            sb = base[b]
            sa_b = sa + sb
            for c in range(b, n):
                s = sa_b + base[c]
                if s in sums:
                    return base  # Invalid start
                sums.add(s)
    
    while len(base) < 80:
        if time.time() > deadline:
            break
        next_min = max(base) + 1
        found = False
        for x in range(next_min, limit + 1):
            if x > limit:
                break
            success, new_sums = _try_add_element(base, x, sums)
            if success:
                base.append(x)
                sums.update(new_sums)
                found = True
                break
        if not found:
            break
    return base

def _compute_sums(lst):
    """Compute all unordered triple sums for a sorted list. Returns None if invalid."""
    sums = set()
    n = len(lst)
    for a in range(n):
        sa = lst[a]
        for b in range(a, n):
            sb = lst[b]
            sa_b = sa + sb
            for c in range(b, n):
                s = sa_b + lst[c]
                if s in sums:
                    return None
                sums.add(s)
    return sums

def _local_search_jump(base, limit, deadline, max_iter=50):
    """Try to improve by removing one element and greedily adding more."""
    base = sorted(base)
    if not base:
        return base
    
    best = list(base)
    best_size = len(base)
    
    current_sums = _compute_sums(base)
    if current_sums is None:
        return best
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        # Try removing each element and see if we can add more
        for remove_idx in range(len(base)):
            if time.time() > deadline:
                break
            
            candidate = [x for i, x in enumerate(base) if i != remove_idx]
            cand_sums = _compute_sums(candidate)
            if cand_sums is None:
                continue
            
            temp_base = list(candidate)
            temp_sums = cand_sums
            next_min = max(temp_base) + 1 if temp_base else 1
            
            for x in range(next_min, limit + 1):
                if time.time() > deadline:
                    break
                if len(temp_base) >= 80:
                    break
                
                success, new_sums = _try_add_element(temp_base, x, temp_sums)
                if success:
                    temp_base.append(x)
                    temp_sums.update(new_sums)
            
            if len(temp_base) > best_size:
                best = temp_base
                best_size = len(best)
                base = temp_base
                current_sums = _compute_sums(base)
                improved = True
                break
        
        if not improved:
            break
    
    return best

def _local_search_swap(base, limit, deadline, max_iter=30):
    """Try to improve by swapping elements to free up space."""
    base = sorted(base)
    if not base:
        return base
    
    best = list(base)
    best_size = len(base)
    n = len(base)
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        # Try swapping each pair
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                
                # Swap base[i] and base[j]
                base[i], base[j] = base[j], base[i]
                base.sort()
                
                swapped_sums = _compute_sums(base)
                if swapped_sums is None:
                    base[i], base[j] = base[j], base[i]
                    base.sort()
                    continue
                
                # Try to add elements greedily
                temp_base = list(base)
                temp_sums = swapped_sums
                next_min = max(temp_base) + 1 if temp_base else 1
                
                for x in range(next_min, limit + 1):
                    if time.time() > deadline:
                        break
                    if len(temp_base) >= 80:
                        break
                    
                    success, new_sums = _try_add_element(temp_base, x, temp_sums)
                    if success:
                        temp_base.append(x)
                        temp_sums.update(new_sums)
                
                if len(temp_base) > best_size:
                    best = temp_base
                    best_size = len(best)
                    base = temp_base
                    improved = True
                    break
                else:
                    # Revert swap
                    base[i], base[j] = base[j], base[i]
                    base.sort()
            
            if improved:
                break
        
        if not improved:
            break
    
    return best

def _greedy_from_seed(seed_vals, limit, deadline):
    """Greedy construction starting from given seed values, adding smallest valid numbers."""
    base = sorted(seed_vals)
    if not base:
        base = [1]
    
    # Compute initial sums
    sums = _compute_sums(base)
    if sums is None:
        return base
    
    while len(base) < 80:
        if time.time() > deadline:
            break
        next_min = max(base) + 1
        found = False
        for x in range(next_min, limit + 1):
            if x > limit:
                break
            success, new_sums = _try_add_element(base, x, sums)
            if success:
                base.append(x)
                sums.update(new_sums)
                found = True
                break
        if not found:
            break
    return base

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    limit = 4096
    best_result = []
    
    # Phase 1: Standard greedy from 1 upward
    order1 = list(range(1, limit + 1))
    result = _greedy_build_order(order1, limit, deadline)
    if len(result) > len(best_result):
        best_result = result
    
    if time.time() > deadline:
        return best_result
    
    # Phase 2: Local search jump on the best result
    result = _local_search_jump(best_result, limit, deadline, max_iter=40)
    if len(result) > len(best_result):
        best_result = result
    
    if time.time() > deadline:
        return best_result
    
    # Phase 2b: Local search swap
    result = _local_search_swap(best_result, limit, deadline, max_iter=20)
    if len(result) > len(best_result):
        best_result = result
    
    if time.time() > deadline:
        return best_result
    
    # Phase 3: Diverse ordering strategies with more attempts
    rng = random.Random(42)
    
    for attempt in range(300):
        if time.time() > deadline - 5:  # Leave 5s buffer
            break
        
        strategy = attempt % 8
        
        if strategy == 0:
            # Random shuffle
            order = list(range(1, limit + 1))
            rng.shuffle(order)
        elif strategy == 1:
            # Random offset: start from offset, then wrap around
            offset = rng.randint(0, 1000)
            order = list(range(1 + offset, limit + 1)) + list(range(1, 1 + offset))
        elif strategy == 2:
            # Two-phase: small numbers shuffled, then large numbers shuffled
            small = list(range(1, 500))
            large = list(range(500, limit + 1))
            rng.shuffle(small)
            rng.shuffle(large)
            order = small + large
        elif strategy == 3:
            # Exponential outward from random center
            center = rng.randint(1, limit)
            order = [center]
            for d in range(1, limit):
                if center - d >= 1:
                    order.append(center - d)
                if center + d <= limit:
                    order.append(center + d)
        elif strategy == 4:
            # Reverse order: large to small
            order = list(range(limit, 0, -1))
        elif strategy == 5:
            # Skip-based: take every kth number, then fill gaps
            k = rng.randint(2, 5)
            order = list(range(1, limit + 1, k))
            remaining = [x for x in range(1, limit + 1) if x % k != 1]
            rng.shuffle(remaining)
            order.extend(remaining)
        elif strategy == 6:
            # Geometric spacing: 1, 3, 9, 27, ... then fill
            order = []
            v = 1
            while v <= limit:
                order.append(v)
                v *= 3
            remaining = [x for x in range(1, limit + 1) if x not in set(order)]
            rng.shuffle(remaining)
            order.extend(remaining)
        else:
            # Random with bias toward smaller numbers
            order = []
            for x in range(1, limit + 1):
                if rng.random() < 0.3:
                    order.append(x)
            remaining = [x for x in range(1, limit + 1) if x not in set(order)]
            rng.shuffle(remaining)
            order.extend(remaining)
        
        result = _greedy_build_order(order, limit, deadline)
        if len(result) > len(best_result):
            best_result = result
            
            # If we found a better result, try local search
            if time.time() < deadline - 15:
                result2 = _local_search_jump(best_result, limit, deadline, max_iter=25)
                if len(result2) > len(best_result):
                    best_result = result2
                result3 = _local_search_swap(best_result, limit, deadline, max_iter=15)
                if len(result3) > len(best_result):
                    best_result = result3
    
    # Phase 4: Try seeding with random subsets and greedy completion
    for seed_attempt in range(20):
        if time.time() > deadline - 5:
            break
        
        seed_size = rng.randint(1, 5)
        seed = rng.sample(range(1, limit + 1), seed_size)
        result = _greedy_from_seed(seed, limit, deadline)
        if len(result) > len(best_result):
            best_result = result
            if time.time() < deadline - 10:
                result2 = _local_search_jump(best_result, limit, deadline, max_iter=20)
                if len(result2) > len(best_result):
                    best_result = result2
                result3 = _local_search_swap(best_result, limit, deadline, max_iter=10)
                if len(result3) > len(best_result):
                    best_result = result3
    
    return best_result

# EVOLVE-BLOCK-END
