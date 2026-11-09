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

def _local_search_jump(base, limit, deadline, max_iter=50):
    """Try to improve by removing one element and greedily adding more."""
    base = sorted(base)
    if not base:
        return base
    
    best = list(base)
    best_size = len(base)
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        # Try removing each element and see if we can add more
        for remove_idx in range(len(base)):
            if time.time() > deadline:
                break
            
            candidate = [x for i, x in enumerate(base) if i != remove_idx]
            
            # Recompute sums
            sums = set()
            n = len(candidate)
            valid = True
            for a in range(n):
                sa = candidate[a]
                for b in range(a, n):
                    sb = candidate[b]
                    sa_b = sa + sb
                    for c in range(b, n):
                        s = sa_b + candidate[c]
                        if s in sums:
                            valid = False
                            break
                        sums.add(s)
                    if not valid:
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Try to add elements greedily
            temp_base = list(candidate)
            temp_sums = sums
            added = 0
            next_min = max(temp_base) + 1 if temp_base else 1
            
            for x in range(next_min, limit + 1):
                if time.time() > deadline:
                    break
                if len(temp_base) >= 80:
                    break
                
                k = len(temp_base)
                conflict = False
                new_sums = []
                
                s = 3 * x
                if s in temp_sums:
                    conflict = True
                else:
                    new_sums.append(s)
                
                if not conflict:
                    for v in temp_base:
                        s = v + 2 * x
                        if s in temp_sums or s in new_sums:
                            conflict = True
                            break
                        new_sums.append(s)
                
                if not conflict:
                    for i in range(k):
                        v1 = temp_base[i]
                        for j in range(i, k):
                            v2 = temp_base[j]
                            s = v1 + v2 + x
                            if s in temp_sums or s in new_sums:
                                conflict = True
                                break
                            new_sums.append(s)
                        if conflict:
                            break
                
                if not conflict:
                    temp_base.append(x)
                    temp_sums.update(new_sums)
                    added += 1
            
            if len(temp_base) > best_size:
                best = temp_base
                best_size = len(best)
                base = temp_base
                improved = True
                break
        
        if not improved:
            break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin before 90s kill
    
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
    result = _local_search_jump(best_result, limit, deadline, max_iter=30)
    if len(result) > len(best_result):
        best_result = result
    
    if time.time() > deadline:
        return best_result
    
    # Phase 3: Power-of-2 core strategy
    # Start with powers of 2, then extend greedily
    pow2_core = [2**i for i in range(0, 13) if 2**i <= limit]
    if _check_valid(pow2_core):
        result = _greedy_from_start(pow2_core, limit, deadline)
        if len(result) > len(best_result):
            best_result = result
        result = _local_search_jump(best_result, limit, deadline, max_iter=20)
        if len(result) > len(best_result):
            best_result = result
    
    if time.time() > deadline:
        return best_result
    
    # Phase 4: Diverse ordering strategies
    rng = random.Random(42)
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        strategy = attempt % 6
        
        if strategy == 0:
            # Random shuffle
            order = list(range(1, limit + 1))
            rng.shuffle(order)
        elif strategy == 1:
            # Random offset: start from offset, then wrap around
            offset = rng.randint(0, 500)
            order = list(range(1 + offset, limit + 1)) + list(range(1, 1 + offset))
        elif strategy == 2:
            # Two-phase: small numbers shuffled, then large numbers shuffled
            small = list(range(1, 300))
            large = list(range(300, limit + 1))
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
        else:
            # Base-spaced candidates: try different base spacings
            base = rng.randint(2, 8)
            density = rng.choice([0.1, 0.2, 0.3, 0.5])
            pool = []
            for x in range(1, limit + 1):
                if (x - 1) % base == 0 or rng.random() < density:
                    pool.append(x)
            rng.shuffle(pool)
            order = pool
        
        result = _greedy_build_order(order, limit, deadline)
        if len(result) > len(best_result):
            best_result = result
            
            # If we found a better result, try local search
            if time.time() < deadline - 10:
                result2 = _local_search_jump(best_result, limit, deadline, max_iter=20)
                if len(result2) > len(best_result):
                    best_result = result2
    
    # Phase 5: Multiple passes to add missing numbers to the best result
    if time.time() < deadline - 5:
        chosen = list(best_result)
        chosen_set = set(chosen)
        
        for pass_num in range(3):
            if time.time() > deadline - 3:
                break
            
            sums = set()
            n = len(chosen)
            for a in range(n):
                sa = chosen[a]
                for b in range(a, n):
                    sb = chosen[b]
                    sa_b = sa + sb
                    for c in range(b, n):
                        sums.add(sa_b + chosen[c])
            
            # Different candidate orderings per pass
            if pass_num == 0:
                candidates = list(range(1, limit + 1))
            elif pass_num == 1:
                candidates = list(range(limit, 0, -1))
            else:
                candidates = list(range(1, limit + 1))
                rng.shuffle(candidates)
            
            added_any = False
            for v in candidates:
                if time.time() > deadline - 3:
                    break
                if v in chosen_set:
                    continue
                k = len(chosen)
                conflict = False
                new_sums = []
                
                s = 3 * v
                if s in sums:
                    conflict = True
                else:
                    new_sums.append(s)
                
                if not conflict:
                    for u in chosen:
                        s = 2 * v + u
                        if s in sums or s in new_sums:
                            conflict = True
                            break
                        new_sums.append(s)
                        s = v + 2 * u
                        if s in sums or s in new_sums:
                            conflict = True
                            break
                        new_sums.append(s)
                
                if not conflict:
                    for ui in range(k):
                        u = chosen[ui]
                        for wi in range(ui, k):
                            w = chosen[wi]
                            s = v + u + w
                            if s in sums or s in new_sums:
                                conflict = True
                                break
                            new_sums.append(s)
                        if conflict:
                            break
                
                if not conflict:
                    sums.update(new_sums)
                    chosen.append(v)
                    chosen_set.add(v)
                    added_any = True
            
            if not added_any:
                break
        
        if _check_valid(chosen) and len(chosen) > len(best_result):
            best_result = chosen
    
    # Sanity: ensure within range, distinct, positive
    best_result = [v for v in best_result if 1 <= v <= limit]
    best_result = sorted(set(best_result))
    
    return best_result

# EVOLVE-BLOCK-END
