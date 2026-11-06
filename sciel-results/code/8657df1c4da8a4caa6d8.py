import time
import random
from math import gcd

def construct():
    deadline = time.time() + 85  # Safety margin
    
    def _lcm(a, b):
        return a // gcd(a, b) * b
    
    n = 128
    
    # Find all bad triples
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            ab = _lcm(a, b)
            for c in range(b + 1, n + 1):
                if _lcm(a, c) == ab and _lcm(b, c) == ab:
                    bad_triples.append((a, b, c))
    
    # Build conflict structures
    conflict_pairs = [set() for _ in range(n + 1)]
    for a, b, c in bad_triples:
        conflict_pairs[a].add((b, c))
        conflict_pairs[b].add((a, c))
        conflict_pairs[c].add((a, b))
        
    # Precompute conflict bitmasks for faster checking
    # conflict_dict[x][u] = bitmask of v such that (u,v) is a blocking pair for x
    conflict_dict = [dict() for _ in range(n + 1)]
    for x in range(1, n + 1):
        for u, v in conflict_pairs[x]:
            if u not in conflict_dict[x]:
                conflict_dict[x][u] = 0
            conflict_dict[x][u] |= (1 << (v - 1))
            if v not in conflict_dict[x]:
                conflict_dict[x][v] = 0
            conflict_dict[x][v] |= (1 << (u - 1))
    
    def can_add(x, mask):
        cd = conflict_dict[x]
        for u, v_mask in cd.items():
            if mask & (1 << (u - 1)):
                if mask & v_mask:
                    return False
        return True

    def greedy(order, mask=0):
        selected = mask
        for x in order:
            if not (selected & (1 << (x - 1))):
                if can_add(x, selected):
                    selected |= (1 << (x - 1))
        return selected

    def popcount(x):
        return bin(x).count('1')
    
    def local_search(mask, time_limit=None):
        if time_limit is None:
            time_limit = deadline - 2
        improved = True
        iters = 0
        max_iters = 100
        
        while improved and time.time() < time_limit and iters < max_iters:
            improved = False
            iters += 1
            
            # 1. Try to add any missing element
            for x in range(1, n + 1):
                if not (mask & (1 << (x - 1))):
                    if can_add(x, mask):
                        mask |= (1 << (x - 1))
                        improved = True
                        break
            if improved:
                continue

            # 2. Try 1-for-2 swap: remove x, add y, z
            found_swap = False
            s_list = [i for i in range(1, n + 1) if (mask & (1 << (i - 1)))]
            for x in s_list:
                if time.time() > time_limit: break
                mask_no_x = mask & ~(1 << (x - 1))
                added_count = 0
                new_mask = mask_no_x
                for y in range(1, n + 1):
                    if y == x: continue
                    if not (new_mask & (1 << (y - 1))):
                        if can_add(y, new_mask):
                            new_mask |= (1 << (y - 1))
                            added_count += 1
                            if added_count == 2:
                                break
                if added_count >= 2:
                    if popcount(new_mask) > popcount(mask):
                        mask = new_mask
                        improved = True
                        found_swap = True
                        break
            if found_swap:
                continue

            # 3. Try 2-for-3 swap: remove x, y, add z, w, v
            if time.time() < time_limit - 1:
                s_list = [i for i in range(1, n + 1) if (mask & (1 << (i - 1)))]
                random.shuffle(s_list)
                tried = 0
                for i in range(min(len(s_list), 15)):
                    if time.time() > time_limit - 1 or tried > 30: break
                    x = s_list[i]
                    for j in range(i+1, min(len(s_list), i+8)):
                        if tried > 30: break
                        y = s_list[j]
                        tried += 1
                        mask_no_xy = mask & ~(1 << (x - 1)) & ~(1 << (y - 1))
                        added_count = 0
                        new_mask = mask_no_xy
                        for z in range(1, n + 1):
                            if z == x or z == y: continue
                            if not (new_mask & (1 << (z - 1))):
                                if can_add(z, new_mask):
                                    new_mask |= (1 << (z - 1))
                                    added_count += 1
                                    if added_count == 3:
                                        break
                        if added_count >= 3:
                            if popcount(new_mask) > popcount(mask):
                                mask = new_mask
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    continue

            # 4. Random perturbation if stuck
            if not improved and iters > 3:
                s_list = [i for i in range(1, n + 1) if (mask & (1 << (i - 1)))]
                if len(s_list) > 4:
                    k = random.randint(1, min(3, len(s_list) - 1))
                    remove_idx = random.sample(range(len(s_list)), k)
                    perturb_mask = mask
                    for idx in remove_idx:
                        perturb_mask &= ~(1 << (s_list[idx] - 1))
                    for x in range(1, n + 1):
                        if not (perturb_mask & (1 << (x - 1))):
                            if can_add(x, perturb_mask):
                                perturb_mask |= (1 << (x - 1))
                    if popcount(perturb_mask) > popcount(mask):
                        mask = perturb_mask
                        improved = True
                        continue

            if not improved:
                break
                
        return mask

    best_mask = 0
    best_size = 0
    
    degrees = [len(conflict_pairs[k]) for k in range(n+1)]
    
    # Run multiple randomized greedy passes
    num_restarts = 2000
    
    for i in range(num_restarts):
        if time.time() > deadline - 8:
            break
            
        strategy = i % 6
        if strategy == 0:
            order = sorted(range(1, n + 1), key=lambda x: degrees[x] + random.random()*0.5)
        elif strategy == 1:
            order = sorted(range(1, n + 1), key=lambda x: -degrees[x] + random.random()*0.5)
        elif strategy == 2:
            order = list(range(1, n + 1))
            random.shuffle(order)
        elif strategy == 3:
            # Weighted random by inverse degree
            weights = [1.0 / (degrees[x] + 1) for x in range(1, n + 1)]
            order = []
            pool = list(range(1, n + 1))
            while pool:
                total_w = sum(weights[pool.index(x) - 1] for x in pool)
                r = random.random() * total_w
                cum = 0
                for x in pool:
                    cum += weights[x - 1]
                    if cum >= r:
                        order.append(x)
                        pool.remove(x)
                        break
        elif strategy == 4:
            # Sort by lcm value (smaller lcm first)
            order = sorted(range(1, n + 1), key=lambda x: x * (n // x + 1) + random.random())
        else:
            # Pure random
            order = list(range(1, n + 1))
            random.shuffle(order)
            
        mask = greedy(order)
        mask = local_search(mask, time_limit=deadline - 8)
        
        size = popcount(mask)
        if size > best_size:
            best_size = size
            best_mask = mask
            
    # Final polish with more time
    remaining = deadline - time.time()
    if remaining > 3:
        best_mask = local_search(best_mask, time_limit=deadline - 1)
    
    result = [i for i in range(1, n + 1) if (best_mask & (1 << (i - 1)))]
    
    if not result:
        result = [1]
        
    return result
