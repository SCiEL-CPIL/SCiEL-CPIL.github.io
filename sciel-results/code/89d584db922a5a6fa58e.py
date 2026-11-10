import time
import random

def construct():
    deadline = time.time() + 55  # Safety margin
    n = 256
    
    # 1. Precompute conflict masks
    conflict_mask = [0] * (n + 1)
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                conflict_mask[a] |= (1 << b)
                conflict_mask[b] |= (1 << a)
    
    # Helper: count bits
    def popcount(x):
        return bin(x).count('1')
    
    # Helper: check if vertex v can be added to mask
    def can_add(v, mask):
        return (conflict_mask[v] & mask) == 0
    
    # 2. Strong initial solution: all odd numbers (mutually compatible)
    # For odd a, b: a+b is even, a*b is odd, so (a*b) % (a+b) != 0 always
    odd_mask = 0
    for i in range(1, n + 1, 2):
        odd_mask |= (1 << i)
    
    # Sort evens by degree ascending (prefer low-conflict evens)
    evens = list(range(2, n + 1, 2))
    evens.sort(key=lambda x: popcount(conflict_mask[x]))
    
    # Greedily add compatible evens to odd base
    def greedy_add_evens(base_mask, order):
        current = base_mask
        for v in order:
            if (current >> v) & 1:
                continue
            if can_add(v, current):
                current |= (1 << v)
        return current
    
    best_mask = greedy_add_evens(odd_mask, evens)
    best_size = popcount(best_mask)
    
    # 3. Local search with add + 1-for-2 swap moves
    def local_search(mask, size, deadline_local):
        improved = True
        while improved and time.time() < deadline_local:
            improved = False
            # Try adding any vertex
            for x in range(1, n + 1):
                if not (mask >> x) & 1:
                    if can_add(x, mask):
                        mask |= (1 << x)
                        size += 1
                        improved = True
            # Try 1-for-2: remove y, add x1 and x2
            for y in range(1, n + 1):
                if not (mask >> y) & 1:
                    continue
                new_mask = mask & ~(1 << y)
                candidates = []
                for x in range(1, n + 1):
                    if not (new_mask >> x) & 1:
                        if can_add(x, new_mask):
                            candidates.append(x)
                if len(candidates) >= 2:
                    for i in range(len(candidates)):
                        for j in range(i + 1, len(candidates)):
                            x1, x2 = candidates[i], candidates[j]
                            if can_add(x2, new_mask | (1 << x1)):
                                new_mask |= (1 << x1) | (1 << x2)
                                size += 1
                                improved = True
                                mask = new_mask
                                break
                        if improved:
                            break
                if improved:
                    break
        return mask, size
    
    best_mask, best_size = local_search(best_mask, best_size, deadline - 5)
    
    # 4. Multiple randomized restarts
    rng = random.Random(42)
    for restart in range(10):
        if time.time() > deadline - 10:
            break
        random_odds = [i for i in range(1, n + 1, 2) if rng.random() < 0.8]
        base = 0
        for i in random_odds:
            base |= (1 << i)
        random_evens = rng.sample(evens, min(len(evens), 20))
        for e in random_evens:
            if can_add(e, base):
                base |= (1 << e)
        m = greedy_add_evens(base, evens)
        sz = popcount(m)
        if sz > best_size:
            best_size = sz
            best_mask = m
        m, sz = local_search(m, sz, deadline - 5)
        if sz > best_size:
            best_size = sz
            best_mask = m
    
    # 5. Branch and Bound with dynamic vertex ordering
    def branch_and_bound(available, current, count):
        nonlocal best_size, best_mask
        if time.time() > deadline - 2:
            return
        if count + popcount(available) <= best_size:
            return
        if available == 0:
            if count > best_size:
                best_size = count
                best_mask = current
            return
        
        # Pick pivot: vertex in available with max degree in available subgraph
        pivot = 0
        max_deg = -1
        temp = available
        while temp:
            v = (temp & -temp).bit_length() - 1
            deg = popcount(conflict_mask[v] & available)
            if deg > max_deg:
                max_deg = deg
                pivot = v
            temp &= temp - 1
        
        if pivot == 0:
            return
        
        # Branch 1: Include pivot
        new_avail_inc = available & ~(1 << pivot) & ~conflict_mask[pivot]
        branch_and_bound(new_avail_inc, current | (1 << pivot), count + 1)
        
        if time.time() > deadline - 2:
            return
        
        # Branch 2: Exclude pivot
        new_avail_exc = available & ~(1 << pivot)
        branch_and_bound(new_avail_exc, current, count)
    
    all_vertices = 0
    for i in range(1, n + 1):
        all_vertices |= (1 << i)
    
    branch_and_bound(all_vertices, 0, 0)
    
    # 6. Final local search polish
    if time.time() < deadline - 2:
        best_mask, best_size = local_search(best_mask, best_size, deadline - 1)
    
    result = [x for x in range(1, n + 1) if (best_mask >> x) & 1]
    return result
