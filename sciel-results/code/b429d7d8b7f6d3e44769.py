import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit

    N = 180
    bits = [1 << i for i in range(N)]  # bits[x-1] is the bitmask for element x

    # Precompute forbidden triples and pair masks
    # pair_masks[x] is a list of (bit(y) | bit(z)) for all pairs {y, z}
    # such that {x, y, z} is a forbidden triple.
    pair_masks = [[] for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        a_sq = a * a
        # Find factor pairs (u, v) of a_sq such that u < v
        # b = a + u, c = a + v
        # We need b != c, so u != v.
        # We need b, c <= N.
        for u in range(1, int(a_sq ** 0.5) + 1):
            if a_sq % u == 0:
                v = a_sq // u
                if u == v:
                    continue
                b = a + u
                c = a + v
                if b > N or c > N:
                    continue
                # Ensure distinct
                if b == c:
                    continue
                # We have a triple (a, b, c)
                # Add masks
                # For a: conflict with pair (b, c)
                pair_masks[a].append(bits[b-1] | bits[c-1])
                # For b: conflict with pair (a, c)
                pair_masks[b].append(bits[a-1] | bits[c-1])
                # For c: conflict with pair (a, b)
                pair_masks[c].append(bits[a-1] | bits[b-1])

    def can_add(x, S_mask):
        """Check if x can be added to S_mask without creating a forbidden triple."""
        for pm in pair_masks[x]:
            if (S_mask & pm) == pm:
                return False
        return True

    def popcount(mask):
        return bin(mask).count('1')

    def greedy_solve(order):
        S_mask = 0
        for x in order:
            if can_add(x, S_mask):
                S_mask |= bits[x-1]
        return S_mask

    def local_search(S_mask, size):
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Phase 1: Try to add any element
            for x in range(1, N + 1):
                if not (S_mask & bits[x-1]):
                    if can_add(x, S_mask):
                        S_mask |= bits[x-1]
                        size += 1
                        improved = True
            
            # Phase 2: Try to swap (remove one, add one or more)
            # This is expensive, so we do it less frequently or with a limit
            if not improved:
                # Try removing one element and re-adding greedily
                # We do this for a subset of elements to save time
                candidates = [x for x in range(1, N + 1) if (S_mask & bits[x-1])]
                random.shuffle(candidates)
                for rem in candidates[:10]: # Limit to 10 random removals per iteration
                    S_temp = S_mask & ~bits[rem-1]
                    size_temp = size - 1
                    # Greedy add
                    for y in range(1, N + 1):
                        if not (S_temp & bits[y-1]):
                            if can_add(y, S_temp):
                                S_temp |= bits[y-1]
                                size_temp += 1
                    if size_temp > size:
                        S_mask = S_temp
                        size = size_temp
                        improved = True
                        break
        
        return S_mask, size

    # Initial solutions
    best_mask = 0
    best_size = 0

    # 1. Greedy descending
    order_desc = list(range(N, 0, -1))
    m = greedy_solve(order_desc)
    s = popcount(m)
    if s > best_size:
        best_mask = m
        best_size = s

    # 2. Greedy ascending
    order_asc = list(range(1, N + 1))
    m = greedy_solve(order_asc)
    s = popcount(m)
    if s > best_size:
        best_mask = m
        best_size = s

    # 3. Greedy random restarts
    for _ in range(50):
        if time.time() > deadline - 30:
            break
        order_rand = list(range(1, N + 1))
        random.shuffle(order_rand)
        m = greedy_solve(order_rand)
        s = popcount(m)
        if s > best_size:
            best_mask = m
            best_size = s

    # Iterated Local Search
    current_mask = best_mask
    current_size = best_size
    
    while time.time() < deadline - 10:
        # Local search
        current_mask, current_size = local_search(current_mask, current_size)
        
        if current_size > best_size:
            best_mask = current_mask
            best_size = current_size
            
        # Perturbation: remove k random elements
        k = random.randint(2, 5)
        elems = [x for x in range(1, N + 1) if (current_mask & bits[x-1])]
        if len(elems) > k:
            to_remove = random.sample(elems, k)
            for x in to_remove:
                current_mask &= ~bits[x-1]
            current_size = popcount(current_mask)
        else:
            # If set is small, restart with random greedy
            order_rand = list(range(1, N + 1))
            random.shuffle(order_rand)
            current_mask = greedy_solve(order_rand)
            current_size = popcount(current_mask)

    return [x for x in range(1, N + 1) if (best_mask & bits[x-1])]
