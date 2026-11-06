import time
import random
from math import gcd

def construct():
    deadline = time.time() + 85  # Safety margin
    
    def _lcm(a, b):
        return a // gcd(a, b) * b
    
    n = 128
    
    # First, find all bad triples
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
        
    # For each x, build: conflict_dict[x][u] = bitmask of v such that (u,v) in conflict_pairs[x]
    conflict_dict = [dict() for _ in range(n + 1)]
    for x in range(1, n + 1):
        for u, v in conflict_pairs[x]:
            if u not in conflict_dict[x]:
                conflict_dict[x][u] = 0
            conflict_dict[x][u] |= (1 << (v - 1))
            if v not in conflict_dict[x]:
                conflict_dict[x][v] = 0
            conflict_dict[x][v] |= (1 << (u - 1))
    
    # Fast can_add using conflict_dict: for x to be addable, no pair (u,v) with both in mask
    # Precompute for each x a list of (u, v_mask) pairs where v_mask has the v's
    conflict_list = [list() for _ in range(n + 1)]
    for x in range(1, n + 1):
        for u, v_mask in conflict_dict[x].items():
            conflict_list[x].append((u, v_mask))

    def can_add(x, mask):
        # x can be added iff for all (u, v_mask) in conflict_list[x], NOT (u in mask AND any v in v_mask in mask)
        for u, v_mask in conflict_list[x]:
            if (mask & (1 << (u - 1))) and (mask & v_mask):
                return False
        return True

    def popcount(x):
        return bin(x).count('1')

    def greedy(order, mask=0):
        selected = mask
        for x in order:
            if not (selected & (1 << (x - 1))):
                if can_add(x, selected):
                    selected |= (1 << (x - 1))
        return selected

    def try_add_two(mask_no_x, exclude_set):
        """Try to add two elements to mask_no_x, avoiding exclude_set. Return (new_mask, count_added)."""
        new_mask = mask_no_x
        added = 0
        # Try all pairs (y, z) not in mask and not in exclude_set
        for y in range(1, n + 1):
            if y in exclude_set or (new_mask & (1 << (y - 1))):
                continue
            if not can_add(y, new_mask):
                continue
            new_mask_y = new_mask | (1 << (y - 1))
            for z in range(y + 1, n + 1):
                if z in exclude_set or (new_mask_y & (1 << (z - 1))):
                    continue
                if can_add(z, new_mask_y):
                    return (new_mask_y | (1 << (z - 1)), 2)
        # If no pair found, try adding one
        for y in range(1, n + 1):
            if y in exclude_set or (new_mask & (1 << (y - 1))):
                continue
            if can_add(y, new_mask):
                return (new_mask | (1 << (y - 1)), 1)
        return (new_mask, 0)

    def local_search(mask, max_iters=50):
        improved = True
        iters = 0
        
        while improved and time.time() < deadline - 2 and iters < max_iters:
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
            best_new_mask = mask
            best_size = popcount(mask)
            s_list = [i for i in range(1, n + 1) if (mask & (1 << (i - 1)))]

            for x in s_list:
                if time.time() > deadline - 2:
                    break
                mask_no_x = mask & ~(1 << (x - 1))
                new_mask, added_count = try_add_two(mask_no_x, {x})
                if added_count >= 2:
                    sz = popcount(new_mask)
                    if sz > best_size:
                        best_size = sz
                        best_new_mask = new_mask

            if best_size > popcount(mask):
                mask = best_new_mask
                improved = True
                continue

            # 3. Try 2-for-3 swap: remove x, y, add z, w, v
            if time.time() < deadline - 3:
                s_list = [i for i in range(1, n + 1) if (mask & (1 << (i - 1)))]
                random.shuffle(s_list)
                best_new_mask = mask
                best_size = popcount(mask)
                for i in range(min(len(s_list), 20)):
                    if time.time() > deadline - 3:
                        break
                    x = s_list[i]
                    for j in range(i + 1, min(len(s_list), i + 5)):
                        y = s_list[j]
                        mask_no_xy = mask & ~(1 << (x - 1)) & ~(1 << (y - 1))
                        # Try to add three elements
                        new_mask = mask_no_xy
                        added_count = 0
                        for z in range(1, n + 1):
                            if z == x or z == y:
                                continue
                            if not (new_mask & (1 << (z - 1))):
                                if can_add(z, new_mask):
                                    new_mask |= (1 << (z - 1))
                                    added_count += 1
                                    if added_count == 3:
                                        break
                        if added_count >= 3:
                            sz = popcount(new_mask)
                            if sz > best_size:
                                best_size = sz
                                best_new_mask = new_mask
                if best_size > popcount(mask):
                    mask = best_new_mask
                    improved = True
                    continue

            # 4. Random perturbation if stuck
            if not improved and iters > 5:
                if len(s_list) > 4:
                    remove_idx = random.sample(range(len(s_list)), 2)
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

    # Precompute degrees for sorting
    degrees = [len(conflict_pairs[k]) for k in range(n + 1)]

    best_mask = 0
    best_size = 0

    # Run multiple randomized greedy passes with diverse orderings
    num_restarts = 800

    for i in range(num_restarts):
        if time.time() > deadline - 25:
            break

        strategy = i % 8
        if strategy == 0:
            order = sorted(range(1, n + 1), key=lambda x: degrees[x] + random.random() * 0.5)
        elif strategy == 1:
            order = sorted(range(1, n + 1), key=lambda x: -degrees[x] + random.random() * 0.5)
        elif strategy == 2:
            order = list(range(1, n + 1))
            random.shuffle(order)
        elif strategy == 3:
            # Weighted random: low degree elements more likely to come early
            weights = [1.0 / (1 + degrees[x]) for x in range(1, n + 1)]
            order = list(range(1, n + 1))
            random.shuffle(order)
            order.sort(key=lambda x: random.random() / weights[x - 1])
        elif strategy == 4:
            # Start with a random subset, then fill
            partial = random.sample(range(1, n + 1), 30)
            rest = [x for x in range(1, n + 1) if x not in partial]
            random.shuffle(rest)
            order = partial + rest
        elif strategy == 5:
            # Prime-heavy first (primes have fewer lcm conflicts)
            def is_prime(m):
                if m < 2: return False
                for p in range(2, int(m**0.5) + 1):
                    if m % p == 0: return False
                return True
            order = sorted(range(1, n + 1), key=lambda x: (not is_prime(x), degrees[x]))
        elif strategy == 6:
            # Power-of-2 heavy first
            def is_pow2(m):
                return (m & (m - 1)) == 0 and m > 0
            order = sorted(range(1, n + 1), key=lambda x: (not is_pow2(x), degrees[x]))
        else:
            # Anti-correlated: sort by negative degree with noise
            order = sorted(range(1, n + 1), key=lambda x: -degrees[x] * random.random())

        mask = greedy(order)
        mask = local_search(mask, max_iters=30)

        size = popcount(mask)
        if size > best_size:
            best_size = size
            best_mask = mask

    # Final polish with more iterations
    if time.time() < deadline - 2:
        best_mask = local_search(best_mask, max_iters=100)
    
    result = [i for i in range(1, n + 1) if (best_mask & (1 << (i - 1)))]
    
    if not result:
        result = [1]
        
    return result
