import time
import random
from math import gcd

def construct():
    deadline = time.time() + 85  # Safety margin before 90s kill
    
    def _lcm(a, b):
        return a // gcd(a, b) * b
    
    n = 128
    
    # Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(1, n + 1):
            lcm_table[a][b] = a // gcd(a, b) * b
    
    # Find all bad triples: (a,b,c) with a<b<c and lcm(a,b)=lcm(a,c)=lcm(b,c)
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            ab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == ab and lcm_table[b][c] == ab:
                    bad_triples.append((a, b, c))
    
    # For each x, build the set of forbidden pairs
    # forbidden[x] = dict mapping u -> bitmask of v values
    # such that if both u and v are in S, then x cannot be added
    forbidden = [dict() for _ in range(n + 1)]
    for a, b, c in bad_triples:
        # For element a: pairs (b,c) are forbidden
        if b not in forbidden[a]:
            forbidden[a][b] = 0
        forbidden[a][b] |= (1 << (c - 1))
        if c not in forbidden[a]:
            forbidden[a][c] = 0
        forbidden[a][c] |= (1 << (b - 1))
        
        # For element b: pairs (a,c) are forbidden
        if a not in forbidden[b]:
            forbidden[b][a] = 0
        forbidden[b][a] |= (1 << (c - 1))
        if c not in forbidden[b]:
            forbidden[b][c] = 0
        forbidden[b][c] |= (1 << (a - 1))
        
        # For element c: pairs (a,b) are forbidden
        if a not in forbidden[c]:
            forbidden[c][a] = 0
        forbidden[c][a] |= (1 << (b - 1))
        if b not in forbidden[c]:
            forbidden[c][b] = 0
        forbidden[c][b] |= (1 << (a - 1))
    
    # Precompute degree (number of forbidden pairs) for each element
    degrees = [0] * (n + 1)
    for x in range(1, n + 1):
        total = 0
        for v_mask in forbidden[x].values():
            total += bin(v_mask).count('1')
        degrees[x] = total
    
    def can_add(x, mask):
        """Check if x can be added to the set represented by mask."""
        fx = forbidden[x]
        for u, v_mask in fx.items():
            if mask & (1 << (u - 1)):
                if mask & v_mask:
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
    
    def local_search(mask, time_limit=None):
        if time_limit is None:
            time_limit = deadline - 2
        improved = True
        iters = 0
        max_iters = 200
        
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
                if time.time() > time_limit:
                    break
                mask_no_x = mask & ~(1 << (x - 1))
                new_mask = mask_no_x
                added_count = 0
                for y in range(1, n + 1):
                    if y == x:
                        continue
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
                for i in range(min(len(s_list), 12)):
                    if time.time() > time_limit - 1 or tried > 25:
                        break
                    x = s_list[i]
                    for j in range(i + 1, min(len(s_list), i + 6)):
                        if tried > 25:
                            break
                        y = s_list[j]
                        tried += 1
                        mask_no_xy = mask & ~(1 << (x - 1)) & ~(1 << (y - 1))
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
                            if popcount(new_mask) > popcount(mask):
                                mask = new_mask
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    continue

            # 4. Random perturbation if stuck
            if not improved and iters > 5:
                s_list = [i for i in range(1, n + 1) if (mask & (1 << (i - 1)))]
                if len(s_list) > 4:
                    k = random.randint(1, min(2, len(s_list) - 1))
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
    
    # Run multiple randomized greedy passes
    num_restarts = 3000
    
    for i in range(num_restarts):
        if time.time() > deadline - 6:
            break
        
        strategy = i % 7
        if strategy == 0:
            order = sorted(range(1, n + 1), key=lambda x: degrees[x] + random.random() * 0.5)
        elif strategy == 1:
            order = sorted(range(1, n + 1), key=lambda x: -degrees[x] + random.random() * 0.5)
        elif strategy == 2:
            order = list(range(1, n + 1))
            random.shuffle(order)
        elif strategy == 3:
            # Weighted random by inverse degree
            pool = list(range(1, n + 1))
            order = []
            while pool:
                w = [1.0 / (degrees[x] + 1) for x in pool]
                total_w = sum(w)
                r = random.random() * total_w
                cum = 0
                for idx, x in enumerate(pool):
                    cum += w[idx]
                    if cum >= r:
                        order.append(x)
                        pool.pop(idx)
                        break
        elif strategy == 4:
            # Sort by value (smaller numbers first - they tend to have more conflicts)
            order = sorted(range(1, n + 1), key=lambda x: x + random.random() * 5)
        elif strategy == 5:
            # Sort by number of prime factors (more factors = more lcm conflicts)
            def num_prime_factors(x):
                count = 0
                d = 2
                tmp = x
                while d * d <= tmp:
                    while tmp % d == 0:
                        count += 1
                        tmp //= d
                    d += 1
                if tmp > 1:
                    count += 1
                return count
            order = sorted(range(1, n + 1), key=lambda x: -num_prime_factors(x) + random.random())
        else:
            # Pure random
            order = list(range(1, n + 1))
            random.shuffle(order)
            
        mask = greedy(order)
        mask = local_search(mask, time_limit=deadline - 6)
        
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
