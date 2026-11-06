import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    N = 512
    # Precompute compatibility bitsets using Python integers as bitsets
    # can[i] has bit (j-1) set if lcm(i,j) <= N
    can = [0] * (N + 1)
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            g = gcd(i, j)
            if (i * j) // g <= N:
                mask |= (1 << (j - 1))
        can[i] = mask
    
    # Precompute degrees
    degrees = [0] * (N + 1)
    for i in range(1, N + 1):
        degrees[i] = bin(can[i]).count('1')
    
    def is_compatible(x, S_mask):
        return (can[x] & S_mask) == S_mask
    
    def greedy_construct(order):
        S_mask = 0
        S_list = []
        for x in order:
            if is_compatible(x, S_mask):
                S_mask |= (1 << (x - 1))
                S_list.append(x)
        return S_list
    
    def local_search(S_list, time_limit):
        S_mask = 0
        for x in S_list:
            S_mask |= (1 << (x - 1))
        
        improved = True
        max_iter = 300
        iter_count = 0
        
        while improved and iter_count < max_iter and time.time() < time_limit:
            iter_count += 1
            improved = False
            n = len(S_list)
            
            # 1. Try to add any compatible x (prioritize by degree)
            best_add = -1
            best_add_deg = -1
            for x in range(1, N + 1):
                if not (S_mask & (1 << (x - 1))):
                    if is_compatible(x, S_mask):
                        # Use degree as heuristic
                        deg = degrees[x]
                        if deg > best_add_deg:
                            best_add_deg = deg
                            best_add = x
            if best_add > 0:
                S_mask |= (1 << (best_add - 1))
                S_list.append(best_add)
                improved = True
                continue

            # 2. Try to swap: remove u, add v
            swapped = False
            u_indices = list(range(n))
            random.shuffle(u_indices)
            for idx in u_indices:
                if time.time() > time_limit:
                    break
                u = S_list[idx]
                S_mask_without_u = S_mask ^ (1 << (u - 1))
                best_v = -1
                best_v_score = -1
                for v in range(1, N + 1):
                    if not (S_mask & (1 << (v - 1))):
                        if is_compatible(v, S_mask_without_u):
                            # Score: prefer v with high degree to remaining elements
                            score = degrees[v]
                            if score > best_v_score:
                                best_v_score = score
                                best_v = v
                if best_v > 0:
                    S_mask = S_mask_without_u | (1 << (best_v - 1))
                    S_list[idx] = best_v
                    swapped = True
                    break
            
            if swapped:
                improved = True
                continue

            # 3. Try to remove 1, add 2 (increase size by 1)
            found_pair = False
            u_indices = list(range(n))
            random.shuffle(u_indices)
            
            for idx in u_indices:
                if time.time() > time_limit:
                    break
                u = S_list[idx]
                S_mask_without_u = S_mask ^ (1 << (u - 1))
                
                candidates = []
                for v in range(1, N + 1):
                    if not (S_mask & (1 << (v - 1))):
                        if is_compatible(v, S_mask_without_u):
                            candidates.append(v)
                
                if len(candidates) < 2:
                    continue
                
                # Sort by degree descending
                candidates.sort(key=lambda x: -degrees[x])
                
                for i in range(len(candidates)):
                    v = candidates[i]
                    for j in range(i + 1, len(candidates)):
                        w = candidates[j]
                        if (can[v] >> (w - 1)) & 1:
                            S_mask = S_mask_without_u | (1 << (v - 1)) | (1 << (w - 1))
                            S_list.pop(idx)
                            S_list.append(v)
                            S_list.append(w)
                            found_pair = True
                            break
                    if found_pair:
                        break
                if found_pair:
                    break
            
            if found_pair:
                improved = True
                continue

            # 4. Try to remove 2, add 3 (increase size by 1)
            if n >= 2:
                found_triple = False
                u_indices = list(range(n))
                random.shuffle(u_indices)
                
                for i1 in u_indices:
                    if time.time() > time_limit:
                        break
                    u1 = S_list[i1]
                    remaining_indices = [i for i in u_indices if i != i1]
                    for i2 in remaining_indices:
                        u2 = S_list[i2]
                        S_mask_without_u1u2 = S_mask ^ (1 << (u1 - 1)) ^ (1 << (u2 - 1))
                        
                        candidates = []
                        for v in range(1, N + 1):
                            if not (S_mask & (1 << (v - 1))):
                                if is_compatible(v, S_mask_without_u1u2):
                                    candidates.append(v)
                        
                        if len(candidates) < 3:
                            continue
                        
                        candidates.sort(key=lambda x: -degrees[x])
                        
                        found = False
                        for a in range(len(candidates)):
                            ca = candidates[a]
                            for b in range(a + 1, len(candidates)):
                                cb = candidates[b]
                                if not ((can[ca] >> (cb - 1)) & 1):
                                    continue
                                for c in range(b + 1, len(candidates)):
                                    cc = candidates[c]
                                    if ((can[ca] >> (cc - 1)) & 1) and ((can[cb] >> (cc - 1)) & 1):
                                        S_mask = S_mask_without_u1u2 | (1 << (ca - 1)) | (1 << (cb - 1)) | (1 << (cc - 1))
                                        if i1 < i2:
                                            S_list.pop(i2)
                                            S_list.pop(i1)
                                        else:
                                            S_list.pop(i1)
                                            S_list.pop(i2)
                                        S_list.append(ca)
                                        S_list.append(cb)
                                        S_list.append(cc)
                                        found = True
                                        break
                                if found:
                                    break
                            if found:
                                break
                        if found:
                            found_triple = True
                            break
                    if found_triple:
                        break
                
                if found_triple:
                    improved = True
                    continue

        return S_list

    best_size = 0
    best_set = []
    
    # Phase 1: Structural seeds - divisors of highly composite numbers
    # Key insight: any valid set S has lcm(S) = L <= 512, so S ⊆ divisors(L)
    # Maximize d(L) over L <= 512
    def count_divisors(n):
        count = 0
        i = 1
        while i * i <= n:
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
            i += 1
        return count
    
    def get_divisors(n):
        divs = []
        i = 1
        while i * i <= n:
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
            i += 1
        return divs
    
    # Find all L with many divisors
    div_counts = []
    for L in range(1, N + 1):
        d = count_divisors(L)
        div_counts.append((d, L))
    div_counts.sort(reverse=True)
    
    # Try top 20 highly composite numbers as seeds
    seen_L = set()
    for d, L in div_counts[:20]:
        if time.time() > deadline:
            break
        if L in seen_L:
            continue
        seen_L.add(L)
        divs = get_divisors(L)
        s = local_search(divs, deadline - 5)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s[:]

    # Phase 2: Greedy with diverse orderings
    order_asc = sorted(range(1, N + 1), key=lambda x: degrees[x])
    order_desc = sorted(range(1, N + 1), key=lambda x: -degrees[x])
    order_natural = list(range(1, N + 1))
    order_reverse = list(range(N, 0, -1))
    
    for order in [order_asc, order_desc, order_natural, order_reverse]:
        if time.time() > deadline:
            break
        s = greedy_construct(order)
        s = local_search(s, deadline - 5)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s[:]

    # Phase 3: Random restarts with perturbation
    random.seed(42)
    while time.time() < deadline - 3:
        order = list(range(1, N + 1))
        random.shuffle(order)
        s = greedy_construct(order)
        s = local_search(s, deadline - 3)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s[:]
        
        # Perturb best solution
        if best_set and random.random() < 0.3:
            s_perturbed = best_set[:]
            if s_perturbed:
                num_remove = random.randint(1, max(1, len(s_perturbed) // 4))
                for _ in range(num_remove):
                    if s_perturbed:
                        idx = random.randint(0, len(s_perturbed) - 1)
                        s_perturbed.pop(idx)
                s = local_search(s_perturbed, deadline - 3)
                if len(s) > best_size:
                    best_size = len(s)
                    best_set = s[:]

    return best_set
