import time
import random
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    limit = 512
    deadline = start_time + 80  # Safety margin (evaluator kills at 90s)
    
    # Precompute compatibility bitsets
    # compat[i] is an integer where bit j is 1 if lcm(i, j) <= 512
    compat = [0] * (limit + 1)
    for i in range(1, limit + 1):
        mask = 0
        for j in range(1, limit + 1):
            if i * j // gcd(i, j) <= limit:
                mask |= (1 << j)
        compat[i] = mask

    # Helper to check if x is compatible with set S (represented as bitset S_mask)
    def is_compatible_with_set(x, S_mask):
        return (compat[x] & S_mask) == S_mask

    # Greedy construction with multiple different orderings
    def greedy_build(order):
        S = []
        S_mask = 0
        for x in order:
            if is_compatible_with_set(x, S_mask):
                S.append(x)
                S_mask |= (1 << x)
        return S, S_mask

    # Try several different initial orderings
    best_S = []
    best_S_mask = 0
    best_size = 0
    
    # Ordering 1: natural order 1..512
    S, S_mask = greedy_build(range(1, limit + 1))
    if len(S) > best_size:
        best_S = S
        best_S_mask = S_mask
        best_size = len(S)
    
    # Ordering 2: reverse order 512..1
    S, S_mask = greedy_build(range(limit, 0, -1))
    if len(S) > best_size:
        best_S = S
        best_S_mask = S_mask
        best_size = len(S)
    
    # Ordering 3: sorted by number of divisors (ascending) - more "primitive" first
    def num_divisors(n):
        count = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                count += 2
                if i != n // i:
                    count -= 1
        return count
    
    order3 = sorted(range(1, limit + 1), key=num_divisors)
    S, S_mask = greedy_build(order3)
    if len(S) > best_size:
        best_S = S
        best_S_mask = S_mask
        best_size = len(S)
    
    # Ordering 4: sorted by value (which is natural, already tried)
    # Ordering 5: random orderings
    for trial in range(10):
        if time.time() > deadline - 30:
            break
        order = list(range(1, limit + 1))
        random.shuffle(order)
        S, S_mask = greedy_build(order)
        if len(S) > best_size:
            best_S = S
            best_S_mask = S_mask
            best_size = len(S)
    
    # Local search: 1-for-1 swaps and 2-for-3 swaps
    def local_search(S, S_mask, max_time):
        S = S[:]
        S_mask = S_mask
        best_local_size = len(S)
        best_local_S = S[:]
        best_local_mask = S_mask
        
        no_improve = 0
        max_no_improve = 200
        
        while time.time() < max_time and no_improve < max_no_improve:
            improved = False
            
            # Try 2-for-3 swap
            if len(S) >= 2:
                # Randomly pick a pair to remove
                for attempt in range(50):
                    u_idx = random.randint(0, len(S) - 1)
                    v_idx = random.randint(0, len(S) - 1)
                    if u_idx == v_idx:
                        continue
                    u = S[u_idx]
                    v = S[v_idx]
                    
                    S_mask_temp = S_mask & ~(1 << u) & ~(1 << v)
                    
                    # Find candidates compatible with S_mask_temp
                    candidates = []
                    for x in range(1, limit + 1):
                        if (S_mask & (1 << x)):
                            continue
                        if is_compatible_with_set(x, S_mask_temp):
                            candidates.append(x)
                    
                    if len(candidates) < 3:
                        continue
                    
                    # Try to find 3 mutually compatible elements in candidates
                    random.shuffle(candidates)
                    c_limit = min(30, len(candidates))
                    
                    found = False
                    for i in range(c_limit):
                        c1 = candidates[i]
                        for j in range(i + 1, c_limit):
                            c2 = candidates[j]
                            if not (compat[c1] & (1 << c2)):
                                continue
                            mask_req = compat[c1] & compat[c2]
                            for k in range(j + 1, c_limit):
                                c3 = candidates[k]
                                if (mask_req & (1 << c3)):
                                    # Found a valid triple
                                    new_S = S[:]
                                    new_S.remove(u)
                                    new_S.remove(v)
                                    new_S.extend([c1, c2, c3])
                                    
                                    # Verify
                                    valid = True
                                    new_mask = 0
                                    for x in new_S:
                                        if not is_compatible_with_set(x, new_mask):
                                            valid = False
                                            break
                                        new_mask |= (1 << x)
                                    
                                    if valid and len(new_S) > best_local_size:
                                        S = new_S
                                        S_mask = new_mask
                                        best_local_size = len(S)
                                        best_local_S = S[:]
                                        best_local_mask = S_mask
                                        improved = True
                                        no_improve = 0
                                        found = True
                                        break
                                if found:
                                    break
                        if found:
                            break
                    if found:
                        break
            
            # Try 1-for-1 swap (to change structure)
            if not improved and len(S) > 0:
                u = random.choice(S)
                S_mask_temp = S_mask & ~(1 << u)
                possible_v = []
                for v in range(1, limit + 1):
                    if not (S_mask & (1 << v)):
                        if is_compatible_with_set(v, S_mask_temp):
                            possible_v.append(v)
                if possible_v:
                    v = random.choice(possible_v)
                    new_S = S[:]
                    new_S.remove(u)
                    new_S.append(v)
                    valid = True
                    new_mask = 0
                    for x in new_S:
                        if not is_compatible_with_set(x, new_mask):
                            valid = False
                            break
                        new_mask |= (1 << x)
                    if valid:
                        S = new_S
                        S_mask = new_mask
                        # Don't count as improvement for size, but reset no_improve
                        no_improve = 0
                        improved = True
            
            if not improved:
                no_improve += 1
        
        return best_local_S, best_local_mask, best_local_size
    
    # Run local search on the best greedy result
    if time.time() < deadline - 10:
        S_ls, mask_ls, size_ls = local_search(best_S, best_S_mask, deadline - 5)
        if size_ls > best_size:
            best_S = S_ls
            best_S_mask = mask_ls
            best_size = size_ls
    
    # Try multiple random restarts with local search
    restarts = 5
    for r in range(restarts):
        if time.time() > deadline - 15:
            break
        order = list(range(1, limit + 1))
        random.shuffle(order)
        S, S_mask = greedy_build(order)
        
        # Quick local search
        S_ls, mask_ls, size_ls = local_search(S, S_mask, deadline - 5)
        if size_ls > best_size:
            best_S = S_ls
            best_S_mask = mask_ls
            best_size = size_ls
    
    return best_S
