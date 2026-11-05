import time
from math import lcm as math_lcm

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    N = 128
    numbers = list(range(1, N + 1))
    
    # Precompute bad masks
    bad_mask = [[0] * N for _ in range(N)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = math_lcm(a, b)
            mask = 0
            for c in range(1, N + 1):
                if c == a or c == b:
                    continue
                if math_lcm(a, c) == lab and math_lcm(b, c) == lab:
                    mask |= (1 << (c - 1))
            bad_mask[a - 1][b - 1] = mask
            bad_mask[b - 1][a - 1] = mask
    
    # Precompute for each element x, the list of pairs (a,b) where x is in the bad mask
    # This helps with efficient conflict counting
    element_conflict_pairs = [[] for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            mask = bad_mask[a - 1][b - 1]
            m = mask
            while m:
                lowbit = m & (-m)
                c_idx = lowbit.bit_length()
                c_val = c_idx
                element_conflict_pairs[c_val].append((a, b))
                m ^= lowbit
    
    def calc_conflict_mask(S):
        mask = 0
        S_list = list(S)
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                mask |= bad_mask[a - 1][b - 1]
        return mask
    
    def try_add(x, S, conflict_mask):
        if (conflict_mask >> (x - 1)) & 1:
            return False, conflict_mask
        
        new_mask = conflict_mask
        for y in S:
            new_mask |= bad_mask[x - 1][y - 1]
        
        return True, new_mask
    
    def greedy_add(order):
        S = set()
        conflict_mask = 0
        
        for x in order:
            if time.time() > deadline - 5:
                break
            ok, new_mask = try_add(x, S, conflict_mask)
            if ok:
                S.add(x)
                conflict_mask = new_mask
        
        return S, conflict_mask
    
    def local_search(S, conflict_mask):
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 3:
                break
            
            # Try to add any remaining element
            for x in range(1, N + 1):
                if x not in S:
                    ok, new_mask = try_add(x, S, conflict_mask)
                    if ok:
                        S.add(x)
                        conflict_mask = new_mask
                        improved = True
                        break
            
            if improved:
                continue
            
            # Try 1-for-2 swaps
            S_list = list(S)
            for a in S_list:
                if time.time() > deadline - 3:
                    break
                
                S_temp = S - {a}
                mask_temp = calc_conflict_mask(S_temp)
                
                added_x = False
                for x in range(1, N + 1):
                    if x not in S_temp:
                        ok, new_mask = try_add(x, S_temp, mask_temp)
                        if ok:
                            S_temp.add(x)
                            mask_temp = new_mask
                            added_x = True
                            break
                
                if not added_x:
                    continue
                
                for y in range(1, N + 1):
                    if y not in S_temp:
                        ok, new_mask = try_add(y, S_temp, mask_temp)
                        if ok:
                            S = S_temp
                            conflict_mask = new_mask
                            improved = True
                            break
                
                if improved:
                    break
        
        return S, conflict_mask
    
    def num_distinct_primes(n):
        count = 0
        d = 2
        while d * d <= n:
            if n % d == 0:
                count += 1
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            count += 1
        return count
    
    # Count how many bad triples each element is in
    triple_count = [0] * (N + 1)
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            cnt = bin(bad_mask[a - 1][b - 1]).count('1')
            triple_count[a] += cnt
            triple_count[b] += cnt
            m = bad_mask[a - 1][b - 1]
            while m:
                lowbit = m & (-m)
                c_val = lowbit.bit_length()
                triple_count[c_val] += 1
                m ^= lowbit
    
    best_set = set()
    best_size = 0
    
    # Multiple greedy orders
    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        sorted(range(1, N + 1), key=lambda x: (-triple_count[x], x)),
        sorted(range(1, N + 1), key=lambda x: (triple_count[x], x)),
        sorted(range(1, N + 1), key=lambda x: (-num_distinct_primes(x), x)),
        sorted(range(1, N + 1), key=lambda x: (num_distinct_primes(x), x)),
    ]
    
    import random
    for _ in range(5):
        random_order = list(range(1, N + 1))
        random.shuffle(random_order)
        orders.append(random_order)
    
    for order in orders:
        if time.time() > deadline - 10:
            break
        
        S, conflict_mask = greedy_add(order)
        S, conflict_mask = local_search(S, conflict_mask)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    # Try starting from full set and removing
    if time.time() < deadline - 15:
        S = set(range(1, N + 1))
        conflict_mask = calc_conflict_mask(S)
        S, conflict_mask = local_search(S, conflict_mask)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
    
    if not best_set:
        return [1]
    
    return list(best_set)
