import time
from math import gcd

def construct():
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    
    N = 256
    candidates = list(range(2, N + 1))
    
    # Precompute: for each a, list of (b,c) pairs where a | b*c, b != c, all in [2,256]
    # Actually, for checking if a can be in a set S, we need:
    # For all b,c in S, b != c, b != a, c != a: a does NOT divide b*c
    
    # Precompute divisibility: divs[a] = set of b where a | b
    # But we need a | b*c which is more complex.
    # a | b*c iff (a / gcd(a,b)) | c
    
    # For efficiency, precompute for each a, b: need_c[a][b] = a // gcd(a, b)
    # Then a | b*c iff c is a multiple of need_c[a][b]
    
    # Precompute multiples lookup: multiples[k] = list of m in [2,256] where m % k == 0
    multiples = [[] for _ in range(N + 1)]
    for k in range(1, N + 1):
        for m in range(k, N + 1, k):
            if m >= 2:
                multiples[k].append(m)
    
    def count_conflicts(a, in_set):
        """Count number of pairs (b,c) in S, b != c, b != a, c != a, such that a | b*c"""
        count = 0
        for b in range(2, N + 1):
            if not in_set[b] or b == a:
                continue
            g = gcd(a, b)
            k = a // g
            for c in multiples[k]:
                if c == b or c == a or not in_set[c]:
                    continue
                count += 1
        return count
    
    def is_valid_add(a, in_set):
        """Check if adding a to in_set keeps the set valid"""
        for b in range(2, N + 1):
            if not in_set[b]:
                continue
            g = gcd(a, b)
            k = a // g
            for c in multiples[k]:
                if c == b and not in_set[c]:
                    continue
                if c == a:
                    continue
                if in_set[c] and c != b:
                    return False
        return True
    
    def is_valid_set(S_list):
        """Check if the set is valid"""
        in_set = [False] * (N + 1)
        for x in S_list:
            in_set[x] = True
        for a in S_list:
            for b in S_list:
                if b == a:
                    continue
                g = gcd(a, b)
                k = a // g
                for c in multiples[k]:
                    if c == b or c == a:
                        continue
                    if in_set[c]:
                        return False
        return True
    
    # Strategy 1: Greedy removal
    # Start with all elements, remove the one with most conflicts iteratively
    def greedy_removal():
        in_set = [False] * (N + 1)
        for x in candidates:
            in_set[x] = True
        S = set(candidates)
        
        while True:
            # Find element with most conflicts
            max_conf = -1
            worst = None
            for a in list(S):
                conf = count_conflicts(a, in_set)
                if conf > max_conf:
                    max_conf = conf
                    worst = a
            if worst is None or max_conf == 0:
                break
            S.remove(worst)
            in_set[worst] = False
        
        return list(S)
    
    # Strategy 2: Greedy addition with different orderings
    def greedy_addition(order):
        in_set = [False] * (N + 1)
        S = []
        for a in order:
            if is_valid_add(a, in_set):
                in_set[a] = True
                S.append(a)
        return S
    
    best_set = []
    
    # Run greedy removal
    S_rem = greedy_removal()
    if len(S_rem) > len(best_set):
        best_set = S_rem[:]
    
    # Run greedy addition with various orderings
    # Order by largest prime factor descending
    lpf = [0] * (N + 1)
    for i in range(2, N + 1):
        if lpf[i] == 0:
            for j in range(i, N + 1, i):
                if lpf[j] == 0:
                    lpf[j] = i
    
    order_lpf_desc = sorted(candidates, key=lambda x: (-lpf[x], -x))
    S = greedy_addition(order_lpf_desc)
    if len(S) > len(best_set):
        best_set = S[:]
    
    order_desc = sorted(candidates, reverse=True)
    S = greedy_addition(order_desc)
    if len(S) > len(best_set):
        best_set = S[:]
    
    # Primes first
    primes = [x for x in candidates if lpf[x] == x]
    composites = [x for x in candidates if lpf[x] != x]
    order_pf = sorted(primes, reverse=True) + sorted(composites, reverse=True)
    S = greedy_addition(order_pf)
    if len(S) > len(best_set):
        best_set = S[:]
    
    # Strategy 3: Local search on best_set
    # Try swapping: remove x from S, add y not in S, if valid and improves
    def local_search(S, max_iter=1000):
        in_set = [False] * (N + 1)
        for x in S:
            in_set[x] = True
        S = S[:]
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter and time.time() < deadline - 2.0:
            improved = False
            iter_count += 1
            for x in S[:]:
                # Try removing x
                in_set[x] = False
                S.remove(x)
                # Try adding each y not in S
                for y in candidates:
                    if in_set[y]:
                        continue
                    if is_valid_add(y, in_set):
                        # Check if this swap is beneficial (it always is since we add one)
                        # Actually we need to check if removing x and adding y is valid
                        # We already checked is_valid_add(y, in_set) where in_set doesn't have x
                        in_set[y] = True
                        S.append(y)
                        improved = True
                        break
                if not improved:
                    # Restore x
                    in_set[x] = True
                    S.append(x)
        return S
    
    if time.time() < deadline - 5.0:
        S_ls = local_search(best_set, max_iter=500)
        if len(S_ls) > len(best_set):
            best_set = S_ls[:]
    
    # Strategy 4: Random restarts with greedy addition
    import random
    random.seed(42)
    
    while time.time() < deadline - 3.0:
        order = candidates[:]
        random.shuffle(order)
        S = greedy_addition(order)
        if len(S) > len(best_set):
            best_set = S[:]
            # Try local search on improved set
            if time.time() < deadline - 5.0:
                S_ls = local_search(best_set, max_iter=200)
                if len(S_ls) > len(best_set):
                    best_set = S_ls[:]
    
    # Final verification
    result = [int(x) for x in best_set]
    return result

