import time
from math import lcm as math_lcm

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    # Precompute bad triple masks
    # bad_mask[a][b] will be a bitmask of c such that (a,b,c) is a bad triple
    # We use 0-indexed for array, but values are 1..128
    # Map value v to index v-1
    N = 128
    bad_mask = [[0] * N for _ in range(N)]
    
    for a in range(1, N + 1):
        for b in range(1, N + 1):
            if a == b:
                continue
            # Find all c such that lcm(a,b) == lcm(a,c) == lcm(b,c)
            # We can just iterate c
            mask = 0
            for c in range(1, N + 1):
                if c == a or c == b:
                    continue
                if math_lcm(a, b) == math_lcm(a, c) == math_lcm(b, c):
                    mask |= (1 << (c - 1))
            bad_mask[a - 1][b - 1] = mask
            bad_mask[b - 1][a - 1] = mask
            
    def can_add(x, S):
        # S is a set of values (1..128)
        # Check if x is compatible with all pairs in S
        S_list = list(S)
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                if (bad_mask[a - 1][b - 1] >> (x - 1)) & 1:
                    return False
        return True

    # Heuristic: Prime powers are "safe" to start with?
    # Actually, let's try a few different greedy orders and pick the best.
    
    def is_prime_power(n):
        if n == 1:
            return True
        for p in range(2, int(n**0.5) + 1):
            if n % p == 0:
                m = n
                while m % p == 0:
                    m //= p
                return m == 1
        return True

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

    best_set = set()
    best_size = 0
    
    # Strategy 1: Greedy by distinct primes desc
    # Strategy 2: Greedy by value asc
    # Strategy 3: Greedy by value desc
    # Strategy 4: Random restarts with local search
    
    orders = [
        sorted(range(1, 129), key=lambda x: (-num_distinct_primes(x), x)),
        sorted(range(1, 129)),
        sorted(range(1, 129), reverse=True),
        sorted(range(1, 129), key=lambda x: (num_distinct_primes(x), x))
    ]
    
    for order in orders:
        if time.time() - start_time > deadline - 5:
            break
        S = set()
        for x in order:
            if time.time() - start_time > deadline - 5:
                break
            if can_add(x, S):
                S.add(x)
        
        # Local search: Try to add remaining
        remaining = [x for x in range(1, 129) if x not in S]
        for x in remaining:
            if time.time() - start_time > deadline - 5:
                break
            if can_add(x, S):
                S.add(x)
        
        # Local search: 1-for-2 swaps
        improved = True
        while improved:
            improved = False
            if time.time() - start_time > deadline - 5:
                break
            S_list = list(S)
            for a in S_list:
                if time.time() - start_time > deadline - 5:
                    break
                S2 = S - {a}
                rem = [x for x in range(1, 129) if x not in S]
                for b in rem:
                    if time.time() - start_time > deadline - 5:
                        break
                    if can_add(b, S2):
                        S2.add(b)
                        for c in rem:
                            if c == b:
                                continue
                            if time.time() - start_time > deadline - 5:
                                break
                            if can_add(c, S2):
                                S = S2 | {c}
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            
        if len(S) > best_size:
            best_size = len(S)
            best_set = S

    return list(best_set)
