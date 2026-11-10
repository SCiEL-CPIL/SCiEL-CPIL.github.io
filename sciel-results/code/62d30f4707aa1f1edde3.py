import time
from math import gcd, isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin

    MAX_N = 256

    # 1. Generate primes and smallest prime factor (SPF)
    def get_spf(n):
        spf = list(range(n + 1))
        for i in range(2, isqrt(n) + 1):
            if spf[i] == i:  # i is prime
                for j in range(i * i, n + 1, i):
                    if spf[j] == j:
                        spf[j] = i
        return spf

    spf = get_spf(MAX_N)
    primes = [i for i in range(2, MAX_N + 1) if spf[i] == i]
    num_primes = len(primes)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # Map prime to index for bitmask operations
    prime_to_bit = {p: (1 << i) for i, p in enumerate(primes)}
    
    # Precompute prime factor bitmask for each number
    factors_mask = [0] * (MAX_N + 1)
    for n in range(2, MAX_N + 1):
        mask = 0
        x = n
        while x > 1:
            p = spf[x]
            mask |= prime_to_bit[p]
            while x % p == 0:
                x //= p
        factors_mask[n] = mask

    # Group numbers by their Smallest Prime Factor
    candidates = [[] for _ in range(num_primes)]
    for n in range(2, MAX_N + 1):
        p = spf[n]
        idx = prime_to_idx[p]
        candidates[idx].append(n)
    
    # Sort candidates in descending order to find high sums early
    for i in range(num_primes):
        candidates[i].sort(reverse=True)

    # Precompute suffix upper bounds: for each prime index, the max possible value
    # that can be added from that prime onwards, assuming no conflicts.
    # This is a valid upper bound because it overestimates by ignoring conflicts.
    suffix_max = [0] * (num_primes + 1)
    for i in range(num_primes - 1, -1, -1):
        max_val = 0
        if candidates[i]:
            max_val = candidates[i][0]  # Largest candidate for this prime
        suffix_max[i] = suffix_max[i + 1] + max_val

    best_sum = 1  # Always include 1
    best_set = {1}
    time_exceeded = [False]

    def branch_and_bound(idx, used_mask, current_sum, current_set):
        nonlocal best_sum, best_set, time_exceeded
        
        if time.time() > deadline:
            time_exceeded[0] = True
            return

        if idx == num_primes:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = set(current_set)
            return

        # Quick upper bound using precomputed suffix
        ub = current_sum + suffix_max[idx]
        if ub <= best_sum:
            return

        # Tighter bound: for each remaining prime, find max compatible candidate
        # Only do this if the quick bound is close to best_sum
        if ub > best_sum:
            tight_ub = current_sum
            for j in range(idx, num_primes):
                max_val = 0
                for n in candidates[j]:
                    if (factors_mask[n] & used_mask) == 0:
                        max_val = n
                        break
                tight_ub += max_val
                if tight_ub <= best_sum:
                    return
            
            if tight_ub <= best_sum:
                return

        p_idx = idx
        
        # Option 1: Skip this prime (no number assigned to it as spf)
        branch_and_bound(idx + 1, used_mask, current_sum, current_set)
        if time_exceeded[0]:
            return
        
        # Option 2: Pick a number n from candidates[p_idx]
        for n in candidates[p_idx]:
            if (factors_mask[n] & used_mask) == 0:
                new_mask = used_mask | factors_mask[n]
                new_sum = current_sum + n
                
                # Quick prune
                if new_sum + suffix_max[idx + 1] <= best_sum:
                    continue
                
                branch_and_bound(idx + 1, new_mask, new_sum, current_set + [n])
                if time_exceeded[0]:
                    return

    # Start B&B
    branch_and_bound(0, 0, 1, [1])

    # Local Search to polish the result
    improved = True
    while improved and time.time() < deadline:
        improved = False
        S_list = list(best_set)
        
        # 1-for-1 swaps: try to replace a with a larger b
        for a in S_list:
            if a == 1:
                continue
            for b in range(256, a, -1):
                if b in best_set:
                    continue
                if all(gcd(b, x) == 1 for x in best_set if x != a):
                    if b > a:
                        best_set = (best_set - {a}) | {b}
                        best_sum = best_sum - a + b
                        improved = True
                        break
            if improved:
                break
        
        # Try to add numbers
        if time.time() < deadline:
            for b in range(256, 1, -1):
                if b in best_set:
                    continue
                if all(gcd(b, x) == 1 for x in best_set):
                    best_set.add(b)
                    best_sum += b
                    improved = True
                    break

    return [int(x) for x in sorted(best_set)]
