import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140

    max_m = 10**18

    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]

    # Use primes up to 200 to increase collision density
    # If too slow, reduce to 100
    primes = sieve(200)
    
    # Split primes into two sets
    # We want to balance the number of generated states.
    # Small primes generate more states.
    # Let's try splitting by index, but adjust based on size.
    # For now, split in half.
    mid = len(primes) // 2
    primes_1 = primes[:mid]
    primes_2 = primes[mid:]

    def generate_list(prime_list):
        n_primes = len(prime_list)
        if n_primes == 0:
            return [(1, 1)] # (k, M_part)
        
        # Precompute powers
        pp = []
        for p in prime_list:
            powers = []
            p_pow = 1
            sigma_p = 1
            while p_pow <= 10**9:
                powers.append((p_pow, sigma_p))
                p_pow *= p
                if p_pow > 10**9:
                    break
                sigma_p += p_pow
            pp.append(powers)

        stack = [(0, 1, 1)]
        results = []
        while stack:
            if time.time() > deadline:
                break
            idx, k, sigma_k = stack.pop()
            if idx == n_primes:
                M_part = k * sigma_k
                if M_part <= max_m:
                    results.append((k, M_part))
                continue
            
            p = prime_list[idx]
            powers = pp[idx]
            for p_pow, sigma_p in powers:
                new_k = k * p_pow
                if new_k > 10**9:
                    break
                new_sigma = sigma_k * sigma_p
                new_M_part = new_k * new_sigma
                if new_M_part > max_m:
                    break
                stack.append((idx + 1, new_k, new_sigma))
        
        return results

    L1 = generate_list(primes_1)
    L2 = generate_list(primes_2)
    
    # Sort L2 by M_part
    L2.sort(key=lambda x: x[1])
    M2s = [x[1] for x in L2]
    
    # We want to find M with max count
    # Count occurrences of M = M1 * M2
    counts = defaultdict(int)
    # We also need to store k's, but that's expensive.
    # Let's just count first.
    
    # Optimization: Iterate L1, binary search L2
    import bisect
    for k1, M1 in L1:
        max_M2 = max_m // M1
        # Find rightmost index in M2s <= max_M2
        idx_right = bisect.bisect_right(M2s, max_M2)
        # Iterate through M2s[0:idx_right]
        # This could be large.
        # If idx_right is large, it's slow.
        # But M1 is usually small? No.
        # Average M1 is large.
        
        # To speed up, we can just iterate if idx_right is small.
        # If idx_right is large, we might need a different approach.
        # But with balanced split, L1 and L2 are ~3000.
        # 3000 * 3000 = 9e6.
        
        for i in range(idx_right):
            M2 = M2s[i]
            M = M1 * M2
            counts[M] += 1

    best_m = 0
    best_count = 0
    for M, c in counts.items():
        if c > best_count:
            best_count = c
            best_m = M

    if best_count == 0:
        return {'M': 333312, 'k': [336, 372, 434]}

    # Re-run to find k's for best_m
    # We need to check which pairs (k1, k2) produce best_m
    # k = k1 * k2
    best_ks = set()
    for k1, M1 in L1:
        if best_m % M1 != 0:
            continue
        M2_needed = best_m // M1
        # Find k2 such that M2 == M2_needed
        # We can use a dict for L2: M2 -> list of k2
        # Let's build it once
        # Actually, we can just search in L2
        # Since L2 is small, linear search is fine for each M1?
        # No, we should map M2 to k2s.
        
        pass # Placeholder

    # Let's build M2_to_k2s
    M2_to_k2s = defaultdict(list)
    for k2, M2 in L2:
        M2_to_k2s[M2].append(k2)

    for k1, M1 in L1:
        if best_m % M1 == 0:
            M2_needed = best_m // M1
            if M2_needed in M2_to_k2s:
                for k2 in M2_to_k2s[M2_needed]:
                    k = k1 * k2
                    if k <= math.isqrt(best_m):
                        best_ks.add(k)

    if not best_ks:
        return {'M': 333312, 'k': [336, 372, 434]}

    return {'M': int(best_m), 'k': [int(k) for k in best_ks]}
