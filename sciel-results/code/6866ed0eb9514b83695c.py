import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s buffer for safety
    
    # Precompute primes up to 20000
    limit = 20000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    # Fast factorization using sieve
    # We just need the set of prime factors
    # Precompute smallest prime factor (SPF) for speed
    spf = list(range(limit + 1))
    for i in range(2, int(limit**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def get_prime_factors(n):
        factors = set()
        while n > 1:
            factors.add(spf[n])
            n //= spf[n]
        return factors

    def evaluate(a):
        primes_union = set()
        n = len(a)
        for i in range(n):
            for j in range(i + 1, n):
                s = a[i] + a[j]
                primes_union.update(get_prime_factors(s))
        return len(primes_union)

    # Strategy: Use modulus M to constrain sums
    # Try M = 210 (2*3*5*7)
    # a_i = 105 + 210 * k_i
    # Max k: (10000 - 105) // 210 = 47
    # We search for k_i in {0..47}
    
    best_score = float('inf')
    best_a = []
    
    # Initial candidate
    k_vals = list(range(12))
    a_curr = [105 + 210 * k for k in k_vals]
    curr_score = evaluate(a_curr)
    best_score = curr_score
    best_a = a_curr[:]
    
    # Local Search
    # We represent the state as a set of k's
    # k_set = set(k_vals)
    
    def get_a_from_k(k_set):
        return [105 + 210 * k for k in sorted(k_set)]
    
    k_set = set(k_vals)
    a_curr = get_a_from_k(k_set)
    curr_score = evaluate(a_curr)
    
    # Hill Climbing
    # We can try to swap one k with another
    # Or just iterate through all possible k replacements
    
    # To speed up, we can cache scores of k sets? No, too many.
    # Just evaluate on the fly.
    
    # We will try to improve by replacing one k_i with a new k
    # There are 48 possible k values.
    
    # We can also try random restarts with different initial k sets
    # But the structure M=210 is strong.
    
    # Let's run a simple hill climbing
    improved = True
    while improved and time.time() < deadline:
        improved = False
        k_list = sorted(k_set)
        for idx, k_old in enumerate(k_list):
            # Try replacing k_old with k_new
            for k_new in range(48):
                if k_new == k_old or k_new in k_set:
                    continue
                
                # Swap
                k_set.remove(k_old)
                k_set.add(k_new)
                
                a_temp = get_a_from_k(k_set)
                score = evaluate(a_temp)
                
                if score < curr_score:
                    curr_score = score
                    if score < best_score:
                        best_score = score
                        best_a = a_temp[:]
                    improved = True
                else:
                    # Revert
                    k_set.remove(k_new)
                    k_set.add(k_old)
                    
                if time.time() > deadline - 5:
                    # Restore state before exiting
                    if k_new in k_set and k_old not in k_set:
                        k_set.remove(k_new)
                        k_set.add(k_old)
                    return best_a
    
    # If we have time, try M=30
    # a_i = 15 + 30 * k_i
    # Max k: (10000 - 15) // 30 = 332
    # This has more search space, but base primes are only 3.
    # If we can get co-factors to add 0 or 1 primes, total 3 or 4.
    # Let's try a quick search here too.
    
    if time.time() < deadline - 10:
        # Initialize with small k
        k_vals_30 = list(range(12))
        k_set_30 = set(k_vals_30)
        a_curr_30 = [15 + 30 * k for k in sorted(k_set_30)]
        curr_score_30 = evaluate(a_curr_30)
        
        if curr_score_30 < best_score:
            best_score = curr_score_30
            best_a = a_curr_30[:]
            
        improved_30 = True
        while improved_30 and time.time() < deadline - 5:
            improved_30 = False
            k_list_30 = sorted(k_set_30)
            for idx, k_old in enumerate(k_list_30):
                # Try a few random new k's to avoid full 332 loop if slow
                # But 332 is small. Let's try all.
                for k_new in range(333):
                    if k_new == k_old or k_new in k_set_30:
                        continue
                    
                    k_set_30.remove(k_old)
                    k_set_30.add(k_new)
                    
                    a_temp = [15 + 30 * k for k in sorted(k_set_30)]
                    score = evaluate(a_temp)
                    
                    if score < curr_score_30:
                        curr_score_30 = score
                        if score < best_score:
                            best_score = score
                            best_a = a_temp[:]
                        improved_30 = True
                    else:
                        k_set_30.remove(k_new)
                        k_set_30.add(k_old)
                    
                    if time.time() > deadline - 5:
                        if k_new in k_set_30 and k_old not in k_set_30:
                            k_set_30.remove(k_new)
                            k_set_30.add(k_old)
                        return best_a

    return best_a

# EVOLVE-BLOCK-START
def construct():
    return _construct_impl()

def _construct_impl():
    import time
    start_time = time.time()
    deadline = start_time + 80.0
    
    limit = 20000
    spf = list(range(limit + 1))
    for i in range(2, int(limit**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Precompute prime factor bitmasks for all sums up to 20000
    # First, collect all primes up to 20000
    primes = [i for i in range(2, limit + 1) if spf[i] == i]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # For bitmask, we need up to num_primes bits. Use Python big ints.
    # Precompute factor_mask for each n up to limit
    factor_mask = [0] * (limit + 1)
    for n in range(2, limit + 1):
        mask = 0
        x = n
        while x > 1:
            p = spf[x]
            mask |= (1 << prime_to_idx[p])
            while x % p == 0:
                x //= p
        factor_mask[n] = mask
    
    def evaluate_mask(a):
        """Evaluate using bitmask union, return (mask, count)"""
        union_mask = 0
        n = len(a)
        for i in range(n):
            ai = a[i]
            for j in range(i + 1, n):
                s = ai + a[j]
                if s <= limit:
                    union_mask |= factor_mask[s]
        return union_mask
    
    def count_bits(mask):
        return bin(mask).count('1')
    
    best_score = float('inf')
    best_a = []
    
    def try_search(k_set, base, mod, max_k, current_mask, current_count):
        nonlocal best_score, best_a
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            k_list = sorted(k_set)
            for idx, k_old in enumerate(k_list):
                for k_new in range(max_k + 1):
                    if k_new == k_old or k_new in k_set:
                        continue
                    
                    k_set.remove(k_old)
                    k_set.add(k_new)
                    
                    a_temp = [base + mod * k for k in sorted(k_set)]
                    new_mask = evaluate_mask(a_temp)
                    new_count = count_bits(new_mask)
                    
                    if new_count < current_count:
                        current_mask = new_mask
                        current_count = new_count
                        if new_count < best_score:
                            best_score = new_count
                            best_a = a_temp[:]
                        improved = True
                    else:
                        k_set.remove(k_new)
                        k_set.add(k_old)
                        
                    if time.time() > deadline - 5:
                        if k_new in k_set and k_old not in k_set:
                            k_set.remove(k_new)
                            k_set.add(k_old)
                        return current_count
        
        return current_count
    
    # Try multiple modular structures
    # Structure 1: M=210, a_i = 105 + 210*k_i, k in 0..47
    k_set_210 = set(range(12))
    a_210 = [105 + 210 * k for k in sorted(k_set_210)]
    mask_210 = evaluate_mask(a_210)
    count_210 = count_bits(mask_210)
    best_score = count_210
    best_a = a_210[:]
    count_210 = try_search(k_set_210, 105, 210, 47, mask_210, count_210)
    
    # Structure 2: M=30, a_i = 15 + 30*k_i, k in 0..332
    if time.time() < deadline - 15:
        k_set_30 = set(range(12))
        a_30 = [15 + 30 * k for k in sorted(k_set_30)]
        mask_30 = evaluate_mask(a_30)
        count_30 = count_bits(mask_30)
        if count_30 < best_score:
            best_score = count_30
            best_a = a_30[:]
        count_30 = try_search(k_set_30, 15, 30, 332, mask_30, count_30)
    
    # Structure 3: M=6, a_i = 3 + 6*k_i, k in 0..1666
    if time.time() < deadline - 15:
        k_set_6 = set(range(12))
        a_6 = [3 + 6 * k for k in sorted(k_set_6)]
        mask_6 = evaluate_mask(a_6)
        count_6 = count_bits(mask_6)
        if count_6 < best_score:
            best_score = count_6
            best_a = a_6[:]
        count_6 = try_search(k_set_6, 3, 6, 1666, mask_6, count_6)
    
    # Structure 4: M=10, a_i = 5 + 10*k_i, k in 0..999
    if time.time() < deadline - 15:
        k_set_10 = set(range(12))
        a_10 = [5 + 10 * k for k in sorted(k_set_10)]
        mask_10 = evaluate_mask(a_10)
        count_10 = count_bits(mask_10)
        if count_10 < best_score:
            best_score = count_10
            best_a = a_10[:]
        count_10 = try_search(k_set_10, 5, 10, 999, mask_10, count_10)
    
    # Structure 5: M=14, a_i = 7 + 14*k_i, k in 0..713
    if time.time() < deadline - 15:
        k_set_14 = set(range(12))
        a_14 = [7 + 14 * k for k in sorted(k_set_14)]
        mask_14 = evaluate_mask(a_14)
        count_14 = count_bits(mask_14)
        if count_14 < best_score:
            best_score = count_14
            best_a = a_14[:]
        count_14 = try_search(k_set_14, 7, 14, 713, mask_14, count_14)
    
    # Structure 6: M=15, a_i = 15 + 15*k_i... wait, need gcd issues
    # Actually a_i = 15 + 15*k means a_i is multiple of 15, sum is multiple of 15
    # But 15+15=30, 15+30=45... all divisible by 15. Base primes {3,5}
    if time.time() < deadline - 15:
        k_set_15 = set(range(1, 13))  # k=1..12 to avoid a_i=0
        a_15 = [15 + 15 * k for k in sorted(k_set_15)]
        mask_15 = evaluate_mask(a_15)
        count_15 = count_bits(mask_15)
        if count_15 < best_score:
            best_score = count_15
            best_a = a_15[:]
        count_15 = try_search(k_set_15, 15, 15, 666, mask_15, count_15)
    
    # Structure 7: M=21, a_i = 10 + 21*k_i (10 is coprime to 21? No, gcd(10,21)=1)
    # sum = 20 + 21(k_i+k_j) = not necessarily divisible by 3 or 7
    # Better: a_i = 21 + 21*k_i, sum = 21(k_i+k_j+2), base primes {3,7}
    if time.time() < deadline - 15:
        k_set_21 = set(range(1, 13))
        a_21 = [21 + 21 * k for k in sorted(k_set_21)]
        mask_21 = evaluate_mask(a_21)
        count_21 = count_bits(mask_21)
        if count_21 < best_score:
            best_score = count_21
            best_a = a_21[:]
        count_21 = try_search(k_set_21, 21, 21, 475, mask_21, count_21)
    
    return best_a
# EVOLVE-BLOCK-END
