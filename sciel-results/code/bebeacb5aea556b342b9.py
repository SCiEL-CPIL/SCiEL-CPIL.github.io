import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    def factorize(n):
        factors = set()
        temp = n
        d = 2
        while d * d <= temp:
            if temp % d == 0:
                factors.add(d)
                while temp % d == 0:
                    temp //= d
            d += 1
        if temp > 1:
            factors.add(temp)
        return factors

    def count_distinct_prime_factors(arr):
        all_factors = set()
        n = len(arr)
        for i in range(n):
            for j in range(i+1, n):
                s = arr[i] + arr[j]
                all_factors |= factorize(s)
        return len(all_factors)

    best_score = float('inf')
    best_result = list(range(1, 13))

    # Strategy: Construct sets of form M * 2^i
    # 1. Identify small primes for M
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19]
    
    # 2. Generate candidate M's (smooth numbers)
    # M must be such that we can fit 12 distinct powers of 2 multiplied by M <= 10000
    # Let exponents be e_1 < e_2 < ... < e_12
    # We need M * 2^{e_12} <= 10000
    # To maximize flexibility, we look for M that allows e_12 to be at least 11 (if starting at 0)
    # Or generally, we just iterate M and find valid exponent sets.
    
    # Pre-compute prime factors of 2^a + 2^b for a,b up to 14 (since 2^14 > 10000/1)
    max_exp = 14
    sum_pf = {}
    for a in range(max_exp + 1):
        for b in range(a + 1, max_exp + 1):
            s = (1 << a) + (1 << b)
            sum_pf[(a, b)] = factorize(s)
            
    # Also pre-compute PF of M for smooth M's
    # Let's generate smooth numbers <= 10000 using primes {2,3,5,7,11,13}
    smooth_ms = set()
    def gen_smooth(cur, idx, limit):
        if cur > limit:
            return
        if idx == len(small_primes):
            smooth_ms.add(cur)
            return
        p = small_primes[idx]
        gen_smooth(cur, idx + 1, limit)
        val = cur
        while val <= limit:
            gen_smooth(val, idx + 1, limit)
            if val > limit // p:
                break
            val *= p
            
    gen_smooth(1, 0, 10000)
    
    # Filter M's that allow at least 12 distinct powers of 2 * M <= 10000
    # This requires M * 2^{e_min+11} <= 10000 if we use consecutive exponents starting at e_min
    # Or more generally, we can pick any 12 exponents from [0, E_max] where M*2^{E_max} <= 10000
    # We want to minimize the union of PF(M) and PF(2^a+2^b)
    
    # To speed up, we can just try M in [1, 100] first, then broader smooth set
    # Actually, let's just iterate over all smooth M's. There aren't too many.
    
    # Sort smooth M's to try smaller ones first? Or all?
    # Let's evaluate all valid M's
    
    for M in smooth_ms:
        if M > 10000:
            continue
        
        # Find max exponent E such that M * 2^E <= 10000
        E_max = 0
        val = M
        while val * 2 <= 10000:
            val *= 2
            E_max += 1
            
        if E_max < 11: # Need at least 12 exponents (0..11)
            continue
            
        # We need to choose 12 exponents from [0, E_max]
        # To minimize prime factors, we should prefer exponents that yield sums with few/small prime factors
        # Let's try consecutive blocks of 12 exponents first
        # Start 0..11, 1..12, etc.
        
        pf_M = factorize(M)
        
        # Try all windows of 12 exponents
        for start_exp in range(0, E_max - 11 + 1):
            exponents = list(range(start_exp, start_exp + 12))
            
            # Calculate union of PFs
            union_pf = set(pf_M)
            valid = True
            for i in range(12):
                for j in range(i+1, 12):
                    a, b = exponents[i], exponents[j]
                    if (a, b) in sum_pf:
                        union_pf |= sum_pf[(a, b)]
                    else:
                        # Fallback factorization (should not happen if max_exp is high enough)
                        s = (1 << a) + (1 << b)
                        union_pf |= factorize(s)
                        
            score = len(union_pf)
            
            if score < best_score:
                best_score = score
                best_result = [M * (1 << e) for e in exponents]
                
                # If score is very low (e.g. 3 or 4), we might stop early or continue to find even lower
                # Theoretical minimum is likely small. 
                # PF(M) for M=1 is empty. Sums of powers of 2: 1+2=3, 1+4=5, 2+4=6(2,3), 1+8=9(3), etc.
                # With M=1, exponents 0..11:
                # Sums include 1+2=3, 1+4=5, 2+4=6, 1+8=9, 2+8=10, 4+8=12, 1+16=17...
                # 17 is a new prime. 1+32=33(3,11). 11 is new.
                # So M=1 likely has many primes.
                # M=3? PF(M)={3}. Sums are 3*(2^a+2^b).
                # If 2^a+2^b only has factors 2 and 3, then total factors {2,3}.
                # 2^a+2^b = 2^min(1+2^diff).
                # 1+2^k is odd.
                # k=1: 3 (ok)
                # k=2: 5 (bad, new prime)
                # k=3: 9=3^2 (ok)
                # k=4: 17 (bad)
                # So we need differences to be 1 or 3?
                # If we pick exponents such that all differences are 1 or 3...
                # 0, 1 (diff 1), 4 (diff 4 from 0, bad).
                # It's hard to keep all diffs in {1,3} for 12 elements.
                # However, M can absorb some primes.
                
    # Local search to improve best_result
    if best_score > 3:
        current = best_result[:]
        random.seed(42)
        
        # Try to swap elements to reduce prime count
        # This is expensive, so we limit iterations
        max_iters = 500
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            improved = False
            # Pick a random index to replace
            idx = random.randint(0, 11)
            old_val = current[idx]
            
            # Try to find a replacement that reduces score
            # Heuristic: try values close to old_val or random smooth values
            # Since we don't know the structure, random search in a local neighborhood
            found_better = False
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                new_val = random.randint(1, 10000)
                if new_val in current and new_val != old_val:
                    continue
                new_current = current[:]
                new_current[idx] = new_val
                if len(set(new_current)) != 12:
                    continue
                score = count_distinct_prime_factors(new_current)
                if score < best_score:
                    best_score = score
                    best_result = new_current[:]
                    current = new_current[:]
                    improved = True
                    found_better = True
                    break
            if found_better:
                continue
            # If no improvement, we could try a different strategy, but for now, break or continue
            # To avoid getting stuck, we break if we don't improve for a while, but here we just loop
            # Actually, random replacement is very unlikely to improve a structured solution.
            # Let's stop local search if we haven't improved in 50 iterations to save time for other strategies if any.
            # But since we are in a specific block, let's just break to be safe if no improvement.
            # The structured search is the main driver.
            break
            
    return best_result
