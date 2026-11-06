import math
import time

def construct():
    """
    Returns an integer n <= 10^18 maximizing the number of adjacent
    coprime pairs in its divisor sequence.
    """
    start_time = time.time()
    TIME_LIMIT = 140  # Safety margin below 150s
    
    max_val = 10**18
    
    # Precompute primes up to a reasonable limit
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]
    
    primes = sieve(200)
    
    def tau_perp(n):
        """Compute the score: number of adjacent coprime pairs in sorted divisors."""
        if n <= 1:
            return 0
        # Factorize n
        factors = []
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                factors.append((p, count))
        if temp > 1:
            factors.append((temp, 1))
        
        # Generate all divisors
        divs = [1]
        for p, exp in factors:
            current_size = len(divs)
            factor = 1
            for _ in range(exp):
                factor *= p
                for i in range(current_size):
                    divs.append(divs[i] * factor)
        
        divs.sort()
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score
    
    candidates = set()
    
    # Family 1: Square-free products of first k primes
    sf_n = 1
    for p in primes:
        if sf_n * p <= max_val:
            sf_n *= p
            candidates.add(sf_n)
        else:
            break
    
    # Family 2: Try removing one prime from the max square-free and adding exponent to a smaller prime
    sf_n = 1
    sf_primes_used = []
    for p in primes:
        if sf_n * p <= max_val:
            sf_n *= p
            sf_primes_used.append(p)
        else:
            break
    
    # Try variations: replace one prime with a higher power of a smaller prime
    for i, p_remove in enumerate(sf_primes_used):
        base = sf_n // p_remove
        for p_add in primes[:i+1]:  # try adding to primes up to the removed one
            for e in range(2, 15):
                if base * (p_add ** e) <= max_val:
                    candidates.add(base * (p_add ** e))
                else:
                    break
    
    # Family 3: Numbers of form 2^a * 3^b * 5^c * 7^d * product of remaining small primes
    # Try to maximize distinct prime count while allowing small exponents
    
    # Start with product of first k primes, then try to increase exponents
    base_primes = []
    base_n = 1
    for p in primes:
        if base_n * p <= max_val:
            base_n *= p
            base_primes.append(p)
        else:
            break
    
    # Now try to increase exponents of the smallest primes while keeping within bound
    # Use a priority queue approach: try increasing exponent of each prime
    from heapq import heappush, heappop
    
    # State: (negative_score_estimate, n, exponents_tuple)
    # We'll do a best-first search on the exponent space
    
    # Actually, let's just generate many candidates systematically
    
    # For each subset of the first ~15 primes, try exponents 1 and 2
    # This is 2^15 = 32768 combinations, manageable
    
    first_15 = primes[:15]
    
    def generate_exponent_candidates(prime_list, max_exponents, max_n):
        """Generate candidates by trying exponents for each prime in prime_list."""
        results = set()
        n = 1
        for i, p in enumerate(prime_list):
            if n > max_n:
                break
            for e in range(0, max_exponents[i] + 1):
                if n * (p ** e) <= max_n:
                    results.add(n * (p ** e))
                else:
                    break
        return results
    
    # Try: first 12 primes with exponents up to 2
    # 3^12 = 531441 combinations, too many. Let's be smarter.
    
    # Key insight: The number of coprime adjacent pairs is related to the structure
    # of the divisor lattice. Numbers with many distinct prime factors (square-free)
    # tend to have more such pairs.
    
    # Let's focus on:
    # 1. Square-free numbers with maximum number of distinct primes
    # 2. Numbers with 2 or 3 primes having exponent 2, rest square-free
    
    # Find max number of distinct primes in a square-free number <= 10^18
    max_sf = 1
    max_sf_count = 0
    for p in primes:
        if max_sf * p <= max_val:
            max_sf *= p
            max_sf_count += 1
        else:
            break
    
    # Add all square-free numbers with max_sf_count - 1 primes (remove one prime)
    sf_primes_list = []
    sf_prod = 1
    for p in primes:
        if sf_prod * p <= max_val:
            sf_prod *= p
            sf_primes_list.append(p)
        else:
            break
    
    # Remove each prime and see if we can add a higher power of a smaller prime
    for i in range(len(sf_primes_list)):
        base = sf_prod // sf_primes_list[i]
        candidates.add(base)  # square-free with one fewer prime
        # Try adding exponent to smaller primes
        for j in range(i):
            p_small = sf_primes_list[j]
            for e in range(2, 20):
                val = base * (p_small ** (e - 1))  # since p_small is already in base
                if val <= max_val:
                    candidates.add(val)
                else:
                    break
    
    # Also try: remove two primes, add exponents to two smaller primes
    for i in range(len(sf_primes_list)):
        for j in range(i):
            base = sf_prod // (sf_primes_list[i] * sf_primes_list[j])
            candidates.add(base)
            # Try various exponent combinations
            for e1 in range(2, 10):
                for e2 in range(2, 10):
                    val = base * (sf_primes_list[0] ** (e1 - 1)) * (sf_primes_list[1] ** (e2 - 1))
                    if val <= max_val:
                        candidates.add(val)
    
    # Family 4: Try numbers with exponents pattern like 2^a * 3^b * 5^c * 7^d * 11^e * ...
    # where a >= b >= c >= d >= e >= ... (non-increasing exponents for optimal packing)
    
    # DFS with non-increasing exponents
    def dfs_exponents(idx, current_n, max_exp_allowed):
        if time.time() - start_time > TIME_LIMIT:
            return
        if current_n > 1:
            candidates.add(current_n)
        
        if idx >= len(primes):
            return
        
        p = primes[idx]
        limit = max_val // current_n
        
        e_max = 0
        temp = 1
        while temp * p <= limit and e_max < max_exp_allowed:
            temp *= p
            e_max += 1
        
        # Try exponents from e_max down to 0 (non-increasing)
        for e in range(min(e_max, max_exp_allowed), -1, -1):
            next_n = current_n * (p ** e)
            dfs_exponents(idx + 1, next_n, e)
            if time.time() - start_time > TIME_LIMIT - 5:
                return
    
    # Start DFS with max exponent 30 for the first prime
    dfs_exponents(0, 1, 30)
    
    # Deduplicate and filter
    candidates = list(candidates)
    candidates = [n for n in candidates if 2 <= n <= max_val]
    
    # Sort by heuristic: prefer more distinct prime factors, then smaller n
    def num_distinct_primes(n):
        temp = n
        count = 0
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                count += 1
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            count += 1
        return count
    
    # Sort: primary key = num distinct primes (desc), secondary = n (asc)
    candidates.sort(key=lambda x: (-num_distinct_primes(x), x))
    
    # Evaluate candidates, starting with most promising
    best_n = candidates[0] if candidates else 2
    best_score = -1
    
    # Limit evaluation to stay within time
    for n in candidates:
        if time.time() - start_time > TIME_LIMIT - 3:
            break
        score = tau_perp(n)
        if score > best_score:
            best_score = score
            best_n = n
    
    return best_n
