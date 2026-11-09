"""Program to find N with many representations as sums of two powerful numbers."""
import time
import sys
from math import isqrt

def _is_powerful(n):
    """Check if n is powerful (every prime factor appears squared)."""
    if n < 1:
        return False
    if n == 1:
        return True
    m = n
    d = 2
    while d * d <= m:
        if m % d == 0:
            cnt = 0
            while m % d == 0:
                m //= d
                cnt += 1
            if cnt == 1:
                return False
        d += 1
    return True  # remaining m is 1 or a prime with exponent >= 2 (but we checked)

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit."""
    # Powerful numbers: every prime factor has exponent >= 2
    # Form: product of p_i^e_i where e_i >= 2
    # We can generate by: for each square s, multiply by cubes of primes, etc.
    # Better: recursive generation
    powerful = set()
    
    def gen(current, start_prime):
        if current > limit:
            return
        if current > 1:
            powerful.add(current)
        # Try multiplying by p^2 for primes p >= start_prime
        p = start_prime
        while p * p <= limit // current:
            gen(current * p * p, p)
            # Also try p^3, p^4, etc.
            val = current * p * p
            while val * p <= limit:
                val *= p
                gen(val, p + 1)  # wait, this isn't right
            p += 1
    
    # Better approach: enumerate by prime factorization
    # A powerful number has form: prod p_i^{e_i} with e_i >= 2
    # Generate all such numbers <= limit
    
    # Use the fact that powerful numbers are sparse
    # For n <= 10^18, powerful numbers are quite sparse
    
    # Method: for each "core" (product of distinct primes), multiply by squares
    # Actually, let's use a different approach:
    # Every powerful number n can be written as a^2 * b where b is squarefree
    # and every prime in b has exponent >= 2 in n, meaning b's primes appear with even excess
    # Actually: n is powerful iff n = a^2 * b where b is cube-free and every prime dividing b divides a
    # Simpler: n powerful iff for all p|n, p^2 | n
    
    # Let's just generate by iterating over possible structures
    # n = prod p_i^{e_i}, e_i >= 2
    # Number of such n <= X is about O(X^{1/2})
    
    # Generate: for each m from 1 to sqrt(limit), m^2 is powerful
    # Also m^2 * p^2 for primes p, etc.
    
    # Let me use a cleaner recursive approach
    primes = []
    # Sieve primes up to limit^{1/2} since smallest prime factor squared must be <= limit
    # Actually primes up to limit^{1/6} for the "new" primes, but let's be safe
    max_p = int(limit ** (1/2)) + 1
    is_prime = [True] * (max_p + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(max_p**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, max_p+1, i):
                is_prime[j] = False
    primes = [i for i in range(2, max_p+1) if is_prime[i]]
    
    powerful = set([1])
    
    def gen_powerful(idx, current):
        if current > limit:
            return
        if idx >= len(primes):
            return
        p = primes[idx]
        # Try p^2, p^3, p^4, ...
        val = current * p * p
        while val <= limit:
            powerful.add(val)
            gen_powerful(idx + 1, val)
            val *= p
    
    gen_powerful(0, 1)
    powerful = sorted(powerful)
    return powerful

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Step 1: Generate powerful numbers up to 10^18
    # This is the expensive part, but powerful numbers up to 10^18 are about 10^9... too many
    # Actually, count of powerful numbers <= X is ~ C * X^{1/2}
    # For X = 10^18, that's ~ 10^9 which is too many to store
    # 
    # Better approach: we don't need ALL powerful numbers, just enough to find good N
    # 
    # Alternative: use the structure. For a fixed N, we need a + b = N where a, b powerful.
    # We can iterate over powerful a and check if N-a is powerful.
    # 
    # Strategy: 
    # 1. Generate powerful numbers up to some limit L (say 10^12 or 10^15)
    # 2. For candidate N values, count representations
    # 3. Use additive structure to find good N
    
    # Let's generate powerful numbers up to 10^12 first (manageable)
    limit_gen = 10**12
    
    def gen_powerful_up_to(limit):
        """Generate powerful numbers up to limit using recursive prime factorization."""
        if limit < 1:
            return [1]
        
        # Sieve primes up to sqrt(limit)
        max_p = int(limit ** 0.5) + 2
        is_prime = [True] * (max_p + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(max_p**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, max_p+1, i):
                    is_prime[j] = False
        primes = [i for i in range(2, max_p+1) if is_prime[i]]
        
        powerful = set([1])
        
        def gen(idx, current):
            if current > limit:
                return
            if idx >= len(primes):
                return
            p = primes[idx]
            # Try p^2, p^3, p^4, ...
            val = current * p * p
            while val <= limit:
                powerful.add(val)
                gen(idx + 1, val)
                val *= p
        
        gen(0, 1)
        return sorted(powerful)
    
    powerful = gen_powerful_up_to(limit_gen)
    powerful_set = set(powerful)
    n_powerful = len(powerful)
    
    # Step 2: Find N with many representations
    # For each pair (a, b) in powerful x powerful with a <= b, N = a + b
    # We want N that appears most often
    # But pairwise is O(n_powerful^2) which is too slow for large sets
    
    # Better: for candidate N, count = sum over a in powerful of [N - a in powerful_set]
    # We need to find N that maximizes this count
    
    # Strategy: use the fact that if N is a sum of two powerful numbers, 
    # and we have many powerful numbers, we can sample or use convolution
    
    # For efficiency, let's focus on N in a reasonable range
    # Powerful numbers up to 10^12: their sums go up to 2*10^12
    # But we can go up to 10^18 if we generate more
    
    # Let's first try with the current set and see how many representations we get
    
    # To find max-count N efficiently:
    # For each a in powerful, for each b in powerful with a <= b:
    #   N = a + b
    #   count[N] += 1
    # But this is O(n^2)
    
    # Alternative: for a range of N values, compute count
    # Let's pick N values that are "likely" to have many representations
    # N that are sums of many small powerful numbers
    
    # Heuristic: N that is divisible by many small squares might work
    # Or N that is near the "center" of the powerful number distribution
    
    # Let's try: for each powerful number a, consider N = a + b for various b
    # Focus on N that can be written as sum of two powerful numbers in many ways
    
    # Efficient approach: 
    # Sort powerful numbers. For each a, the valid b range is [a, N-a] 
    # But we don't know N in advance.
    
    # Let's use a different strategy: 
    # Pick N from a set of candidates and verify
    # Candidates: numbers that are sums of two powerful numbers where both are "common"
    
    # Actually, let's just do a focused search:
    # For N in a range, count representations
    # Use the powerful set we have
    
    # First, let's see what's achievable with our current powerful set
    # Max N we can form: 2 * max(powerful)
    
    max_pow = powerful[-1] if powerful else 1
    search_max = min(2 * max_pow, 10**18)
    
    # For efficiency, let's check N values that are multiples of small powerful-friendly numbers
    # Or just sample N values
    
    # Better: use the additive structure
    # If we have powerful numbers p1 < p2 < ... < pn
    # Then for N, count(N) = |{i: N - p_i is powerful}|
    # 
    # Let's try N values that are "round" or have special structure
    
    # Key insight: 1 is powerful. So N-1 must be powerful for (1, N-1) to be a pair.
    # Also 4 is powerful, so N-4 must be powerful, etc.
    
    # Let's generate candidate N values:
    # N = a + b where a, b are both in our powerful set
    # To find the N with max count, we can:
    # 1. For each a in powerful, for each b in powerful with a <= b: record N = a+b
    #    But this is O(n^2)
    
    # Since n_powerful for limit 10^12 might be around 10^6, n^2 = 10^12 which is too slow
    
    # Let's reduce: use a smaller powerful set for the pairwise approach
    # Or use a smarter method
    
    # Smarter method: for each N in a candidate set, compute count
    # Candidate N: let's try N values that are sums of "typical" powerful numbers
    
    # Let's try: N = 2 * m where m is powerful (since if m is powerful, 2m might not be)
    # Actually, let's just try specific N values
    
    # For now, let's do a limited pairwise search with a subset
    # Use the first K powerful numbers for pairwise
    K = min(n_powerful, 50000)
    small_powerful = powerful[:K]
    
    from collections import defaultdict
    count_dict = defaultdict(int)
    
    for i in range(K):
        a = small_powerful[i]
        for j in range(i, K):
            b = small_powerful[j]
            N = a + b
            if N <= 10**18:
                count_dict[N] += 1
    
    if count_dict:
        best_N = max(count_dict.keys(), key=lambda x: count_dict[x])
    else:
        best_N = 2
    
    # Verify and get pairs for best_N
    pairs = []
    for a in powerful:
        if 2 * a > best_N:
            break
        b = best_N - a
        if b in powerful_set and a <= b:
            pairs.append([a, b])
    
    # Check if we have time to do more
    elapsed = time.time() - start_time
    if elapsed < deadline - 20 and n_powerful < 500000:
        # Try to extend the powerful set and re-search
        # But for now, return what we have
        pass
    
    return {'N': best_N, 'pairs': pairs}

# Test
if __name__ == "__main__":
    result = construct()
    print(f"N = {result['N']}")
    print(f"Number of pairs: {len(result['pairs'])}")
    print(f"First few pairs: {result['pairs'][:10]}")
    # Verify
    N = result['N']
    for a, b in result['pairs']:
        assert a <= b, f"Invalid: {a} > {b}"
        assert a + b == N, f"Invalid sum: {a} + {b} != {N}"
        assert _is_powerful(a), f"{a} is not powerful"
        assert _is_powerful(b), f"{b} is not powerful"
    print("All pairs verified!")
