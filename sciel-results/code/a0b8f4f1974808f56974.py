"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit."""
    powerful = set()
    powerful.add(1)
    # A powerful number has the form prod(p_i^(e_i)) where e_i >= 2
    # We can generate by iterating over possible "square-free kernel" structures
    # Alternative: for each n up to limit, check if powerful (slow for large limits)
    # Better: generate via recursive construction
    # Powerful numbers: n = a^2 * b where b is squarefree and every prime in b has exponent >= 2 in n
    # Actually simpler: n is powerful iff for every prime p|n, p^2|n
    # Generation: n = prod(p_i^(2*e_i)) where e_i >= 1
    # So n = (prod p_i^e_i)^2 * prod(p_i^(e_i % 2)) ... no
    # Let's use: n = a^2 * b where b is squarefree and for each prime q|b, q^2|n
    # This means n = a^2 * b where b is squarefree AND b | a^2... that's circular
    # Direct approach: n is powerful iff n = prod(p_i^(2k_i)) for k_i >= 1
    # Generate by trying all combinations
    # For limit up to 10^8, number of powerful numbers is ~ O(sqrt(limit)) ≈ 10^4
    # Let's use a sieve-like approach or direct enumeration
    
    # Direct enumeration: for each possible "core" c (squarefree), multiply by squares
    # Actually, the simplest correct approach for limit <= 10^8:
    # A number n <= limit is powerful iff for all primes p|n, p^2|n
    # We can generate all powerful numbers by:
    # n = prod(p_i^(2*e_i)) where e_i >= 1
    # So n = m^2 * s where s is a product of distinct primes each appearing to an even power >= 2
    # Wait, that's not right either.
    # Let me just use: powerful numbers are n such that rad(n)^2 | n
    # Generate by: for each squarefree d, n = d * k^2 where d is squarefree and for each p|d, p|k
    # So n = d * (d * m)^2 / d ... 
    # Simplest: n = prod(p_i^(2*e_i)) = (prod p_i^e_i)^2 * prod(p_i^(e_i mod 2))
    # If all exponents even: n = (prod p_i^(e_i/2))^2, a perfect square
    # If some odd: not a perfect square but still powerful
    # Example: 12 = 2^2 * 3^1 -> NOT powerful (3^1, need 3^2)
    # 72 = 2^3 * 3^2 = 2^2*2 * 3^2 -> 2^2|72 yes, 3^2|72 yes -> powerful
    # So exponents must be >= 2, not necessarily even
    # n = prod(p_i^(e_i)) where e_i >= 2
    # Generate: iterate over possible prime factorizations
    # For limit 10^8, max prime factor is 10^4 (since 10^4^2 = 10^8)
    # But primes can be larger if exponent is 2: p^2 <= 10^8 -> p <= 10^4
    # So all prime factors are <= 10^4
    
    # Let's use a different approach: 
    # For each a from 1 to sqrt(limit), for each b from 1 to limit/a^2:
    #   n = a^2 * b is powerful iff for every prime p|b, p|a
    # This is still complex. Let me just precompute using the known fact that
    # powerful numbers up to N are O(N^(1/2)) in count.
    
    # Practical approach for limit up to ~10^8:
    # Generate all numbers of the form a^2 * b where b is squarefree and gcd(b, a) has all primes of b
    # i.e., b | a^2 (which is automatic if b is squarefree and every prime in b divides a)
    # So: for each squarefree b, for each a multiple of rad(b) (i.e., a = rad(b)*k), n = a^2 * b <= limit
    # But this overcounts. Let me just use a direct check for a reasonable limit.
    
    # For the search, we need powerful numbers up to about N_max/2 where N_max is our search limit
    # Let's set search limit to 10^7 for N, so we need powerful numbers up to 5*10^6
    # Actually, let's be smarter. Let's generate powerful numbers up to 10^7.
    
    # Efficient generation:
    # A powerful number n has the form n = prod(p_i^(e_i)) with e_i >= 2
    # We can generate by considering n = a^2 * b where b is squarefree and for each p|b, p|a
    # Equivalently: n = (a * sqrt(b))^2 ... no
    # Let's just do: for each m from 1 to sqrt(limit), for each squarefree b with b | m^2 and b <= limit/m^2:
    #   n = m^2 * b is powerful
    # But "b | m^2 and b squarefree" means b divides the squarefree part of m^2, which is rad(m)^2 / ... 
    # Actually if b is squarefree and b | m^2, then b | m (since b squarefree)
    # So: for each m from 1 to sqrt(limit), for each squarefree b dividing m with b <= limit/m^2:
    #   n = m^2 * b is powerful
    # And every powerful number is uniquely represented this way? 
    # n = prod(p_i^(e_i)), let m = prod(p_i^(floor(e_i/2))), b = prod(p_i^(e_i mod 2))
    # Then b is squarefree, and if e_i mod 2 = 1, then p_i | m only if e_i >= 2, which is true.
    # Wait, if e_i = 2, floor(e_i/2) = 1, so p_i^1 | m, and e_i mod 2 = 0, so p_i not in b. Good.
    # If e_i = 3, floor(e_i/2) = 1, p_i | m, e_i mod 2 = 1, p_i | b. And b | m? p_i | m yes. Good.
    # So b | m always holds. And n = m^2 * b.
    # But is this unique? m = prod p_i^floor(e_i/2), b = prod p_i^(e_i mod 2). Yes unique.
    
    powerful_set = set()
    sqrt_limit = isqrt(limit)
    # Precompute squarefree divisors for each m? That's expensive.
    # Alternative: for each m, find all squarefree b | m such that m^2 * b <= limit
    # b | m and b squarefree means b is a squarefree divisor of m
    # For m up to 10^4, we can factor m and enumerate squarefree divisors
    
    # Factor all numbers up to sqrt_limit
    # Actually, let's just iterate m and for each, try b as squarefree divisors of m
    # To find squarefree divisors of m, factor m first
    
    # Simpler: just generate by trying all b (squarefree) and all m multiples of b
    # For squarefree b up to limit, m from b to sqrt(limit/b) step b:
    #   n = m^2 * b <= limit
    # Count: sum over squarefree b of sqrt(limit/b)/b ≈ sum 1/b^0.5 over squarefree b up to limit
    # This is manageable for limit = 10^7
    
    # Generate squarefree numbers up to limit
    # Use inclusion or just check
    # For efficiency, let's limit b to small values since m^2*b <= limit means b <= limit/m^2
    # For m=1, b <= limit. For m=10^3, b <= limit/10^6
    
    # Let's just do it directly
    for m in range(1, sqrt_limit + 1):
        # Find all squarefree divisors b of m such that m^2 * b <= limit
        # Factor m
        temp = m
        factors = []
        for p in range(2, isqrt(temp) + 1):
            while temp % p == 0:
                factors.append(p)
                temp //= p
        if temp > 1:
            factors.append(temp)
        # Remove duplicates for squarefree divisors
        unique_primes = list(set(factors))
        # Enumerate all subsets of unique_primes to get squarefree divisors
        n_primes = len(unique_primes)
        for mask in range(1 << n_primes):
            b = 1
            for i in range(n_primes):
                if mask & (1 << i):
                    b *= unique_primes[i]
            if b > limit // (m * m):
                continue
            powerful_set.add(m * m * b)
    
    return sorted(powerful_set)

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    # Step 1: Generate powerful numbers up to a reasonable limit
    # We'll search for N up to N_MAX. Need powerful numbers up to N_MAX.
    # Let's start with N_MAX = 10^7, powerful numbers up to 10^7
    N_MAX = 10_000_000
    
    powerful = _generate_powerful(N_MAX)
    powerful_set = set(powerful)
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Fallback: return a small valid result
        N = 2
        return {'N': N, 'pairs': [[1, 1]]}
    
    # Step 2: For each N, count representations
    # To be efficient, iterate over powerful a and powerful b = N - a
    # For a given N, count = sum over a in powerful with a <= N-a of (N-a in powerful_set)
    # This is O(len(powerful)) per N, which is too slow for N_MAX = 10^7
    
    # Better: for each pair (a, b) with a <= b, a+b = N, mark N
    # Use convolution-like approach or just iterate
    # Since len(powerful) for N_MAX=10^7 is about 10^3-10^4, 
    # we can do: for each a in powerful, for each b in powerful with b >= a:
    #   N = a + b, if N <= N_MAX: count[N] += 1
    # This is O(len(powerful)^2) which for 10^4 is 10^8, too slow in Python
    
    # Alternative: for each a in powerful, for each b in powerful:
    #   if a <= b and a+b <= N_MAX: count[a+b] += 1
    # Still O(P^2)
    
    # Let's use numpy for speed
    import numpy as np
    arr = np.array(powerful)
    # For each a, b = N - a, need b in powerful_set
    # Let's just do it in a smarter way
    # For each a in powerful (sorted), the valid N range is [2*a, N_MAX]
    # b = N - a must be in powerful_set
    # So for each a, N = a + b for b in powerful with b >= a and a+b <= N_MAX
    
    # Let's use a dictionary for counts
    from collections import defaultdict
    count = defaultdict(int)
    pairs_dict = defaultdict(list)
    
    P = len(powerful)
    # For efficiency, for each a, binary search the range of b
    import bisect
    
    for i, a in enumerate(powerful):
        if 2 * a > N_MAX:
            break
        # b >= a, b <= N_MAX - a
        # Find b in powerful in [a, N_MAX - a]
        lo = bisect.bisect_left(powerful, a)
        hi = bisect.bisect_right(powerful, N_MAX - a)
        for j in range(lo, hi):
            b = powerful[j]
            N = a + b
            count[N] += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    if time.time() > deadline:
        # Return best found so far
        if count:
            best_N = max(count, key=lambda n: count[n])
            best_count = count[best_N]
            # Reconstruct pairs
            pairs = []
            for a in powerful:
                if 2 * a > best_N:
                    break
                b = best_N - a
                if b >= a and b in powerful_set:
                    pairs.append([a, b])
            return {'N': best_N, 'pairs': pairs}
        else:
            return {'N': 2, 'pairs': [[1, 1]]}
    
    best_N = max(count, key=lambda n: count[n])
    best_count = count[best_N]
    pairs = []
    for a in powerful:
        if 2 * a > best_N:
            break
        b = best_N - a
        if b >= a and b in powerful_set:
            pairs.append([a, b])
    
    return {'N': best_N, 'pairs': pairs}
# EVOLVE-BLOCK-END
