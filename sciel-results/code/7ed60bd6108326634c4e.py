import math
import time

def generate_divisors(factors):
    """
    factors: list of (prime, exponent)
    returns sorted list of divisors
    """
    divisors = [1]
    for p, e in factors:
        new_divisors = []
        power = 1
        for _ in range(e + 1):
            for d in divisors:
                new_divisors.append(d * power)
            power *= p
        divisors = new_divisors
    divisors.sort()
    return divisors

def tau_perp(n):
    if n < 2:
        return 0
    
    # Factorize n
    factors = []
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            e = 0
            while temp % p == 0:
                temp //= p
                e += 1
            factors.append((p, e))
        p += 1 if p == 2 else 2 # Simple sieve-like step, though for 10^18 this is slow if n is prime.
        # But our candidates are composite with small factors.
    if temp > 1:
        factors.append((temp, 1))
        
    divisors = generate_divisors(factors)
    count = 0
    for i in range(len(divisors) - 1):
        if math.gcd(divisors[i], divisors[i+1]) == 1:
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    candidates = []
    
    # Candidate 1: Primorial of first 15 primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    p15 = 1
    for p in primes:
        p15 *= p
    if p15 <= 10**18:
        candidates.append(p15)
    else:
        # Should not happen, 6.1e17
        p15 //= primes[-1]
        candidates.append(p15)
        
    # Candidate 2: Highly composite number near 1e18
    # 963460800000000000 = 2^11 * 3^7 * 5^5 * 7^3 * 11^2 * 13^2 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43
    hcn = 963460800000000000
    if hcn <= 10**18:
        candidates.append(hcn)
        
    # Candidate 3: Try a variant with slightly different exponents
    # e.g. 2^10 * 3^7 * 5^5 * 7^3 * 11^2 * 13^2 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47 ?
    # hcn / 2 * 47 = 963460800000000000 / 2 * 47 = 481730400000000000 * 47 = 22641328800000000000 > 1e18. No.
    
    # Candidate 4: P14 * 47^2?
    p14 = 1
    for p in primes[:14]:
        p14 *= p
    c4 = p14 * 47 * 47
    if c4 <= 10**18:
        candidates.append(c4)
        
    best_n = max(candidates)
    best_score = 0
    
    for n in candidates:
        if time.time() > deadline:
            break
        s = tau_perp(n)
        if s > best_score:
            best_score = s
            best_n = n
            
    return best_n

# EVOLVE-BLOCK-START
import math
import time
import itertools

def _factorize(n):
    """Factorize n into prime factors."""
    factors = []
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            e = 0
            while temp % p == 0:
                temp //= p
                e += 1
            factors.append((p, e))
        p += 1 if p == 2 else 2
    if temp > 1:
        factors.append((temp, 1))
    return factors

def _generate_divisors_from_factors(factors):
    """Generate sorted divisors from prime factorization."""
    divisors = [1]
    for p, e in factors:
        new_divs = []
        power = 1
        for _ in range(e + 1):
            for d in divisors:
                new_divs.append(d * power)
            power *= p
        divisors = new_divs
    divisors.sort()
    return divisors

def _score_from_factors(factors):
    """Compute tau_perp from prime factorization without generating all divisors if possible."""
    # For numbers with many divisors, we need to generate them
    divisors = _generate_divisors_from_factors(factors)
    score = 0
    for i in range(len(divisors) - 1):
        if math.gcd(divisors[i], divisors[i+1]) == 1:
            score += 1
    return score

def _primes_up_to(limit):
    """Generate primes up to limit."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Get primes up to 100
    primes = _primes_up_to(100)
    
    # Strategy 1: Squarefree numbers with k distinct primes
    # For squarefree n with primes p1 < p2 < ... < pk, 
    # the divisors are all products of subsets.
    # Adjacent coprime pairs occur when consecutive divisors in sorted order share no prime.
    
    # For small k, we can enumerate prime sets
    best_n = 2
    best_score = 0
    
    # Try squarefree numbers with different prime sets
    # Start with known good candidates
    candidate_sets = []
    
    # Primorial of first 15 primes
    primes_15 = primes[:15]
    p15 = 1
    for p in primes_15:
        p15 *= p
    if p15 <= 10**18:
        candidate_sets.append(primes_15)
    
    # Try replacing some primes with larger ones to get more coprime pairs
    # The key insight: for squarefree numbers, the divisor lattice structure
    # means that coprime adjacent pairs depend on the prime set structure.
    
    # Let's try systematic search: start with 15 smallest primes, then try
    # swapping out some for larger primes
    base_set = set(primes_15)
    
    # Try all single swaps
    for i in range(len(primes_15)):
        for j in range(15, len(primes)):
            new_set = list(base_set)
            new_set[i] = primes[j]
            n = 1
            for p in new_set:
                n *= p
            if n <= 10**18:
                candidate_sets.append(sorted(new_set))
    
    # Try double swaps
    for i in range(len(primes_15)):
        for j in range(i+1, len(primes_15)):
            for k in range(15, min(20, len(primes))):
                for l in range(k+1, min(20, len(primes))):
                    new_set = list(base_set)
                    new_set[i] = primes[k]
                    new_set[j] = primes[l]
                    n = 1
                    for p in new_set:
                        n *= p
                    if n <= 10**18:
                        candidate_sets.append(sorted(new_set))
    
    # Also try with 14 primes and higher exponents on small primes
    # e.g., 2^a * 3^b * 5^c * ... where a >= b >= c >= ...
    
    # Generate candidate numbers
    candidates = []
    
    # From prime sets (squarefree)
    for pset in candidate_sets:
        n = 1
        for p in pset:
            n *= p
        if 2 <= n <= 10**18:
            candidates.append(n)
    
    # Try numbers with exponents: 2^a * 3^b * 5^c * 7^d * ... 
    # where exponents are non-increasing
    # Start with base: 2^10 * 3^6 * 5^4 * 7^3 * 11^2 * 13^2 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43
    base_exponents = [(2,10), (3,6), (5,4), (7,3), (11,2), (13,2), (17,1), (19,1), (23,1), (29,1), (31,1), (37,1), (41,1), (43,1)]
    base_n = 1
    for p, e in base_exponents:
        base_n *= (p ** e)
    if base_n <= 10**18:
        candidates.append(base_n)
    
    # Try variations: increase some exponents, decrease others
    # 2^11 * 3^6 * 5^4 * 7^3 * 11^2 * 13^2 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43
    for exp2 in [10, 11, 12]:
        for exp3 in [5, 6, 7]:
            for exp5 in [4, 5]:
                for exp7 in [2, 3]:
                    for exp11 in [1, 2]:
                        for exp13 in [1, 2]:
                            n = (2**exp2) * (3**exp3) * (5**exp5) * (7**exp7) * (11**exp11) * (13**exp13)
                            for p in [17, 19, 23, 29, 31, 37, 41, 43]:
                                n *= p
                            if 2 <= n <= 10**18:
                                candidates.append(n)
    
    # Remove duplicates
    candidates = list(set(candidates))
    
    # Evaluate candidates
    for n in candidates:
        if time.time() > deadline:
            break
        factors = _factorize(n)
        score = _score_from_factors(factors)
        if score > best_score:
            best_score = score
            best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
