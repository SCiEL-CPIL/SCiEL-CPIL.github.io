import math
import time

def get_divisors(n):
    divs = [1]
    i = 2
    while i * i <= n:
        if n % i == 0:
            # If n is divisible by i, then n/i is also a divisor
            # But this simple method is slow for large n.
            # Better to factorize n first.
            pass
        i += 1
    # This is too slow. We need factorization based generation.
    return divs

def factorize(n):
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def generate_divisors(factors):
    divs = [1]
    for p, e in factors.items():
        new_divs = []
        power = 1
        for _ in range(e + 1):
            for d in divs:
                new_divs.append(d * power)
            power *= p
        divs = new_divs
    return divs

def calc_score(n):
    if n < 2:
        return 0
    factors = factorize(n)
    divs = generate_divisors(factors)
    divs.sort()
    score = 0
    for i in range(len(divs) - 1):
        if math.gcd(divs[i], divs[i+1]) == 1:
            score += 1
    return score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Candidate 1: Largest Primorial
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    p15 = 1
    for p in primes:
        p15 *= p
    # p15 is 614889782588491410
    
    candidates = [p15]
    
    # Generate some variants
    # Increase exponent of 2
    if p15 * 2 <= 10**18:
        candidates.append(p15 * 2)
    if p15 * 4 <= 10**18:
        candidates.append(p15 * 4)
        
    # Try replacing largest prime with power of 2?
    # e.g. remove 47, add 2^k
    # p15 / 47 * 2^k
    base = p15 // 47
    k = 1
    while base * (2**k) <= 10**18:
        candidates.append(base * (2**k))
        k += 1
        
    # Try highly composite number
    # The number with max divisors <= 10^18 is 897612484786617600
    # Let's add it
    candidates.append(897612484786617600)
    
    best_n = max(candidates)
    best_score = -1
    
    # Evaluate initial candidates
    for n in candidates:
        if time.time() > deadline:
            break
        if n < 2 or n > 10**18:
            continue
        s = calc_score(n)
        if s > best_score:
            best_score = s
            best_n = n
            
    # Local Search
    # We will try to improve best_n
    # We need the factorization of best_n to modify it
    # Since best_n is likely square-free or close, we can work with its prime factors
    
    # To allow general local search, we represent n by its prime factorization
    # But modifying factorization and checking product is tricky.
    # Simpler: Just try a set of hand-picked "good" structures.
    
    # Let's add more specific candidates
    # P14 * 2^2
    p14 = p15 // 47
    candidates.append(p14 * 4)
    # P13 * 2^3
    p13 = p14 // 43
    candidates.append(p13 * 8)
    
    # Re-evaluate all unique candidates
    seen = set()
    for n in candidates:
        if n in seen:
            continue
        seen.add(n)
        if time.time() > deadline:
            break
        if n < 2 or n > 10**18:
            continue
        s = calc_score(n)
        if s > best_score:
            best_score = s
            best_n = n
            
    return best_n

# EVOLVE-BLOCK-START
def construct():
    import time
    import math
    
    def factorize(n):
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return factors

    def generate_divisors(factors):
        divs = [1]
        for p, e in factors.items():
            new_divs = []
            power = 1
            for _ in range(e + 1):
                for d in divs:
                    new_divs.append(d * power)
                power *= p
            divs = new_divs
        return divs

    def calc_score(n):
        if n < 2:
            return 0
        factors = factorize(n)
        divs = generate_divisors(factors)
        divs.sort()
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score

    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    # Calculate P15
    p15 = 1
    for p in primes:
        p15 *= p
        
    candidates = [p15]
    
    # Add variants
    # 1. P15 * 2 (if fits)
    if p15 * 2 <= 10**18:
        candidates.append(p15 * 2)
        
    # 2. P14 * 2^2
    p14 = p15 // 47
    if p14 * 4 <= 10**18:
        candidates.append(p14 * 4)
        
    # 3. P13 * 2^3
    p13 = p14 // 43
    if p13 * 8 <= 10**18:
        candidates.append(p13 * 8)
        
    # 4. P12 * 2^4
    p12 = p13 // 41
    if p12 * 16 <= 10**18:
        candidates.append(p12 * 16)
        
    # 5. Highly Composite Number (max divisors)
    candidates.append(897612484786617600)
    
    # 6. Another highly composite number
    candidates.append(735134400) # Small, just for safety
    
    # 7. P15 with 47 replaced by 2^2 (i.e. P14 * 4) is already there.
    # Let's try P15 * 3? No, 15*3 > 10^18 likely? 
    # P15 ~ 6.1e17. *3 = 1.8e18 > 1e18. So no.
    
    # Let's try a few random highly composite-like numbers
    # 2^a * 3^b * 5^c ...
    
    best_n = 2
    best_score = -1
    seen = set()
    
    for n in candidates:
        if time.time() > deadline:
            break
        if n in seen:
            continue
        seen.add(n)
        if n < 2 or n > 10**18:
            continue
        s = calc_score(n)
        if s > best_score:
            best_score = s
            best_n = n
            
    # If we have time, do a small local search around best_n
    # This is complex to implement generally without breaking the factorization.
    # Given the discrete nature, the hand-picked candidates likely cover the optimum.
    # The primorial P15 is the strongest candidate for "many distinct primes".
    
    return best_n
# EVOLVE-BLOCK-END
