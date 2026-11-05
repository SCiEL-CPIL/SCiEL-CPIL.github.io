import math
import time
from itertools import combinations

def compute_score_from_primes(primes):
    """Compute tau_perp for a squarefree number given its prime factors."""
    # Generate all divisors
    divs = [1]
    for p in primes:
        divs.extend([d * p for d in divs[:]])
    divs.sort()
    count = 0
    for i in range(len(divs) - 1):
        if math.gcd(divs[i], divs[i + 1]) == 1:
            count += 1
    return count

def compute_score_general(n):
    """Compute tau_perp for any n."""
    if n < 2:
        return 0
    # Factorize
    temp = n
    factors = []
    d = 2
    while d * d <= temp:
        if temp % d == 0:
            cnt = 0
            while temp % d == 0:
                temp //= d
                cnt += 1
            factors.append((d, cnt))
        d += 1 if d == 2 else 2
    if temp > 1:
        factors.append((temp, 1))
    
    # Generate divisors
    divs = [1]
    for p, e in factors:
        current = list(divs)
        mult = 1
        for _ in range(e):
            mult *= p
            divs.extend([d * mult for d in current])
    
    divs.sort()
    count = 0
    for i in range(len(divs) - 1):
        if math.gcd(divs[i], divs[i + 1]) == 1:
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s kill
    
    best_score = 0
    best_n = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47
    
    # First, evaluate the baseline
    best_score = compute_score_from_primes([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47])
    
    # Search: try replacing some of the larger primes with even larger ones
    # The product of first 14 primes is about 1.3e17, so we can multiply by primes up to ~7000
    base14 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]
    base14_prod = 1
    for p in base14:
        base14_prod *= p
    
    # Try single replacements for the 15th prime
    # Primes to try: 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, and larger
    candidate_primes = []
    p = 47
    while base14_prod * p <= 10**18:
        candidate_primes.append(p)
        # Next prime (simple sieve for small range)
        p += 1
        while p > 2 and any(p % d == 0 for d in range(2, int(p**0.5) + 1)):
            p += 1
    
    # Evaluate candidates with 15 primes (14 base + 1 large)
    for p in candidate_primes:
        if time.time() > deadline:
            break
        primes_set = base14 + [p]
        score = compute_score_from_primes(primes_set)
        if score > best_score:
            best_score = score
            n = base14_prod * p
            best_n = n
    
    # Also try: replace 2 of the last primes with 2 larger ones
    base13 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
    base13_prod = 1
    for p in base13:
        base13_prod *= p
    
    # Try pairs (p1, p2) where p1 < p2, p1 >= 43, p2 > p1
    # base13_prod * 43 * 47 = ~1.3e17 * 2021 ~ 2.6e20... too big
    # Actually base13_prod * 43 = ~1.3e17, so p2 can be up to ~7000
    # Let's try p1 = 43, 47, 53, ... and p2 > p1
    p1_candidates = [43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    for p1 in p1_candidates:
        if time.time() > deadline:
            break
        max_p2 = int(10**18 / (base13_prod * p1))
        if max_p2 < p1 + 2:
            continue
        # Try a few p2 values: the smallest prime > p1, and some larger ones
        p2_start = p1 + 1
        while p2_start <= 2 and p2_start * base13_prod * p1 <= 10**18:
            p2_start += 1
        # Just try p1+1 if prime, and a few others
        tested_p2 = set()
        for offset in [1, 2, 4, 6, 8, 10, 14, 18, 22, 30, 40, 60, 100, 200, 500, 1000, 5000]:
            p2 = p1 + offset
            if p2 in tested_p2:
                continue
            if p2 * base13_prod * p1 > 10**18:
                continue
            # Check if p2 is prime
            if p2 < 2:
                continue
            is_prime = True
            for d in range(2, int(p2**0.5) + 1):
                if p2 % d == 0:
                    is_prime = False
                    break
            if not is_prime:
                continue
            tested_p2.add(p2)
            primes_set = base13 + [p1, p2]
            score = compute_score_from_primes(primes_set)
            if score > best_score:
                best_score = score
                best_n = base13_prod * p1 * p2
    
    # Try non-squarefree: high divisor count numbers
    # n = 2^a * 3^b * 5^c * ... with product <= 10^18
    # The number of divisors is (a+1)(b+1)(c+1)...
    # More divisors = more potential coprime pairs
    
    # Try some hand-picked non-squarefree candidates
    test_candidates = [
        # 2^10 * 3^6 * 5^4 * 7^2 * 11 * 13 * 17 * 19 * 23 * 29
        (2, 10), (3, 6), (5, 4), (7, 2), (11, 1), (13, 1), (17, 1), (19, 1), (23, 1), (29, 1),
    ]
    # Build the number
    n_test = 1
    for p, e in test_candidates:
        n_test *= p ** e
    if n_test <= 10**18:
        score = compute_score_general(n_test)
        if score > best_score:
            best_score = score
            best_n = n_test
    
    # Try another non-squarefree
    test_candidates2 = [(2, 8), (3, 5), (5, 3), (7, 2), (11, 1), (13, 1), (17, 1), (19, 1), (23, 1), (29, 1), (31, 1)]
    n_test2 = 1
    for p, e in test_candidates2:
        n_test2 *= p ** e
    if n_test2 <= 10**18:
        score = compute_score_general(n_test2)
        if score > best_score:
            best_score = score
            best_n = n_test2
    
    # Try: 2^5 * 3^4 * 5^3 * 7^2 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37
    test_candidates3 = [(2, 5), (3, 4), (5, 3), (7, 2), (11, 1), (13, 1), (17, 1), (19, 1), (23, 1), (29, 1), (31, 1), (37, 1)]
    n_test3 = 1
    for p, e in test_candidates3:
        n_test3 *= p ** e
    if n_test3 <= 10**18:
        score = compute_score_general(n_test3)
        if score > best_score:
            best_score = score
            best_n = n_test3
    
    # Try maximizing divisor count: highly composite-like
    # 2^9 * 3^5 * 5^3 * 7^2 * 11 * 13 * 17 * 19 = ?
    test_candidates4 = [(2, 9), (3, 5), (5, 3), (7, 2), (11, 1), (13, 1), (17, 1), (19, 1)]
    n_test4 = 1
    for p, e in test_candidates4:
        n_test4 *= p ** e
    if n_test4 <= 10**18:
        score = compute_score_general(n_test4)
        if score > best_score:
            best_score = score
            best_n = n_test4
    
    # Try: 2^6 * 3^4 * 5^3 * 7^2 * 11^2 * 13 * 17 * 19 * 23
    test_candidates5 = [(2, 6), (3, 4), (5, 3), (7, 2), (11, 2), (13, 1), (17, 1), (19, 1), (23, 1)]
    n_test5 = 1
    for p, e in test_candidates5:
        n_test5 *= p ** e
    if n_test5 <= 10**18:
        score = compute_score_general(n_test5)
        if score > best_score:
            best_score = score
            best_n = n_test5
    
    return best_n
