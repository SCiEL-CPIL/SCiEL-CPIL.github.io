import math
import time
from itertools import product

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # Safety margin before 120s limit
    
    LIMIT = 10**18
    
    def get_divisors_from_factors(factors):
        """Generate all divisors from list of (prime, exponent) pairs."""
        divisors = [1]
        for p, e in factors:
            new_divs = []
            for d in divisors:
                val = d
                for _ in range(e + 1):
                    new_divs.append(val)
                    val *= p
            divisors = new_divs
        divisors.sort()
        return divisors
    
    def tau_perp(n, factors=None):
        """Compute the score: number of adjacent coprime pairs in sorted divisors."""
        if factors is None:
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
                p += 1 if p == 2 else 2
            if temp > 1:
                factors.append((temp, 1))
        
        divisors = get_divisors_from_factors(factors)
        score = 0
        for i in range(len(divisors) - 1):
            if math.gcd(divisors[i], divisors[i+1]) == 1:
                score += 1
        return score, len(divisors)
    
    # Candidate 1: Product of first 9 primes
    primes_9 = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    n1 = 1
    for p in primes_9:
        n1 *= p
    score1, num_div1 = tau_perp(n1, [(p, 1) for p in primes_9])
    
    best_n = n1
    best_score = score1
    
    # Try product of first 10 primes
    primes_10 = primes_9 + [29]
    n2 = n1 * 29
    if n2 <= LIMIT:
        score2, num_div2 = tau_perp(n2, [(p, 1) for p in primes_10])
        if score2 > best_score:
            best_n = n2
            best_score = score2
    
    # Try some numbers with mixed exponents
    # n = 2^a * 3^b * 5^c * ... to get more divisors
    
    # Let's try: maximize number of divisors while keeping n <= 10^18
    # More divisors means more potential adjacent pairs
    
    # Strategy: try various exponent combinations
    # For squarefree with k primes: 2^k divisors
    # With exponents: product(a_i + 1) divisors
    
    # Let's try n = 2^a * 3^b * 5^c * 7^d * 11^e * 13^f * 17^g * 19^h
    # and see which gives best score
    
    # First, let's understand the structure better.
    # For squarefree n, divisors are subsets. Two adjacent divisors in sorted order
    # are coprime iff their prime supports are disjoint.
    
    # Let's try a few more constructions:
    
    # n = 2^1 * 3^1 * 5^1 * 7^1 * 11^1 * 13^1 * 17^1 * 19^1 * 23^1 (512 divisors)
    # n = 2^2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 = 4 * 223092870 / 2 = ... let me compute
    # Actually 2^2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 = 4 * 9699690 = 38798760
    # That's 3*2^7 = 384 divisors. Less than 512.
    
    # So squarefree with more primes is better for divisor count.
    
    # Can we fit 10 primes? 2*3*5*7*11*13*17*19*23*29 = 6469693230
    # That's ~6.5e9, well under 10^18.
    # 11 primes: * 31 = 200560490130 ~ 2e11
    # 12 primes: * 37 = 7420738134810 ~ 7.4e12
    # 13 primes: * 41 = 304250263527210 ~ 3e14
    # 14 primes: * 43 = 13082761331670030 ~ 1.3e16
    # 15 primes: * 47 = 614889782588491410 ~ 6.1e17
    # 16 primes: * 53 = 32589158477190044730 > 10^18. No good.
    
    # So we can fit up to 15 primes!
    primes_15 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    n3 = 1
    for p in primes_15:
        n3 *= p
    
    if n3 <= LIMIT:
        score3, num_div3 = tau_perp(n3, [(p, 1) for p in primes_15])
        if score3 > best_score:
            best_n = n3
            best_score = score3
    
    # Check time
    if time.time() - start_time > DEADLINE:
        return best_n
    
    # Try 14 primes (in case 15 is too large or has lower score)
    primes_14 = primes_15[:14]
    n4 = 1
    for p in primes_14:
        n4 *= p
    if n4 <= LIMIT:
        score4, num_div4 = tau_perp(n4, [(p, 1) for p in primes_14])
        if score4 > best_score:
            best_n = n4
            best_score = score4
    
    if time.time() - start_time > DEADLINE:
        return best_n
    
    # Now let's think about whether non-squarefree could be better.
    # With 15 primes squarefree: 2^15 = 32768 divisors.
    # Could we get more divisors with fewer primes but higher exponents?
    # E.g., 2^a * 3^b * ... where product(a_i+1) > 32768
    # But the coprime adjacency might be lower because divisors cluster around powers of small primes.
    
    # Let's try a few non-squarefree candidates:
    # n = 2^5 * 3^4 * 5^3 * 7^2 * 11^2 * 13 * 17 * 19 * 23
    # Divisors: 6*5*4*3*3*2*2*2*2 = 6*5*4*3*3*16 = 17280. Less than 32768.
    
    # n = 2^4 * 3^4 * 5^3 * 7^3 * 11^2 * 13^2 * 17 * 19
    # = 16 * 81 * 125 * 343 * 121 * 169 * 17 * 19
    # Let me compute: 2^4=16, 3^4=81, 5^3=125, 7^3=343, 11^2=121, 13^2=169, 17, 19
    # 16*81 = 1296
    # 1296*125 = 162000
    # 162000*343 = 55566000
    # 55566000*121 = 6723486000
    # 6723486000*169 = 1136269134000
    # 1136269134000*17 = 19316575278000
    # 19316575278000*19 = 367014930282000 ~ 3.67e14 < 10^18
    # Divisors: 5*5*4*4*3*3*2*2 = 5*5*4*4*3*3*4 = 7200. Less than 32768.
    
    # It seems squarefree with many primes gives the most divisors.
    # Let's verify: what's the max number of divisors for n <= 10^18?
    # It's around 103680 for highly composite numbers, but those have few distinct primes.
    # 2^a * 3^b * 5^c * ... with many primes but small exponents.
    
    # Actually, for maximizing coprime adjacencies, we want:
    # 1. Many divisors (more potential pairs)
    # 2. Divisors spread out so that adjacent ones tend to be coprime
    
    # Squarefree numbers have the property that each divisor is a product of distinct primes,
    # and coprime means disjoint supports. This should give good coprime adjacency.
    
    # Let's also try: what if we use the 15 primes but replace the largest with a smaller
    # prime to allow an extra prime? No, we already use the smallest primes.
    
    # What about n = product of first 15 primes * some small factor?
    # That would make it non-squarefree and reduce the "spread" of coprime pairs.
    
    # Let's also try replacing some primes to see if different prime sets help.
    # Actually, the specific primes shouldn't matter much for the coprime structure,
    # only the number of distinct primes and their relative sizes (which affect ordering).
    
    # Let me try a slightly different approach: use 15 primes but check if 
    # using primes up to 47 is optimal or if we should adjust.
    
    # Actually, let's just go with the 15-prime squarefree number if it fits.
    
    # Verify n3
    print(f"n3 = {n3}, divisors = {num_div3}, score = {score3}")
    print(f"n1 = {n1}, divisors = {num_div1}, score = {score1}")
    
    if time.time() - start_time > DEADLINE:
        return best_n
    
    # Let's try to see if we can do better with a different structure.
    # What if we use 14 primes with one having exponent 2?
    # n = 2^2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43
    # = 4 * (3*5*7*11*13*17*19*23*29*31*37*41*43)
    # Product of 3 through 43: 43! / (2 * 1!) ... let me just compute
    prod_3_43 = 3*5*7*11*13*17*19*23*29*31*37*41*43
    n5 = 4 * prod_3_43
    if n5 <= LIMIT:
        factors5 = [(2, 2)] + [(p, 1) for p in [3,5,7,11,13,17,19,23,29,31,37,41,43]]
        score5, num_div5 = tau_perp(n5, factors5)
        if score5 > best_score:
            best_n = n5
            best_score = score5
    
    if time.time() - start_time > DEADLINE:
        return best_n
    
    # Try 13 primes with two having exponent 2
    # n = 2^2 * 3^2 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41
    prod_rest = 5*7*11*13*17*19*23*29*31*37*41
    n6 = 4 * 9 * prod_rest
    if n6 <= LIMIT:
        factors6 = [(2, 2), (3, 2)] + [(p, 1) for p in [5,7,11,13,17,19,23,29,31,37,41]]
        score6, num_div6 = tau_perp(n6, factors6)
        if score6 > best_score:
            best_n = n6
            best_score = score6
    
    if time.time() - start_time > DEADLINE:
        return best_n
    
    return best_n
