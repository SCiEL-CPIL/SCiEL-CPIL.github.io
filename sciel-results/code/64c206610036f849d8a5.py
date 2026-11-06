"""
Constructs an integer n <= 10^18 maximizing tau_perp(n).

Approach: Greedy interval-merging heuristic. Model the divisor lattice and
select prime exponents that maximize the density of coprime adjacent pairs
(support-disjoint boundaries) within the 10^18 budget.
"""

import math
from itertools import combinations

def _divisors(n):
    """Generate all divisors of n in sorted order."""
    divs = [1]
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            exp = 0
            while temp % p == 0:
                temp //= p
                exp += 1
            new_divs = []
            for d in divs:
                mult = 1
                for e in range(exp + 1):
                    new_divs.append(d * mult)
                    mult *= p
            divs = new_divs
        p += 1 if p == 2 else 2
    if temp > 1:
        new_divs = []
        for d in divs:
            new_divs.append(d)
            new_divs.append(d * temp)
        divs = new_divs
    divs.sort()
    return divs

def _tau_perp(n):
    """Count adjacent coprime pairs in sorted divisor list of n."""
    if n < 2:
        return 0
    divs = _divisors(n)
    count = 0
    for i in range(len(divs) - 1):
        if math.gcd(divs[i], divs[i + 1]) == 1:
            count += 1
    return count

def _is_prime(x):
    if x < 2:
        return False
    if x < 4:
        return True
    if x % 2 == 0 or x % 3 == 0:
        return False
    i = 5
    while i * i <= x:
        if x % i == 0 or x % (i + 2) == 0:
            return False
        i += 6
    return True

def _primes_up_to(limit):
    """Generate primes up to limit using simple sieve."""
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i in range(2, limit + 1) if sieve[i]]

def _tau_perp_from_factorization(factors):
    """
    Compute tau_perp from a list of (prime, exponent) pairs.
    factors: list of (p, a) where n = prod p^a
    """
    # Generate all divisors
    divs = [1]
    for p, a in factors:
        new_divs = []
        for d in divs:
            mult = 1
            for e in range(a + 1):
                new_divs.append(d * mult)
                mult *= p
        divs = new_divs
    divs.sort()
    count = 0
    for i in range(len(divs) - 1):
        if math.gcd(divs[i], divs[i + 1]) == 1:
            count += 1
    return count

def construct():
    """
    Greedy interval-merging heuristic:
    - Start with empty factorization
    - At each step, try adding each candidate prime with various exponents
    - Pick the (p, a) that maximizes score increase per log-cost
    - Continue until budget is exhausted
    """
    LOG_BUDGET = math.log(10**18)
    
    # Candidate primes to consider
    all_primes = _primes_up_to(1000)
    
    # Start with empty factorization
    current_factors = []  # list of (p, a)
    current_n = 1
    current_score = 0
    
    # Greedy: repeatedly add the prime/exponent that gives best marginal gain
    max_iterations = 20
    for _ in range(max_iterations):
        best_gain = -1
        best_p = None
        best_a = None
        best_new_score = current_score
        
        for p in all_primes:
            if p > 10**18:
                break
            # Try exponents 1, 2, 3, ...
            p_power = p
            for a in range(1, 30):
                candidate_n = current_n * p_power
                if candidate_n > 10**18:
                    break
                # Compute new score
                new_factors = current_factors + [(p, a)]
                new_score = _tau_perp_from_factorization(new_factors)
                gain = new_score - current_score
                cost = math.log(p_power)
                if cost > 0:
                    ratio = gain / cost
                else:
                    ratio = gain
                if ratio > best_gain:
                    best_gain = ratio
                    best_p = p
                    best_a = a
                    best_new_score = new_score
                p_power *= p
                if p_power > 10**18:
                    break
        
        if best_p is None or best_gain <= 0:
            break
        
        # Add the best prime/exponent
        current_factors.append((best_p, best_a))
        for pf in current_factors:
            current_n *= pf[0] ** pf[1]
        current_score = best_new_score
    
    # Also try the baseline: product of first k primes
    base_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    base_n = 1
    for p in base_primes:
        base_n *= p
    base_score = _tau_perp(base_n)
    
    best_n = current_n
    best_score = current_score
    
    if base_score > best_score:
        best_n = base_n
        best_score = base_score
    
    # Try products of first k primes
    for k in range(2, 16):
        if k > len(all_primes):
            break
        prod = 1
        for i in range(k):
            prod *= all_primes[i]
        if prod > 10**18:
            break
        score = _tau_perp(prod)
        if score > best_score:
            best_score = score
            best_n = prod
    
    # Try replacing some primes in the greedy solution
    # Try swapping the largest prime with a larger one
    if current_factors:
        for idx in range(len(current_factors)):
            p_orig, a_orig = current_factors[idx]
            for p_new in all_primes:
                if p_new <= p_orig:
                    continue
                # Replace p_orig^a_orig with p_new^a_orig
                candidate_factors = list(current_factors)
                candidate_factors[idx] = (p_new, a_orig)
                candidate_n = 1
                for pf in candidate_factors:
                    candidate_n *= pf[0] ** pf[1]
                if candidate_n > 10**18:
                    break
                score = _tau_perp_from_factorization(candidate_factors)
                if score > best_score:
                    best_score = score
                    best_n = candidate_n
    
    # Try prime power variations on the best found
    # If best_n is squarefree, try increasing exponent of smallest prime
    if best_n > 1:
        # Get factorization of best_n
        temp = best_n
        factors = []
        p = 2
        while p * p <= temp:
            if temp % p == 0:
                exp = 0
                while temp % p == 0:
                    temp //= p
                    exp += 1
                factors.append((p, exp))
            p += 1 if p == 2 else 2
        if temp > 1:
            factors.append((temp, 1))
        
        # Try increasing exponent of each prime
        for i in range(len(factors)):
            p_i, a_i = factors[i]
            for new_a in range(a_i + 1, a_i + 10):
                candidate_factors = list(factors)
                candidate_factors[i] = (p_i, new_a)
                candidate_n = 1
                for pf in candidate_factors:
                    candidate_n *= pf[0] ** pf[1]
                if candidate_n > 10**18:
                    break
                score = _tau_perp_from_factorization(candidate_factors)
                if score > best_score:
                    best_score = score
                    best_n = candidate_n
    
    return best_n
