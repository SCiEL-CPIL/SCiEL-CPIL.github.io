import math
import time
import itertools
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s limit
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

    best_n = 1
    best_score = 0

    # Helper: evaluate a candidate given its factor list
    def try_candidate(factors):
        nonlocal best_n, best_score
        n = 1
        for p, e in factors:
            n *= p ** e
        if n > LIMIT:
            return
        score, _ = tau_perp(n, factors)
        if score > best_score:
            best_score = score
            best_n = n
        return score, n

    # Baseline: product of first 15 primes
    primes_15 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    n_base = 1
    for p in primes_15:
        n_base *= p
    if n_base <= LIMIT:
        score_base, _ = tau_perp(n_base, [(p, 1) for p in primes_15])
        best_n = n_base
        best_score = score_base

    if time.time() - start_time > DEADLINE:
        return best_n

    # === GREEDY INTERVAL-MERGING HEURISTIC ===
    # Model: each prime p with exponent e creates "intervals" in log-space.
    # Adjacent coprime pairs occur at boundaries between divisors with disjoint supports.
    # We greedily select primes/exponents to maximize boundary density within budget.
    #
    # Key insight: for squarefree n with primes p1 < p2 < ... < pk,
    # the divisors sorted by value have coprime adjacencies where consecutive
    # divisors share no prime factor. The density of such pairs depends on
    # how "spread out" the primes are in log-space.
    #
    # Greedy strategy:
    # 1. Start with the smallest primes (maximizes count within budget)
    # 2. Try swapping in larger primes to improve log-spacing
    # 3. Try adding exponents to small primes to create more divisors
    # 4. Use a scoring heuristic to guide search

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

    # Generate primes up to 300
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                    53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    for p in range(101, 300):
        if is_prime(p):
            small_primes.append(p)

    # --- Phase 1: Explore 15-prime squarefree sets with log-spacing optimization ---
    # The product of the first 15 primes is ~6.15e17, leaving room for one more
    # or for swapping to slightly larger primes.
    # 
    # Greedy: fix the first 12 primes, optimize the last 3 for log-spacing
    fixed_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    fixed_prod = 1
    for p in fixed_primes:
        fixed_prod *= p

    # Candidates for the last 3 primes: primes from 41 to ~200
    candidates = [p for p in small_primes if p >= 41 and p <= 200]
    
    # Score a 15-prime set by its tau_perp
    # But full evaluation is expensive, so use a fast heuristic first:
    # The number of coprime adjacencies correlates with log-spacing uniformity.
    
    # Try all combinations of 3 from candidates (limited to keep time manageable)
    comb_count = 0
    best_combo_score = 0
    best_combo = None
    
    for combo in combinations(candidates, 3):
        if time.time() - start_time > DEADLINE:
            break
        n = fixed_prod * combo[0] * combo[1] * combo[2]
        if n > LIMIT:
            continue
        factors = [(p, 1) for p in fixed_primes] + [(combo[0], 1), (combo[1], 1), (combo[2], 1)]
        score, _ = tau_perp(n, factors)
        if score > best_score:
            best_score = score
            best_n = n
        comb_count += 1
        if comb_count % 5000 == 0:
            if time.time() - start_time > DEADLINE:
                break

    if time.time() - start_time > DEADLINE:
        return best_n

    # --- Phase 2: Try 14 primes with one exponent-2 (more divisors: 3*2^13 = 24576) ---
    # vs 15 squarefree: 2^15 = 32768 divisors
    # But the coprime density may differ.
    
    primes_14 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]
    n_14_base = 1
    for p in primes_14:
        n_14_base *= p
    
    # Try each prime with exponent 2
    for i, p in enumerate(primes_14):
        if time.time() - start_time > DEADLINE:
            break
        n_test = n_14_base * p  # multiply by p to make exponent 2
        if n_test > LIMIT:
            continue
        factors = [(pp, 2 if pp == p else 1) for pp in primes_14]
        score, _ = tau_perp(n_test, factors)
        if score > best_score:
            best_score = score
            best_n = n_test

    if time.time() - start_time > DEADLINE:
        return best_n

    # --- Phase 3: Try 13 primes with two exponents-2 (4*2^12 = 16384 divisors) ---
    # Or 13 primes with one exponent-3 (4*2^12 = 16384)
    primes_13 = primes_14[:13]
    n_13_base = 1
    for p in primes_13:
        n_13_base *= p

    # Try pairs of primes with exponent 2
    for i in range(len(primes_13)):
        for j in range(i+1, len(primes_13)):
            if time.time() - start_time > DEADLINE:
                break
            n_test = n_13_base * primes_13[i] * primes_13[j]
            if n_test > LIMIT:
                continue
            factors = [(pp, 2 if pp in (primes_13[i], primes_13[j]) else 1) for pp in primes_13]
            score, _ = tau_perp(n_test, factors)
            if score > best_score:
                best_score = score
                best_n = n_test

    if time.time() - start_time > DEADLINE:
        return best_n

    # --- Phase 4: Greedy log-spacing optimization for 15 primes ---
    # Try to find a 15-prime set that maximizes log-spacing uniformity
    # by swapping individual primes.
    
    current_primes = list(primes_15)
    current_prod = 1
    for p in current_primes:
        current_prod *= p
    
    # Try swapping each prime with a nearby larger prime
    improved = True
    while improved:
        improved = False
        for idx in range(len(current_primes)):
            if time.time() - start_time > DEADLINE:
                break
            # Try replacing current_primes[idx] with a larger prime
            old_p = current_primes[idx]
            for new_p in small_primes:
                if new_p <= old_p:
                    continue
                if new_p in current_primes:
                    continue
                new_prod = current_prod // old_p * new_p
                if new_prod > LIMIT:
                    continue
                new_primes = current_primes[:idx] + [new_p] + current_primes[idx+1:]
                new_primes.sort()
                factors = [(p, 1) for p in new_primes]
                score, _ = tau_perp(new_prod, factors)
                if score > best_score:
                    best_score = score
                    best_n = new_prod
                    current_primes = new_primes
                    current_prod = new_prod
                    improved = True
                    break

    if time.time() - start_time > DEADLINE:
        return best_n

    # --- Phase 5: Try 16 primes if possible ---
    # Product of first 16 primes exceeds 10^18, but maybe a different set?
    # 2*3*5*7*11*13*17*19*23*29*31*37*41*43*47*53 > 10^18
    # So 16 distinct primes won't work with the smallest.
    # But what about 15 primes where we can add one more small factor?
    # That would make it non-squarefree.
    
    # Try: 15 primes + one of them squared
    for i, p in enumerate(primes_15):
        if time.time() - start_time > DEADLINE:
            break
        n_test = n_base * p
        if n_test > LIMIT:
            continue
        factors = [(pp, 2 if pp == p else 1) for pp in primes_15]
        score, _ = tau_perp(n_test, factors)
        if score > best_score:
            best_score = score
            best_n = n_test

    if time.time() - start_time > DEADLINE:
        return best_n

    # --- Phase 6: Final polish - try nearby numbers around best_n ---
    # Sometimes a small perturbation helps
    if best_n > 1:
        # Try multiplying/dividing by small primes
        for p in [2, 3, 5, 7, 11, 13]:
            if time.time() - start_time > DEADLINE:
                break
            n_test = best_n * p
            if n_test <= LIMIT:
                factors = None  # let tau_perp factorize
                score, _ = tau_perp(n_test, factors)
                if score > best_score:
                    best_score = score
                    best_n = n_test
            n_test2 = best_n // p if best_n % p == 0 else None
            if n_test2 is not None and n_test2 >= 2:
                factors = None
                score, _ = tau_perp(n_test2, factors)
                if score > best_score:
                    best_score = score
                    best_n = n_test2

    return best_n
