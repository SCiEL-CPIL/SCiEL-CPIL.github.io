import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Precompute primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]
    
    def get_divisors_fast(prime_powers):
        """Get sorted divisors from prime factorization as list of (prime, exponent) tuples."""
        divs = [1]
        for p, e in prime_powers:
            if e == 0:
                continue
            current_len = len(divs)
            power = 1
            for _ in range(e):
                power *= p
                for i in range(current_len):
                    divs.append(divs[i] * power)
        divs.sort()
        return divs
    
    def calc_score_from_divisors(divs):
        if len(divs) < 2:
            return 0
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i + 1]) == 1:
                score += 1
        return score
    
    # Strategy: We want to maximize the number of adjacent divisor pairs with gcd 1.
    # Key insight: If n is square-free, divisors are products of subsets of primes.
    # Two consecutive divisors in sorted order often share factors.
    # However, if we use many small primes, the divisors are "spread out" and
    # consecutive ones might differ by a single prime factor, but that doesn't guarantee gcd 1.
    
    # Actually, let's look at the structure.
    # If n = p1 * p2 * ... * pk (square free), the divisors are all products of subsets.
    # Consider n = 2 * 3 * 5 * ... * pk.
    # The divisor 1 is adjacent to the smallest prime. gcd(1, p) = 1. Score +1.
    # The largest divisor is n, adjacent to n/p_min. gcd(n/p_min, n) = p_min != 1. Score 0.
    
    # Let's try to construct n such that many adjacent pairs are coprime.
    # This happens if the divisors are "interleaved" such that they don't share prime factors.
    
    # A known good strategy for maximizing coprime adjacent pairs is to use a number with many distinct prime factors.
    # Let's search for the optimal set of primes and exponents within 10^18.
    
    # We will use a greedy approach:
    # 1. Start with empty factorization.
    # 2. Try adding primes (starting from smallest) with exponent 1, then maybe increase exponents if beneficial.
    # 3. Since the space is large, we will focus on square-free numbers first, as they tend to have more "structure" 
    #    for coprime pairs? Or maybe not.
    
    # Let's evaluate a few candidates directly.
    # Candidate 1: Product of first k primes.
    best_n = 2
    best_score = 0
    
    # Precompute prefix products of primes
    prefix_prod = [1]
    for p in primes:
        prefix_prod.append(prefix_prod[-1] * p)
    
    # Find max k such that product of first k primes <= 10^18
    max_k = 0
    for i in range(1, len(primes) + 1):
        if prefix_prod[i] <= 10**18:
            max_k = i
        else:
            break
    
    # Evaluate square-free products of first k primes for k in a range around max_k
    # For square-free product of first k primes, we know k directly from the loop variable
    
    def evaluate_factorization(powers):
        """powers: list of (prime, exponent). Returns (score, n) or (-1, -1) if invalid."""
        n = 1
        for p, e in powers:
            if e > 0:
                n *= (p ** e)
        if n > 10**18 or n < 2:
            return -1, -1
        divs = get_divisors_fast(powers)
        s = calc_score_from_divisors(divs)
        return s, n
    
    # 1. Square-free products of first k primes - evaluate directly with known k
    for k in range(1, max_k + 1):
        if time.time() > deadline:
            break
        pows = [(primes[i], 1) for i in range(k)]
        s, n = evaluate_factorization(pows)
        if s > best_score and n > 0:
            best_score = s
            best_n = n
    
    # 2. Try variations: increase exponents of small primes, possibly drop large primes
    # For each number of distinct primes k, try various exponent patterns
    for k in range(max(1, max_k - 6), max_k + 1):
        if time.time() > deadline:
            break
        # Base: all exponents 1
        base_pows = [(primes[i], 1) for i in range(k)]
        
        # Try increasing exponent of prime i by 1, 2, or 3, dropping largest primes if needed
        for i in range(min(4, k)):
            for extra in range(1, 4):
                # Start with all exponent 1, then add 'extra' to prime i
                pows = [(primes[j], 1) for j in range(k)]
                pows[i] = (primes[i], 1 + extra)
                
                # Check if it fits
                n_val = 1
                for p, e in pows:
                    n_val *= (p ** e)
                
                if n_val <= 10**18:
                    s, n = evaluate_factorization(pows)
                    if s > best_score and n > 0:
                        best_score = s
                        best_n = n
                else:
                    # Try dropping largest primes until it fits
                    # Drop from the end (largest primes)
                    for drop_start in range(k - 1, i, -1):
                        # Try dropping primes from drop_start to k-1 (some subset)
                        # Simple: just drop the single largest prime
                        if drop_start >= k:
                            continue
                        new_pows = pows[:drop_start]  # Keep primes 0..drop_start-1
                        # But we need to make sure prime i is still included
                        if i >= drop_start:
                            continue
                        n_val = 1
                        for p, e in new_pows:
                            n_val *= (p ** e)
                        if n_val <= 10**18 and n_val >= 2:
                            s, n = evaluate_factorization(new_pows)
                            if s > best_score and n > 0:
                                best_score = s
                                best_n = n
                        break  # Only try one drop level per (i, extra)
    
    # 3. Try "almost square-free": drop one large prime, increase exponent of 2
    for k in range(max(2, max_k - 4), max_k + 1):
        if time.time() > deadline:
            break
        # Use primes 0 to k-2 (drop the largest prime p_{k-1})
        for a in range(1, 15):
            pows = [(primes[0], a)]
            for i in range(1, k - 1):
                pows.append((primes[i], 1))
            s, n = evaluate_factorization(pows)
            if s > best_score and n > 0:
                best_score = s
                best_n = n
    
    # 4. Try dropping two large primes, increase exponents of 2 and 3
    for k in range(max(3, max_k - 4), max_k + 1):
        if time.time() > deadline:
            break
        for a in range(1, 10):
            for b in range(1, 5):
                pows = [(primes[0], a), (primes[1], b)]
                for i in range(2, k - 2):
                    pows.append((primes[i], 1))
                s, n = evaluate_factorization(pows)
                if s > best_score and n > 0:
                    best_score = s
                    best_n = n
    
    # Final check: ensure best_n is valid
    if best_n < 2:
        best_n = 2
        best_score = 1  # divisors of 2: [1, 2], gcd(1,2)=1, score=1

    return best_n
