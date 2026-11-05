import math
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    # Primes up to 97
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

    def get_divisors(n):
        divs = [1]
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                new_divs = []
                power = 1
                while temp % p == 0:
                    temp //= p
                    power *= p
                    for d in divs:
                        new_divs.append(d * power)
                divs += new_divs
            if temp == 1:
                break
        if temp > 1:
            for d in divs[:]:
                divs.append(d * temp)
        return sorted(divs)

    def calc_score(n):
        if n < 2:
            return 0
        divs = get_divisors(n)
        if len(divs) < 2:
            return 0
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i + 1]) == 1:
                score += 1
        return score

    def make_n(exponents):
        """exponents is a list of (prime_index, exponent) pairs."""
        n = 1
        for idx, exp in exponents:
            n *= primes[idx] ** exp
        return n

    candidates = set()

    # --- Baseline: Square-free product of first 15 primes ---
    # 2*3*5*...*47
    n_sq15 = 1
    for p in primes[:15]:
        n_sq15 *= p
    if n_sq15 <= 10**18:
        candidates.add(n_sq15)

    # --- Systematic Search over Exponent Vectors ---
    # We focus on the first K primes.
    # For a fixed K, we search for exponents e_0, e_1, ..., e_{K-1}
    # such that prod(p_i^e_i) <= 10^18.
    # The score is roughly proportional to the number of coprime transitions.
    # For square-free, it's high. Adding powers reduces the "spread" but increases tau.
    # The optimal structure is often to put extra weight on small primes (2, 3)
    # because they appear in more divisors and their powers create finer granularity,
    # but we must balance the size constraint.

    # Let's define a search space for exponents of the first few primes.
    # Primes involved: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47
    
    # We will iterate over possible "shapes" of the exponent vector.
    # A shape is defined by which primes have exponent > 1.
    # Given the limit 10^18, we can't have many primes with high exponents.
    
    # Strategy:
    # 1. Try square-free for K=14, 15.
    # 2. Try variations where 2 and/or 3 have higher exponents for K=13, 14, 15.
    # 3. Try variations where 2 has very high exponent for smaller K.

    # Helper to add candidate if valid
    def add_candidate(n):
        if 2 <= n <= 10**18:
            candidates.add(n)

    # --- Case 1: K=15 (primes 2..47) ---
    # Base: 2^1 3^1 ... 47^1 (Square free)
    add_candidate(n_sq15)
    
    # We have ~10^18 / n_sq15 ~ 10^18 / 6.1e17 ~ 1.6 multiplier available? 
    # Actually n_sq15 = 614889782588491410.
    # 10^18 / 6.14e17 ~ 1.62.
    # So we can multiply by 2^1 (total 2^2) -> 2.24e18 (Too big)
    # Wait, 6.14 * 2 = 12.28 > 10. So 2^2 * 3^1 * ... is too big.
    # So for K=15, only square-free is possible?
    # Let's check 2^2 * 3^1 * 5^1 ... 
    # n_sq15 * 2 = 1.22e18 > 1e18.
    # So NO extra powers for K=15.
    
    # --- Case 2: K=14 (primes 2..43) ---
    # Base: 2^1 3^1 ... 43^1
    n_sq14 = 1
    for p in primes[:14]:
        n_sq14 *= p
    # n_sq14 = 13082761331670030
    # Multiplier available: 10^18 / 1.3e16 ~ 76.
    # We can distribute this multiplier among 2, 3, 5...
    # Let e2, e3, e5... be the *extra* exponents added to the base 1.
    # So total exp of 2 is 1+e2, etc.
    # Product of (p_i^(1+e_i)) <= 10^18.
    # Equivalent to: n_sq14 * (2^e2 * 3^e3 * 5^e5 * ...) <= 10^18.
    # Let M = 10^18 // n_sq14.
    M14 = 10**18 // n_sq14
    # Search for e2, e3, e5, e7 such that 2^e2 * 3^e3 * 5^e5 * 7^e7 <= M14
    # Since M14 is ~76, e2 can be up to 6 (2^6=64), e3 up to 4 (3^4=81>76, so 3^3=27), etc.
    for e2 in range(0, 7):
        if 2**e2 > M14: break
        rem2 = M14 // (2**e2)
        for e3 in range(0, 5):
            if 3**e3 > rem2: break
            rem3 = rem2 // (3**e3)
            for e5 in range(0, 4):
                if 5**e5 > rem3: break
                rem4 = rem3 // (5**e5)
                for e7 in range(0, 3):
                    if 7**e7 > rem4: break
                    # Construct n
                    # Base exponents are all 1.
                    # Extra exponents: 2->e2, 3->e3, 5->e5, 7->e7
                    exps = []
                    for i in range(14):
                        if i == 0: exps.append((0, 1 + e2))
                        elif i == 1: exps.append((1, 1 + e3))
                        elif i == 2: exps.append((2, 1 + e5))
                        elif i == 3: exps.append((3, 1 + e7))
                        else: exps.append((i, 1))
                    n = make_n(exps)
                    add_candidate(n)

    # --- Case 3: K=13 (primes 2..41) ---
    n_sq13 = 1
    for p in primes[:13]:
        n_sq13 *= p
    M13 = 10**18 // n_sq13
    # n_sq13 = 304250263527210. M13 ~ 3286.
    # Search e2, e3, e5, e7, e11
    for e2 in range(0, 12):
        if 2**e2 > M13: break
        rem2 = M13 // (2**e2)
        for e3 in range(0, 8):
            if 3**e3 > rem2: break
            rem3 = rem2 // (3**e3)
            for e5 in range(0, 6):
                if 5**e5 > rem3: break
                rem4 = rem3 // (5**e5)
                for e7 in range(0, 5):
                    if 7**e7 > rem4: break
                    rem5 = rem4 // (7**e7)
                    for e11 in range(0, 4):
                        if 11**e11 > rem5: break
                        exps = []
                        for i in range(13):
                            if i == 0: exps.append((0, 1 + e2))
                            elif i == 1: exps.append((1, 1 + e3))
                            elif i == 2: exps.append((2, 1 + e5))
                            elif i == 3: exps.append((3, 1 + e7))
                            elif i == 4: exps.append((4, 1 + e11))
                            else: exps.append((i, 1))
                        n = make_n(exps)
                        add_candidate(n)
                        if time.time() > deadline:
                            return candidates.pop() if candidates else 2

    # --- Case 4: K=12 (primes 2..37) ---
    n_sq12 = 1
    for p in primes[:12]:
        n_sq12 *= p
    M12 = 10**18 // n_sq12
    # n_sq12 = 7420738134810. M12 ~ 134756.
    # Search e2, e3, e5, e7, e11, e13
    for e2 in range(0, 18):
        if 2**e2 > M12: break
        rem2 = M12 // (2**e2)
        for e3 in range(0, 12):
            if 3**e3 > rem2: break
            rem3 = rem2 // (3**e3)
            for e5 in range(0, 10):
                if 5**e5 > rem3: break
                rem4 = rem3 // (5**e5)
                for e7 in range(0, 8):
                    if 7**e7 > rem4: break
                    rem5 = rem4 // (7**e7)
                    for e11 in range(0, 7):
                        if 11**e11 > rem5: break
                        rem6 = rem5 // (11**e11)
                        for e13 in range(0, 6):
                            if 13**e13 > rem6: break
                            exps = []
                            for i in range(12):
                                if i == 0: exps.append((0, 1 + e2))
                                elif i == 1: exps.append((1, 1 + e3))
                                elif i == 2: exps.append((2, 1 + e5))
                                elif i == 3: exps.append((3, 1 + e7))
                                elif i == 4: exps.append((4, 1 + e11))
                                elif i == 5: exps.append((5, 1 + e13))
                                else: exps.append((i, 1))
                            n = make_n(exps)
                            add_candidate(n)
                            if time.time() > deadline:
                                return candidates.pop() if candidates else 2

    # --- Case 5: K=11 (primes 2..31) ---
    n_sq11 = 1
    for p in primes[:11]:
        n_sq11 *= p
    M11 = 10**18 // n_sq11
    # n_sq11 = 20086291530. M11 ~ 49785771.
    # This space is larger. We might miss some deep combos if we go too deep.
    # Let's limit exponents slightly to keep it fast.
    for e2 in range(0, 26):
        if 2**e2 > M11: break
        rem2 = M11 // (2**e2)
        for e3 in range(0, 16):
            if 3**e3 > rem2: break
            rem3 = rem2 // (3**e3)
            for e5 in range(0, 13):
                if 5**e5 > rem3: break
                rem4 = rem3 // (5**e5)
                for e7 in range(0, 11):
                    if 7**e7 > rem4: break
                    rem5 = rem4 // (7**e7)
                    for e11 in range(0, 10):
                        if 11**e11 > rem5: break
                        rem6 = rem5 // (11**e11)
                        for e13 in range(0, 9):
                            if 13**e13 > rem6: break
                            rem7 = rem6 // (13**e13)
                            for e17 in range(0, 8):
                                if 17**e17 > rem7: break
                                exps = []
                                for i in range(11):
                                    if i == 0: exps.append((0, 1 + e2))
                                    elif i == 1: exps.append((1, 1 + e3))
                                    elif i == 2: exps.append((2, 1 + e5))
                                    elif i == 3: exps.append((3, 1 + e7))
                                    elif i == 4: exps.append((4, 1 + e11))
                                    elif i == 5: exps.append((5, 1 + e13))
                                    elif i == 6: exps.append((6, 1 + e17))
                                    else: exps.append((i, 1))
                                n = make_n(exps)
                                add_candidate(n)
                                if time.time() > deadline:
                                    return candidates.pop() if candidates else 2

    # --- Evaluate Candidates ---
    # To avoid timeouts, we prioritize candidates with higher tau(n) or known good structures.
    # But simply evaluating all might be slow if there are thousands.
    # Let's sort by n (smaller n often allows more primes, but not necessarily).
    # Actually, let's just evaluate them. The number of candidates generated above is manageable.
    # K=15: 1
    # K=14: ~ 7*5*4*3 = 420
    # K=13: ~ 12*8*6*5*4 = 11520
    # K=12: ~ 18*12*10*8*7*6 = 725760 (Too many to evaluate all with full divisor generation)
    # K=11: Even more.
    
    # We need to be selective.
    # The score is correlated with the number of divisors, but penalized if divisors are "clumped".
    # Square-free numbers have the most "spread".
    # High powers of 2 make divisors very dense at the low end.
    # Let's filter candidates: 
    # 1. Always keep square-free ones.
    # 2. For non-square-free, keep only those with "moderate" exponents?
    # Or, we can estimate the score without full divisor generation?
    # No, we need the exact score.
    
    # Strategy: 
    # 1. Evaluate all K=15, K=14 candidates.
    # 2. For K=13, K=12, K=11, we have too many. 
    #    We will sample or prioritize.
    #    Prioritize: Lower total exponent sum? Or higher tau?
    #    Actually, let's just evaluate a random subset or the ones with highest tau?
    #    Calculating tau is fast.
    
    def get_tau(n, primes_list):
        tau = 1
        temp = n
        for p in primes_list:
            if p * p > temp:
                break
            if temp % p == 0:
                cnt = 0
                while temp % p == 0:
                    temp //= p
                    cnt += 1
                tau *= (cnt + 1)
        if temp > 1:
            tau *= 2
        return tau

    # Pre-filter: Calculate tau for all candidates.
    # Sort by tau descending.
    # Evaluate top N candidates.
    
    # Re-generate candidates into a list to allow sorting
    candidate_list = list(candidates)
    
    # If list is huge, we might want to prune.
    # Let's assume the generation above didn't blow up too much.
    # K=12 loop: 18*12*10*8*7*6 is 725k. That's a lot to store and sort.
    # Let's refine the K=12 and K=11 loops to be less exhaustive or add pruning during generation.
    
    # Actually, the previous code added to a set. 
    # Let's just take the top 1000 by tau for evaluation to be safe on time.
    
    if len(candidate_list) > 5000:
        # Calculate tau for all and keep top 5000
        scored_candidates = []
        for n in candidate_list:
            if time.time() > deadline - 10:
                break
            t = get_tau(n, primes)
            scored_candidates.append((t, n))
        scored_candidates.sort(reverse=True)
        candidate_list = [n for t, n in scored_candidates[:5000]]
    else:
        # Calculate tau for sorting if smaller
        scored_candidates = []
        for n in candidate_list:
            t = get_tau(n, primes)
            scored_candidates.append((t, n))
        scored_candidates.sort(reverse=True)
        candidate_list = [n for t, n in scored_candidates]

    best_n = 2
    best_score = 0
    
    # Evaluate candidates
    for n in candidate_list:
        if time.time() > deadline:
            break
        s = calc_score(n)
        if s > best_score:
            best_score = s
            best_n = n

    return best_n
