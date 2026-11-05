import math
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s margin for safety
    
    max_n = 10**18
    
    # The key insight: to maximize the number of coprime adjacent pairs in the divisor sequence,
    # we want divisors that differ by a single prime factor (or small prime factors) where possible.
    # 
    # A number n = p1^a1 * p2^a2 * ... * pk^ak has (a1+1)(a2+1)...(ak+1) divisors.
    # The divisors sorted in increasing order form a sequence. Two adjacent divisors d_i, d_{i+1}
    # are coprime if and only if they share no common prime factor.
    #
    # Key structural insight: if n is squarefree (all ai = 1), then the divisors are all products
    # of distinct primes. Two adjacent divisors in sorted order... actually for squarefree numbers,
    # the divisor lattice is a Boolean lattice. In the sorted order, consecutive divisors often share
    # many prime factors.
    #
    # Let me think differently. Consider n = 2^a * 3^b * 5^c * ...
    # The divisors sorted: 1, 2, 3, 4, 5, 6, ...
    # gcd(1,2)=1, gcd(2,3)=1, gcd(3,4)=1, gcd(4,5)=1, gcd(5,6)=1, gcd(6,7)=1...
    # Wait, for n=6: divisors are 1,2,3,6. Pairs: (1,2)gcd=1, (2,3)gcd=1, (3,6)gcd=3. Score=2.
    # For n=30: divisors 1,2,3,5,6,10,15,30. 
    # (1,2)=1, (2,3)=1, (3,5)=1, (5,6)=1, (6,10)=2, (10,15)=5, (15,30)=15. Score=4.
    # For n=210: divisors 1,2,3,5,6,7,10,14,15,21,30,35,42,70,105,210
    # (1,2)=1,(2,3)=1,(3,5)=1,(5,6)=1,(6,7)=1,(7,10)=1,(10,14)=2,(14,15)=1,(15,21)=3,(21,30)=3,
    # (30,35)=5,(35,42)=7,(42,70)=14,(70,105)=35,(105,210)=105. Score=7.
    #
    # Pattern: for n = product of first k primes (squarefree), the score seems to be k+... let me check.
    # k=1: n=2, divisors 1,2. Score=1.
    # k=2: n=6, divisors 1,2,3,6. Score=2.
    # k=3: n=30, divisors 1,2,3,5,6,10,15,30. Score=4.
    # k=4: n=210, divisors 1,2,3,5,6,7,10,14,15,21,30,35,42,70,105,210. Score=7.
    # 
    # The score for squarefree n with k primes appears to follow a pattern. Let me think about it.
    # In the sorted divisor list of a squarefree number, each divisor is a subset of primes.
    # The sorted order is by numerical value. Two consecutive divisors are coprime iff they are
    # disjoint subsets.
    #
    # Actually, let me just compute for various structures and find what works best.
    
    # Strategy: enumerate candidate numbers by their prime factorization, compute the score,
    # and find the maximum.
    
    # First, let's think about what structure maximizes the score.
    # The total number of divisors is tau(n). The score is at most tau(n) - 1.
    # To maximize score, we want many divisors AND many coprime adjacent pairs.
    
    # Let me try a different approach: for each candidate n, compute the score efficiently.
    # Generate divisors, sort them, count coprime adjacent pairs.
    
    # The challenge is finding which n to try. Let me use a smart search.
    
    # Key observation: numbers with many small prime factors tend to have more coprime adjacent pairs
    # because the divisors are spread out in a way that consecutive ones often differ by a single prime.
    
    # Let me try: n = 2^a * 3^b * 5^c * 7^d * 11^e * ...
    # with a >= b >= c >= d >= ... (non-increasing exponents for HCN-like numbers)
    
    # But actually, for maximizing coprime pairs, we might want exponents to be small (1 or 2)
    # to keep the divisor structure "spread out".
    
    # Let me do a focused search over exponent vectors.
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    def compute_score(n):
        """Compute tau_perp(n): number of adjacent coprime pairs in sorted divisors."""
        if n <= 1:
            return 0
        divisors = get_divisors(n)
        divisors.sort()
        score = 0
        for i in range(len(divisors) - 1):
            if math.gcd(divisors[i], divisors[i+1]) == 1:
                score += 1
        return score
    
    def get_divisors(n):
        """Generate all divisors of n."""
        divs = [1]
        temp = n
        p = 2
        while p * p <= temp:
            if temp % p == 0:
                power = 1
                while temp % p == 0:
                    temp //= p
                    power *= p
                    new_divs = []
                    for d in divs:
                        new_divs.append(d * power)
                    divs.extend(new_divs)
            p += 1 if p == 2 else 2
        if temp > 1:
            new_divs = []
            for d in divs:
                new_divs.append(d * temp)
            divs.extend(new_divs)
        return divs
    
    best_score = 0
    best_n = 2
    
    # Phase 1: Try squarefree numbers (product of first k primes)
    # Product of first k primes <= 10^18
    # k=15: 2*3*5*7*11*13*17*19*23*29*31*37*41*43*47 = 614889782588491410
    # k=16: *53 = 32589158477190044730 > 10^18
    # So max squarefree with distinct small primes is k=15
    
    product = 1
    for k in range(1, 16):
        product *= primes[k-1]
        if product > max_n:
            break
        score = compute_score(product)
        if score > best_score:
            best_score = score
            best_n = product
        if time.time() > deadline:
            return best_n
    
    # Phase 2: Try numbers with some exponents > 1
    # The idea: 2^2 * 3 * 5 * 7 * ... might give more divisors and more coprime pairs
    # Let's search over exponent patterns
    
    # For a number n = 2^a * 3^b * 5^c * ... with a >= b >= c >= ...
    # The number of divisors is (a+1)(b+1)(c+1)...
    # We want this to be large while keeping n <= 10^18
    
    # Let me do a DFS over exponent vectors with non-increasing exponents
    
    def search_exponents(idx, current_n, current_exp, max_exp):
        """DFS over exponent vectors with non-increasing exponents."""
        nonlocal best_score, best_n
        if time.time() > deadline:
            return
        if idx >= len(primes):
            if current_n > 1:
                score = compute_score(current_n)
                if score > best_score:
                    best_score = score
                    best_n = current_n
            return
        
        p = primes[idx]
        # Try exponents from max_exp down to 1 (or 0 to stop)
        for e in range(max_exp, -1, -1):
            if e > 0:
                new_n = current_n * (p ** e)
                if new_n > max_n:
                    continue
            else:
                new_n = current_n
            
            if e == 0:
                # No more primes, evaluate current_n
                if current_n > 1:
                    score = compute_score(current_n)
                    if score > best_score:
                        best_score = score
                        best_n = current_n
                return
            
            search_exponents(idx + 1, new_n, e, e)
    
    # The DFS above might be too slow for deep searches. Let me limit the depth.
    # Actually, let me be smarter. The number of divisors tau(n) gives an upper bound on score.
    # For n <= 10^18, the maximum tau(n) is about 103680 (for highly composite numbers).
    # But the score is at most tau(n) - 1, so we need tau(n) to be large.
    
    # However, not all high-tau numbers have high coprime-pair scores.
    # Let me focus on numbers with many distinct prime factors (small exponents).
    
    # Phase 3: Try numbers of the form 2^a * 3^b * 5^c * 7^d * 11^e * 13^f * ...
    # with small exponents (1 or 2)
    
    # Let me enumerate combinations of exponents for the first few primes
    # where each exponent is 0, 1, or 2
    
    # For the first 10 primes, if all exponents are 1 or 2:
    # 2^2 * 3^2 * 5^2 * 7^2 * 11^2 * 13^2 * 17^2 * 19^2 * 23^2 * 29^2 is way too big
    # Let me be more careful.
    
    # Actually, let me just do a targeted search.
    # The best candidates likely have the form: product of first k primes, possibly with
    # some small exponents on the smallest primes.
    
    # Let me try: for each k from 5 to 15 (number of distinct primes), try all combinations
    # of exponents (1 or 2) for the first few primes, keeping the product <= 10^18.
    
    # This is still a large search space. Let me use a greedy + local search approach.
    
    # Phase 4: Local search from the best squarefree number
    # Start with the best squarefree number, then try multiplying by small primes (increasing exponents)
    # and see if the score improves.
    
    # Actually, let me just do a more systematic DFS but with a time limit.
    # The key is to explore the space of exponent vectors efficiently.
    
    # Let me reset and do a cleaner search.
    # I'll enumerate numbers by their prime factorization, focusing on the first 12 primes,
    # with exponents constrained to keep n <= 10^18.
    
    # For efficiency, I'll use the fact that if n = p1^a1 * ... * pk^ak, then
    # the divisors can be generated and scored.
    
    # Let me try a different structural approach:
    # For n with prime factorization, the sorted divisors form a sequence.
    # Two consecutive divisors d_i and d_{i+1} are coprime iff they are "adjacent" in the
    # sense that d_{i+1}/d_i is a prime power and d_i is not divisible by that prime.
    
    # Actually, in the sorted divisor list, consecutive divisors don't necessarily have
    # a simple ratio. But for numbers with many small prime factors, the divisors are
    # densely packed and consecutive ones often differ by a single small prime.
    
    # Let me just do a brute-force search over a carefully chosen set of candidates.
    
    # Candidate set:
    # 1. Product of first k primes (squarefree) for k = 1 to 15
    # 2. For k = 8 to 15, try replacing the largest prime with a power of 2
    #    e.g., 2^2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 instead of 2 * 3 * 5 * ... * 23
    # 3. Try 2^a * 3^b * 5^c * 7^d * 11^e * 13^f * 17^g * 19^h with small exponents
    
    # Let me implement a focused search.
    
    # First, let me compute scores for the squarefree numbers to establish a baseline.
    # (Already done in Phase 1)
    
    # Now, let me try numbers with one prime having exponent 2.
    # For each squarefree number n = p1 * p2 * ... * pk, try n * pi for each i.
    # This gives numbers with one prime having exponent 2.
    
    product = 1
    for k in range(1, 16):
        product *= primes[k-1]
        if product > max_n:
            break
        # Try multiplying by each of the first k primes
        for i in range(k):
            candidate = product * primes[i]
            if candidate > max_n:
                continue
            if time.time() > deadline:
                return best_n
            score = compute_score(candidate)
            if score > best_score:
                best_score = score
                best_n = candidate
    
    # Try numbers with two primes having exponent 2.
    # For each squarefree number, try multiplying by pi * pj for i < j.
    product = 1
    for k in range(1, 16):
        product *= primes[k-1]
        if product > max_n:
            break
        for i in range(k):
            for j in range(i+1, k):
                candidate = product * primes[i] * primes[j]
                if candidate > max_n:
                    continue
                if time.time() > deadline:
                    return best_n
                score = compute_score(candidate)
                if score > best_score:
                    best_score = score
                    best_n = candidate
    
    # Try 2^3 * 3 * 5 * 7 * ...
    product = 1
    for k in range(1, 16):
        product *= primes[k-1]
        if product > max_n:
            break
        # Try 2^3 instead of 2^1
        candidate = product * 4  # multiply by 2^2 to get 2^3
        if candidate <= max_n:
            if time.time() > deadline:
                return best_n
            score = compute_score(candidate)
            if score > best_score:
                best_score = score
                best_n = candidate
    
    # Try 2^2 * 3^2 * 5 * 7 * ...
    product = 1
    for k in range(1, 16):
        product *= primes[k-1]
        if product > max_n:
            break
        candidate = product * 2 * 3  # multiply by 2*3 to get 2^2 * 3^2
        if candidate <= max_n:
            if time.time() > deadline:
                return best_n
            score = compute_score(candidate)
            if score > best_score:
                best_score = score
                best_n = candidate
    
    # Let me also try a more systematic approach:
    # For the first 10 primes, try all combinations of exponents from {0, 1, 2}
    # but prune when the product exceeds 10^18.
    
    # This is 3^10 = 59049 combinations, which is feasible.
    # But I need to be careful about the product size.
    
    # Let me do a DFS over the first 12 primes with exponents 0, 1, or 2.
    
    def search_small_exponents(idx, current_n):
        """Search over exponents 0, 1, 2 for the first 12 primes."""
        nonlocal best_score, best_n
        if time.time() > deadline:
            return
        if idx >= 12:  # Only first 12 primes
            if current_n > 1:
                score = compute_score(current_n)
                if score > best_score:
                    best_score = score
                    best_n = current_n
            return
        
        p = primes[idx]
        for e in range(3):  # 0, 1, 2
            new_n = current_n * (p ** e)
            if new_n > max_n:
                break
            search_small_exponents(idx + 1, new_n)
    
    search_small_exponents(0, 1)
    
    # Phase 5: Try a few more specific structures
    # Numbers like 2^a * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41
    # with a = 1, 2, 3, 4
    
    base = 1
    for k in range(1, 15):
        base *= primes[k]  # 3 * 5 * 7 * ... * p_k
    # base = 3 * 5 * 7 * ... * p_k (excluding 2)
    # Now try 2^a * base for a = 1, 2, 3, 4, 5
    
    # Actually, let me just try a few more targeted candidates.
    
    # Let me try n = 2^a * 3^b * 5^c * 7^d * 11^e * 13^f * 17^g * 19^h * 23^i * 29^j * 31^k * 37^l
    # with a >= b >= c >= ... and small exponents.
    
    # I'll do a DFS with max exponent 4 for the first 8 primes, and exponent 1 for the rest.
    
    # Actually, let me just return the best found so far.
    # The search above should have covered the most promising candidates.
    
    return best_n
