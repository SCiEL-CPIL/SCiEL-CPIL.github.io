import math
import time
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s limit to be safe
    
    # Helper: sigma(n) for small n
    def sigma(n):
        res = 1
        i = 2
        while i * i <= n:
            if n % i == 0:
                while n % i == 0:
                    n //= i
                res *= (i * (i + 1) // 2)
            i += 1
        if n > 1:
            res *= (n + 1)
        return res

    # Helper: Check if k is valid for M
    def check(k, M):
        if k * sigma(k) == M:
            return True
        return False

    # Generate candidate blocks g(p, e)
    # We want blocks that are "small" to allow many of them in M <= 10^18
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    blocks = []
    for p in primes:
        for e in range(1, 6):
            val = (p ** e) * sigma(p ** e)
            if val < 10**6: # Keep blocks small
                blocks.append((val, p, e))
    
    # Sort blocks by value
    blocks.sort()
    
    best_M = 333312
    best_k = [336, 372, 434]
    best_score = 3
    
    # Strategy: Pick a set of blocks, compute M, then count solutions.
    # To keep it fast, we limit the number of blocks to ~10-12.
    
    # Pre-compute prime factorization for M counting
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

    def count_solutions(M, primes_subset):
        # Count k such that k*sigma(k)=M and k is composed of primes in primes_subset
        # This is an approximation. We will verify candidates later.
        # For now, we just return the number of blocks we used as a heuristic score
        # and verify the top few.
        pass

    # We will try to construct M by multiplying blocks.
    # We want to maximize the number of distinct k.
    # A good heuristic is to use blocks that share factors.
    
    # Let's try a specific high-collision candidate found in literature or by search.
    # Known: M = 2^a 3^b 5^c ...
    
    # We'll do a limited search over combinations of the first 10 smallest blocks.
    small_blocks = blocks[:20]
    
    # To avoid exponential blowup, we fix the set of primes P and vary exponents.
    # Let's try P = {2, 3, 5, 7, 11, 13}
    P = [2, 3, 5, 7, 11, 13]
    
    # We want to find exponents e_p for p in P such that M = prod g(p, e_p) <= 10^18
    # and the number of solutions is high.
    
    # Since we can't easily count solutions, we will generate a few candidates M
    # and for each, find all k <= sqrt(M) such that k*sigma(k)=M.
    # To speed up k search, we note that k must be composed of primes in P (mostly).
    
    candidates = []
    
    # Generate candidates by varying exponents of 2 and 3, and setting others to 1 or 2.
    for e2 in range(1, 8):
        for e3 in range(1, 6):
            for e5 in range(1, 4):
                for e7 in range(1, 4):
                    for e11 in range(1, 3):
                        for e13 in range(1, 3):
                            M = 1
                            for p, e in zip(P, [e2, e3, e5, e7, e11, e13]):
                                M *= (p ** e) * sigma(p ** e)
                            if M <= 10**18:
                                candidates.append((M, [e2, e3, e5, e7, e11, e13]))
                            if M > 10**18:
                                break
                    if M > 10**18: break
                if M > 10**18: break
            if M > 10**18: break
        if M > 10**18: break
        if time.time() > deadline - 20: break
    
    # Evaluate candidates
    # To find solutions k, we can iterate k up to sqrt(M) is too slow.
    # Instead, we generate k from the prime factors of M.
    # k must divide M? No. k*sigma(k)=M. k can have primes not in M?
    # If p|k, then p|M? No. sigma(k) has p+1.
    # However, for small primes, k is likely composed of small primes.
    
    # We will check the top 100 candidates by M value (larger M might have more solutions? No, complexity).
    # Let's just pick the one with the most "blocks" in the construction.
    
    # Actually, let's just return the seed if we don't find a better one quickly.
    # But we can try one specific known good M.
    # M = 333312 has 3 solutions.
    
    # Let's try M = 2^4 * 3 * 7 * 31 ... no.
    
    # A better approach:
    # Use the fact that if M = prod g(p_i, e_i), then k = prod p_i^e_i is a solution.
    # Other solutions come from regrouping.
    
    # Let's try to find M with > 3 solutions by checking a few random combinations.
    
    import random
    random.seed(42)
    
    found = []
    for _ in range(1000):
        if time.time() > deadline - 10:
            break
        # Random exponents for P
        exps = [random.randint(1, 5) for _ in P]
        M = 1
        for p, e in zip(P, exps):
            M *= (p ** e) * sigma(p ** e)
        if M <= 10**18:
            # Find solutions
            # k must be <= sqrt(M)
            # We can generate divisors of M? No.
            # We can iterate k up to min(10000, isqrt(M))?
            # If M is large, sqrt(M) is large.
            # But k is composed of small primes.
            # We can generate all k composed of P with exponents <= max(exps)+2.
            
            max_exp = max(exps) + 2
            def gen_k(idx, current_k, current_sigma_prod):
                if idx == len(P):
                    if current_k * current_sigma_prod == M:
                        if current_k <= math.isqrt(M):
                            found.append(current_k)
                    return
                p = P[idx]
                for e in range(0, max_exp + 1):
                    term_k = p ** e
                    term_sigma = sigma(p ** e)
                    # Pruning: if current_k * term_k > isqrt(M), break?
                    # Not necessarily, but k <= sqrt(M).
                    if current_k * term_k > math.isqrt(M):
                        break
                    gen_k(idx + 1, current_k * term_k, current_sigma_prod * term_sigma)
            
            found.clear()
            gen_k(0, 1, 1)
            if len(found) > best_score:
                best_score = len(found)
                best_M = M
                best_k = found
                print(f"New best: M={M}, score={best_score}, k={best_k}")
    
    if best_score > 3:
        return {'M': best_M, 'k': best_k}
    else:
        return {'M': 333312, 'k': [336, 372, 434]}

# EVOLVE-BLOCK-END
