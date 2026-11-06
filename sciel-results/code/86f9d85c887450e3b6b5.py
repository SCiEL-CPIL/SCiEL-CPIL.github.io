import math
import time

def construct():
    start = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill
    
    MAX_N = 10**18
    
    # Precomputed primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]
    
    # Primorials
    primorials = [1]
    for p in primes:
        primorials.append(primorials[-1] * p)
    
    def factorize(n):
        """Factorize n using precomputed primes."""
        factors = []
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                cnt = 0
                while temp % p == 0:
                    temp //= p
                    cnt += 1
                factors.append((p, cnt))
        if temp > 1:
            factors.append((temp, 1))
        return factors
    
    def gen_divisors(factors):
        """Generate all divisors from prime factorization."""
        divs = [1]
        for p, e in factors:
            current = list(divs)
            pw = 1
            for _ in range(e):
                pw *= p
                for d in current:
                    divs.append(d * pw)
        divs.sort()
        return divs
    
    def tau_perp(n):
        """Compute the score: number of adjacent coprime pairs in sorted divisors."""
        if n <= 1:
            return 0
        factors = factorize(n)
        divs = gen_divisors(factors)
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i + 1]) == 1:
                score += 1
        return score
    
    def tau(n):
        """Number of divisors."""
        factors = factorize(n)
        t = 1
        for _, e in factors:
            t *= (e + 1)
        return t
    
    # Candidate evaluation with caching
    eval_cache = {}
    
    def check_candidate(n):
        if n in eval_cache:
            return eval_cache[n]
        if n < 2 or n > MAX_N:
            return 0
        # Quick tau bound: if tau is small, score can't be large
        t = tau(n)
        # Max possible score is at most tau - 1 (all adjacent coprime, which is impossible but upper bound)
        # Actually max score <= tau - 1
        score = tau_perp(n)
        eval_cache[n] = score
        return score
    
    best_score = 0
    best_n = 2
    
    def update_best(n):
        nonlocal best_score, best_n
        if n < 2 or n > MAX_N:
            return
        if time.time() > DEADLINE:
            return
        s = check_candidate(n)
        if s > best_score:
            best_score = s
            best_n = n
    
    # Strategy 1: Primorial-based candidates
    # P_k * m where m is small-prime-smooth
    small_primes = primes[:10]  # 2,3,5,7,11,13,17,19,23,29
    
    def gen_smooth(idx, max_exp, current_m, limit_m):
        """Generate smooth numbers m = prod(p_i^e_i) with e_i <= max_exp, non-increasing."""
        if time.time() > DEADLINE:
            return
        if idx >= len(small_primes):
            update_best(current_m)
            return
        p = small_primes[idx]
        e_max = 0
        temp = 1
        while temp * p <= limit_m and e_max < max_exp:
            temp *= p
            e_max += 1
        for e in range(0, e_max + 1):
            next_m = current_m * (p ** e)
            next_limit = limit_m // (p ** e)
            gen_smooth(idx + 1, e, next_m, next_limit)
            if time.time() > DEADLINE:
                return
    
    # Try primorials up to what fits in 10^18
    for k in range(1, len(primorials)):
        pk = primorials[k]
        if pk > MAX_N:
            break
        limit_m = MAX_N // pk
        if limit_m >= 1:
            gen_smooth(0, 8, 1, limit_m)
            if time.time() > DEADLINE - 30:
                break
    
    if time.time() > DEADLINE:
        return best_n
    
    # Strategy 2: Pure highly-composite-like numbers
    # Numbers of form prod(p_i^e_i) with e_1 >= e_2 >= ... >= e_k >= 1
    def dfs_hc(idx, max_exp, current_n, current_tau):
        if time.time() > DEADLINE:
            return
        if current_n > 1:
            update_best(current_n)
        if idx >= len(primes):
            return
        p = primes[idx]
        e_max = 0
        temp = 1
        limit_n = MAX_N // current_n
        while temp * p <= limit_n and e_max < max_exp:
            temp *= p
            e_max += 1
        for e in range(1, e_max + 1):
            next_n = current_n * (p ** e)
            next_tau = current_tau * (e + 1)
            dfs_hc(idx + 1, e, next_n, next_tau)
            if time.time() > DEADLINE:
                return
    
    dfs_hc(0, 60, 1, 1)
    
    if time.time() > DEADLINE:
        return best_n
    
    # Strategy 3: Local improvement around best candidate
    if best_n > 1:
        factors_best = factorize(best_n)
        
        # Try increasing one exponent
        for i, (p, e) in enumerate(factors_best):
            if best_n * p <= MAX_N:
                update_best(best_n * p)
            if best_n * p * p <= MAX_N:
                update_best(best_n * p * p)
            if best_n * p * p * p <= MAX_N:
                update_best(best_n * p * p * p)
        
        # Try decreasing one exponent
        for i, (p, e) in enumerate(factors_best):
            if e > 1:
                update_best(best_n // p)
                if e > 2:
                    update_best(best_n // (p * p))
        
        # Try swapping a prime for the next one (if exponent allows)
        for i, (p, e) in enumerate(factors_best):
            p_idx = primes.index(p) if p in primes else -1
            if p_idx >= 0 and p_idx + 1 < len(primes):
                next_p = primes[p_idx + 1]
                # Remove p^e, add next_p^e
                new_n = best_n // (p ** e) * (next_p ** e)
                if 2 <= new_n <= MAX_N:
                    update_best(new_n)
                # Also try with e-1 exponent on next prime
                if e > 1:
                    new_n2 = best_n // (p ** e) * (next_p ** (e - 1))
                    if 2 <= new_n2 <= MAX_N:
                        update_best(new_n2)
        
        # Try adding a new prime factor
        for p in primes[:15]:
            if best_n % p != 0 and best_n * p <= MAX_N:
                update_best(best_n * p)
            if best_n % p != 0 and best_n * p * p <= MAX_N:
                update_best(best_n * p * p)
    
    if time.time() > DEADLINE:
        return best_n
    
    # Strategy 4: Randomized local search
    import random
    random.seed(42)
    
    # Build a pool of candidates to explore
    pool = set()
    for k in range(1, min(16, len(primorials))):
        pk = primorials[k]
        if pk <= MAX_N:
            pool.add(pk)
            # Add pk * small multipliers
            for m in range(2, 20):
                if pk * m <= MAX_N:
                    pool.add(pk * m)
    
    # Add highly composite-like numbers
    for _ in range(50):
        # Random exponent vector
        n = 1
        exps = []
        max_e = 12
        for p in primes[:12]:
            e = random.randint(0, max_e)
            # Enforce non-increasing
            if exps and e > exps[-1]:
                e = exps[-1]
            if e > 0 and n * (p ** e) > MAX_N:
                break
            exps.append(e)
            n *= (p ** e)
            if n > MAX_N:
                break
        if n > 1 and n <= MAX_N:
            pool.add(n)
    
    # Evaluate pool and do local search on top ones
    scored_pool = []
    for n in pool:
        if 2 <= n <= MAX_N:
            s = check_candidate(n)
            scored_pool.append((s, n))
    
    scored_pool.sort(key=lambda x: x[0], reverse=True)
    
    # Local search on top 50
    for s, n in scored_pool[:50]:
        if time.time() > DEADLINE - 5:
            break
        factors_n = factorize(n)
        
        # Try single-prime modifications
        for i, (p, e) in enumerate(factors_n):
            # Increase
            if n * p <= MAX_N:
                update_best(n * p)
            # Decrease
            if e > 1:
                update_best(n // p)
            # Replace with next prime
            p_idx = primes.index(p) if p in primes else -1
            if p_idx >= 0 and p_idx + 1 < len(primes):
                next_p = primes[p_idx + 1]
                new_n = n // (p ** e) * (next_p ** e)
                if 2 <= new_n <= MAX_N:
                    update_best(new_n)
        
        # Try adding a new prime
        for p in primes[:20]:
            if n % p != 0 and n * p <= MAX_N:
                update_best(n * p)
    
    return best_n
