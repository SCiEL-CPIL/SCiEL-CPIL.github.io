import math
import time
import random
from functools import lru_cache

def construct():
    start_time = time.time()
    DEADLINE = 115.0
    LIMIT = 10**18

    # Precompute primes up to a reasonable bound
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i in range(2, n+1) if is_prime[i]]

    all_primes = sieve(2000)
    prime_to_idx = {p: i for i, p in enumerate(all_primes)}

    def fast_tau_perp_from_factors(factors):
        """
        factors: dict {prime: exponent}
        Compute tau_perp by generating divisors and checking adjacency.
        Optimized: use set for fast gcd checks.
        """
        if not factors:
            return 0
        
        # Generate all divisors
        divs = [1]
        for p, e in factors.items():
            new_divs = []
            p_pow = 1
            for _ in range(e + 1):
                for d in divs:
                    new_divs.append(d * p_pow)
                p_pow *= p
            divs = new_divs
        
        if len(divs) < 2:
            return 0
        
        divs.sort()
        
        # For each adjacent pair, check gcd == 1
        # Optimization: precompute prime factor sets for each divisor
        # But that's expensive. Instead, use a trick:
        # Two numbers are coprime iff they share no prime factor.
        # For squarefree case, we can use bitmasks.
        
        # For general case, just compute gcd for adjacent pairs
        score = 0
        prev = divs[0]
        for i in range(1, len(divs)):
            curr = divs[i]
            if math.gcd(prev, curr) == 1:
                score += 1
            prev = curr
        return score

    def fast_tau_perp_squarefree(prime_list):
        """
        For squarefree n with given prime list, compute tau_perp.
        Uses bitmask representation for speed.
        """
        k = len(prime_list)
        if k == 0:
            return 0
        
        # Each divisor corresponds to a subset (bitmask) of primes
        # Two divisors are coprime iff their bitmasks are disjoint
        # So we need to count adjacent pairs in sorted order with disjoint bitmasks
        
        # Generate all (bitmask, value) pairs
        divs = []
        for mask in range(1 << k):
            val = 1
            for i in range(k):
                if mask & (1 << i):
                    val *= prime_list[i]
            divs.append((val, mask))
        
        divs.sort()
        
        score = 0
        prev_mask = divs[0][1]
        for i in range(1, len(divs)):
            curr_mask = divs[i][1]
            if prev_mask & curr_mask == 0:
                score += 1
            prev_mask = curr_mask
        return score

    best_score = -1
    best_n = 2

    # Phase 1: Try squarefree products with geometrically spaced primes
    # The idea: if primes are far apart in value, their subset products
    # create more "gaps" where adjacent divisors are coprime.
    
    # Try various base and ratio combinations
    for k in range(2, 16):
        # Try geometric sequences of primes
        # Start with small primes and multiply by increasing factors
        
        # Strategy 1: First k primes (baseline)
        primes_seq = all_primes[:k]
        n = 1
        for p in primes_seq:
            n *= p
        if n > LIMIT:
            break
        score = fast_tau_perp_squarefree(primes_seq)
        if score > best_score:
            best_score = score
            best_n = n
        
        # Strategy 2: Geometrically spaced - pick primes that are roughly
        # exponentially spaced to maximize coprime adjacencies
        # Use primes at positions: 0, 1, 2, 4, 8, ... (exponential spacing in index)
        if k <= 15:
            indices = [min(int(2**i), len(all_primes)-1) for i in range(k)]
            # Remove duplicates and sort
            indices = sorted(set(indices))
            if len(indices) >= 2:
                primes_seq = [all_primes[i] for i in indices if i < len(all_primes)]
                if len(primes_seq) >= 2:
                    n = 1
                    for p in primes_seq:
                        n *= p
                    if n <= LIMIT:
                        score = fast_tau_perp_squarefree(primes_seq)
                        if score > best_score:
                            best_score = score
                            best_n = n
        
        # Strategy 3: Skip primes to create larger gaps
        # Pick every other prime, every third, etc.
        for skip in range(1, 5):
            primes_seq = all_primes[::skip]
            # Take first k primes from this subsequence
            if len(primes_seq) >= k:
                primes_seq = primes_seq[:k]
            else:
                continue
            n = 1
            for p in primes_seq:
                n *= p
            if n > LIMIT:
                continue
            score = fast_tau_perp_squarefree(primes_seq)
            if score > best_score:
                best_score = score
                best_n = n

    # Phase 2: Random search with diverse prime selections
    random.seed(123)
    tried = set()
    
    # Determine max k for squarefree
    max_k_sq = 0
    prod = 1
    for i, p in enumerate(all_primes):
        if prod * p <= LIMIT:
            prod *= p
            max_k_sq = i + 1
        else:
            break

    for target_k in range(max(2, max_k_sq - 3), max_k_sq + 1):
        if target_k < 2:
            continue
        pool_size = min(50, len(all_primes))
        pool = all_primes[:pool_size]
        
        num_tries = 300
        for _ in range(num_tries):
            if time.time() > DEADLINE:
                break
            subset = random.sample(pool, target_k)
            subset.sort()
            key = tuple(subset)
            if key in tried:
                continue
            tried.add(key)
            
            n = 1
            for p in subset:
                n *= p
            if n > LIMIT:
                continue
            
            score = fast_tau_perp_squarefree(subset)
            if score > best_score:
                best_score = score
                best_n = n

    # Phase 3: Local search - swap primes in best_n
    if time.time() < DEADLINE - 10:
        def factorize_squarefree(n):
            """Factorize n assuming it's squarefree, return list of primes."""
            temp = n
            factors = []
            for p in all_primes:
                if p * p > temp:
                    break
                if temp % p == 0:
                    factors.append(p)
                    temp //= p
            if temp > 1:
                factors.append(temp)
            return factors
        
        best_primes = factorize_squarefree(best_n)
        
        for _ in range(2000):
            if time.time() > DEADLINE:
                break
            
            if not best_primes:
                break
            
            # Pick a random prime factor to replace
            idx = random.randint(0, len(best_primes) - 1)
            p_old = best_primes[idx]
            
            # Find a replacement prime (not already in n)
            p_new = None
            # Try nearby primes first
            old_idx = prime_to_idx.get(p_old, -1)
            for offset in range(1, 20):
                for cand_idx in [old_idx + offset, old_idx - offset]:
                    if 0 <= cand_idx < len(all_primes):
                        candidate = all_primes[cand_idx]
                        if candidate not in best_primes:
                            n_test = best_n // p_old * candidate
                            if n_test <= LIMIT and n_test >= 2:
                                p_new = candidate
                                break
                if p_new:
                    break
            
            if p_new is None:
                # Try random
                for _ in range(10):
                    candidate = random.choice(all_primes[:100])
                    if candidate not in best_primes:
                        n_test = best_n // p_old * candidate
                        if n_test <= LIMIT and n_test >= 2:
                            p_new = candidate
                            break
            
            if p_new is None:
                continue
            
            n_test = best_n // p_old * p_new
            if n_test < 2 or n_test > LIMIT:
                continue
            
            new_primes = factorize_squarefree(n_test)
            score_test = fast_tau_perp_squarefree(new_primes)
            if score_test > best_score:
                best_score = score_test
                best_n = n_test
                best_primes = new_primes

    # Phase 4: Try adding small exponents to best squarefree
    if time.time() < DEADLINE - 5:
        best_primes = factorize_squarefree(best_n)
        
        # Try squaring one prime and removing another to stay under LIMIT
        for i, p_sq in enumerate(best_primes):
            for j, p_rm in enumerate(best_primes):
                if i == j:
                    continue
                # Remove p_rm, square p_sq
                n_test = best_n // p_rm * p_sq
                if n_test <= LIMIT and n_test >= 2:
                    # Factorize
                    temp = n_test
                    factors = {}
                    for p in all_primes:
                        if p * p > temp:
                            break
                        if temp % p == 0:
                            cnt = 0
                            while temp % p == 0:
                                temp //= p
                                cnt += 1
                            factors[p] = cnt
                    if temp > 1:
                        factors[temp] = 1
                    
                    score = fast_tau_perp_from_factors(factors)
                    if score > best_score:
                        best_score = score
                        best_n = n_test

    return best_n

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
