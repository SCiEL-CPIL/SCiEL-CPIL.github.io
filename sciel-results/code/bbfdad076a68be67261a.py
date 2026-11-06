"""Rewritten program: targeted search for A,B with 9 distinct pairwise-coprime sums."""
import time
import math
import random

_DEADLINE = 85.0

def _is_prime(n):
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

def _smallest_prime_factor(n):
    """Return smallest prime factor of n, or n if prime."""
    if n < 2:
        return 1
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    i = 5
    while i * i <= n:
        if n % i == 0:
            return i
        if n % (i + 2) == 0:
            return i + 2
        i += 6
    return n

def _pairwise_coprime(values):
    """Check if all distinct values in the list are pairwise coprime using SPF tracking."""
    seen_spf = set()
    for v in values:
        if v in seen_spf:
            continue  # duplicate value, skip
        spf = _smallest_prime_factor(v)
        if spf != 1:
            # Check if any already-seen value shares this prime factor
            # More efficient: track all prime factors used
            pass
        # Full pairwise check for correctness
        for v2 in values:
            if v2 > v:  # avoid double check
                if math.gcd(v, v2) != 1:
                    return False
    return True

def _pairwise_coprime_fast(values):
    """Faster pairwise coprime check using prime factor tracking."""
    vals = list(set(values))
    used_primes = set()
    for v in vals:
        n = v
        p = 2
        while p * p <= n:
            if n % p == 0:
                if p in used_primes:
                    return False
                used_primes.add(p)
                while n % p == 0:
                    n //= p
            p += 1 if p == 2 else 2
        if n > 1:
            if n in used_primes:
                return False
            used_primes.add(n)
    return True

def _sumset(A, B):
    return set(a + b for a in A for b in B)

def _valid_and_score(A, B):
    s = _sumset(A, B)
    if _pairwise_coprime_fast(s):
        return len(s)
    return 0

def construct():
    """Return two lists A,B each with exactly 3 distinct integers in [1,128].
    All distinct values in A+B must be pairwise coprime. Maximize |A+B|."""
    start_time = time.time()
    
    # Known valid solution as fallback
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = _valid_and_score(best_A, best_B)
    
    # Strategy 1: Try to find all-9-prime solutions
    # For B = {b1, b2, b3}, we need A = {a1, a2, a3} such that
    # all 9 values a_i + b_j are distinct primes
    
    # Precompute primes up to 256 (max sum = 128+128)
    primes_set = set(p for p in range(2, 257) if _is_prime(p))
    
    # Try many B configurations
    rng = random.Random(42)
    
    # Phase 1: Structured search over small B values
    found = False
    for b1 in range(1, 30):
        if time.time() - start_time > _DEADLINE - 20:
            break
        for b2 in range(b1 + 1, 50):
            if time.time() - start_time > _DEADLINE - 20:
                break
            for b3 in range(b2 + 1, 70):
                if time.time() - start_time > _DEADLINE - 20:
                    break
                
                B = [b1, b2, b3]
                
                # For each a, the three sums a+b1, a+b2, a+b3 should be primes
                # Find all a in [1,128] where a+b1, a+b2, a+b3 are all prime
                valid_ais = []
                for a in range(1, 129):
                    if (a + b1) in primes_set and (a + b2) in primes_set and (a + b3) in primes_set:
                        valid_ais.append(a)
                
                if len(valid_ais) >= 3:
                    # Try all triples from valid_ais
                    for i in range(len(valid_ais)):
                        if time.time() - start_time > _DEADLINE - 15:
                            break
                        for j in range(i + 1, len(valid_ais)):
                            if time.time() - start_time > _DEADLINE - 15:
                                break
                            for k in range(j + 1, len(valid_ais)):
                                A = [valid_ais[i], valid_ais[j], valid_ais[k]]
                                s = _sumset(A, B)
                                if len(s) == 9:
                                    # All sums are prime, so pairwise coprime
                                    return A, B
                                elif len(s) > best_score:
                                    best_score = len(s)
                                    best_A, best_B = A, B
                                if len(s) == 9:
                                    found = True
                                    return A, B
    
    # Phase 2: Random search with prime bias
    if not found:
        for _ in range(2000000):
            if time.time() - start_time > _DEADLINE - 5:
                break
            
            # Generate B with some structure
            b1 = rng.randint(1, 40)
            b2 = rng.randint(b1 + 3, 80)
            b3 = rng.randint(b2 + 3, 128)
            B = [b1, b2, b3]
            
            # Find A values that give prime sums
            valid_ais = []
            for a in range(1, 129):
                if (a + b1) in primes_set and (a + b2) in primes_set and (a + b3) in primes_set:
                    valid_ais.append(a)
            
            if len(valid_ais) >= 3:
                # Pick 3 distinct a's
                for _trial in range(5):
                    A = sorted(rng.sample(valid_ais, min(3, len(valid_ais))))
                    if len(A) < 3:
                        break
                    s = _sumset(A, B)
                    if len(s) == 9:
                        return A, B
                    elif len(s) > best_score and _pairwise_coprime_fast(s):
                        best_score = len(s)
                        best_A, best_B = A, B
    
    # Phase 3: Broader random search (not just prime sums, just pairwise coprime)
    if best_score < 9:
        for _ in range(500000):
            if time.time() - start_time > _DEADLINE - 2:
                break
            A = sorted(rng.sample(range(1, 129), 3))
            B = sorted(rng.sample(range(1, 129), 3))
            s = _sumset(A, B)
            if len(s) == 9 and _pairwise_coprime_fast(s):
                return A, B
            elif len(s) > best_score and _pairwise_coprime_fast(s):
                best_score = len(s)
                best_A, best_B = A, B
    
    # Verify final answer
    if _valid_and_score(best_A, best_B) == 0:
        return [1, 7, 13], [4, 10, 16]
    
    return best_A, best_B
