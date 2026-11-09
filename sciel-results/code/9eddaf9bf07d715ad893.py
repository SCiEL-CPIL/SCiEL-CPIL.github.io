"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from functools import lru_cache

_START = time.time()
DEADLINE = _START + 140  # 140s hard limit, 10s safety margin

def _elapsed():
    return time.time() - _START

def _time_left():
    return DEADLINE - time.time()

def _factorize(n):
    """Return prime factorization as list of (p, e) pairs."""
    if n <= 1:
        return []
    factors = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            e = 0
            while n % d == 0:
                n //= d
                e += 1
            factors.append((d, e))
        d += 1 if d == 2 else 2
    if n > 1:
        factors.append((n, 1))
    return factors

@lru_cache(maxsize=2000000)
def _sigma_cached(n):
    """Compute sigma(n) = sum of divisors of n using factorization."""
    if n <= 1:
        return n
    factors = _factorize(n)
    result = 1
    for p, e in factors:
        result *= (p**(e+1) - 1) // (p - 1)
    return result

def _sigma(n):
    return _sigma_cached(n)

def _divisors(n):
    """Generate all divisors of n."""
    factors = _factorize(n)
    divs = [1]
    for p, e in factors:
        new_divs = []
        for d in divs:
            for i in range(e + 1):
                new_divs.append(d * (p ** i))
        divs = new_divs
    return divs

def _count_solutions(M):
    """Count distinct k <= sqrt(M) with k*sigma(k) = M. Returns (count, set)."""
    if M < 1:
        return 0, set()
    sqrtM = math.isqrt(M)
    solutions = set()
    for k in _divisors(M):
        if k > sqrtM:
            continue
        if k * _sigma(k) == M:
            solutions.add(k)
    return len(solutions), solutions

def _num_divisors(n):
    factors = _factorize(n)
    result = 1
    for _, e in factors:
        result *= (e + 1)
    return result

def _try_candidate(M, best_state):
    """Try a single M candidate; update best_state if better."""
    if M < 1 or M > 10**18:
        return best_state
    count, solutions = _count_solutions(M)
    if count > best_state[0]:
        return (count, M, sorted(solutions))
    return best_state

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 130s working limit, 20s safety
    
    # Start with known good solution
    best_count = 3
    best_M = 333312
    best_k = [336, 372, 434]
    
    # === PHASE 1: Systematic search over products of small primes ===
    # M = 2^a * 3^b * 5^c * 7^d * 11^e * 13^f
    # We want many divisors, so balanced/high exponents on small primes
    
    max_val = 10**18
    primes_to_use = [2, 3, 5, 7, 11, 13]
    
    # Compute max exponents
    max_exp = {}
    for p in primes_to_use:
        e = 0
        v = 1
        while v * p <= max_val:
            v *= p
            e += 1
        max_exp[p] = e
    
    # Generate candidates via recursive exponent enumeration
    # But limit total candidates to avoid explosion
    candidates = set([333312])
    
    # Use a more targeted approach: enumerate exponent vectors
    # with a budget on total number of candidates
    MAX_CANDIDATES = 200000
    
    def gen_candidates(idx, current, exps):
        nonlocal best_count, best_M, best_k
        if time.time() > deadline - 5:
            return
        if idx == len(primes_to_use):
            if current <= max_val and current > 1:
                candidates.add(current)
            return
        p = primes_to_use[idx]
        for e in range(max_exp[p] + 1):
            new_current = current * (p ** e)
            if new_current > max_val:
                break
            exps.append(e)
            gen_candidates(idx + 1, new_current, exps)
            exps.pop()
            if len(candidates) >= MAX_CANDIDATES:
                return
    
    gen_candidates(0, 1, [])
    
    # Also try adding one more prime factor to some candidates
    extra_primes = [17, 19, 23, 29, 31, 37]
    base_candidates = list(candidates)
    for base in base_candidates:
        if time.time() > deadline - 5:
            break
        for ep in extra_primes:
            val = base * ep
            if val <= max_val:
                candidates.add(val)
                val2 = val * ep
                if val2 <= max_val:
                    candidates.add(val2)
            if len(candidates) >= MAX_CANDIDATES * 2:
                break
    
    # === PHASE 2: Evaluate candidates, sorted by divisor count ===
    # First, filter out candidates with very few divisors (unlikely to have many solutions)
    # Sort by num_divisors descending to find good ones first
    
    # For efficiency, only sort a subset if too many
    cand_list = list(candidates)
    if len(cand_list) > 100000:
        # Sample top by divisor count
        # Compute divisor counts for a sample
        import random
        random.seed(42)
        sample_size = min(50000, len(cand_list))
        sample = random.sample(cand_list, sample_size)
        sample.sort(key=lambda x: -_num_divisors(x))
        # Also add the known good one
        if 333312 not in sample:
            sample.append(333312)
        cand_list = sample
    
    cand_list.sort(key=lambda x: -_num_divisors(x))
    
    evaluated = 0
    for M in cand_list:
        if time.time() > deadline - 10:
            break
        if M < 1 or M > 10**18:
            continue
        count, solutions = _count_solutions(M)
        evaluated += 1
        if count > best_count:
            best_count = count
            best_M = M
            best_k = sorted(solutions)
            if best_count >= 20:
                # Good enough, break early if time is tight
                if time.time() > deadline - 30:
                    break
    
    # === PHASE 3: Local search around best M ===
    # Try multiplying/dividing by small primes to find neighbors with more solutions
    if time.time() < deadline - 15:
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        neighbors = set()
        for p in small_primes:
            if best_M * p <= max_val:
                neighbors.add(best_M * p)
            if best_M % p == 0:
                neighbors.add(best_M // p)
                if best_M % (p * p) == 0:
                    neighbors.add(best_M // (p * p))
        
        for M in neighbors:
            if time.time() > deadline - 10:
                break
            if M < 1 or M > 10**18:
                continue
            count, solutions = _count_solutions(M)
            if count > best_count:
                best_count = count
                best_M = M
                best_k = sorted(solutions)
    
    # === PHASE 4: Broader local search ===
    if time.time() < deadline - 20:
        # Try more neighbors: multiply by products of two small primes
        small_primes2 = [2, 3, 5, 7, 11, 13]
        neighbors2 = set()
        for p1 in small_primes2:
            for p2 in small_primes2:
                pp = p1 * p2
                if best_M * pp <= max_val:
                    neighbors2.add(best_M * pp)
                if best_M % pp == 0:
                    neighbors2.add(best_M // pp)
        
        for M in neighbors2:
            if time.time() > deadline - 10:
                break
            if M < 1 or M > 10**18:
                continue
            count, solutions = _count_solutions(M)
            if count > best_count:
                best_count = count
                best_M = M
                best_k = sorted(solutions)
    
    # === Final validation ===
    sqrtM = math.isqrt(best_M)
    valid_k = []
    for k in best_k:
        if 1 <= k <= sqrtM and isinstance(k, int) and k * _sigma(k) == best_M:
            valid_k.append(k)
    
    if not valid_k:
        # Fallback to known good
        return {'M': 333312, 'k': [336, 372, 434]}
    
    # Ensure types are correct
    best_k = [int(k) for k in valid_k]
    
    return {'M': int(best_M), 'k': best_k}

# EVOLVE-BLOCK-END
