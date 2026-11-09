"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math

# Precompute prime factors for all numbers up to 20000
MAX_SUM = 20000
_pf_cache = [set() for _ in range(MAX_SUM + 1)]

def _init_prime_factors():
    for i in range(2, MAX_SUM + 1):
        if not _pf_cache[i]:
            f = 2
            n = i
            while f * f <= n:
                while n % f == 0:
                    _pf_cache[i].add(f)
                    n //= f
                f += 1
            if n > 1:
                _pf_cache[i].add(n)
            # Propagate to multiples
            for j in range(i*2, MAX_SUM+1, i):
                if not _pf_cache[j]:
                    _pf_cache[j] = set(_pf_cache[i])
                    # Need to add other factors
                    n = j // i
                    while n > 1:
                        for f in range(2, int(math.isqrt(n)) + 1):
                            if n % f == 0:
                                _pf_cache[j].add(f)
                                while n % f == 0:
                                    n //= f
                        if n > 1:
                            _pf_cache[j].add(n)
                        break
    # Simpler: just compute each directly
    for i in range(2, MAX_SUM + 1):
        factors = set()
        n = i
        f = 2
        while f * f <= n:
            while n % f == 0:
                factors.add(f)
                n //= f
            f += 1
        if n > 1:
            factors.add(n)
        _pf_cache[i] = factors

_init_prime_factors()

def count_distinct_primes(nums):
    primes = set()
    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            primes |= _pf_cache[nums[i] + nums[j]]
    return len(primes)

def construct():
    n = 12
    max_val = 10000
    
    # Multiple initializations
    candidates = []
    
    # Init 1: small odd numbers
    candidates.append(list(range(1, 2*n+1, 2)))
    
    # Init 2: numbers ≡ 1 mod 6
    candidates.append([6*i+1 for i in range(n)])
    
    # Init 3: powers of 2 times small odd
    candidates.append([2**i * 3 for i in range(n)])
    
    # Init 4: Fibonacci-like
    fib = [1, 1]
    for i in range(n-2):
        fib.append(fib[-1] + fib[-2])
    candidates.append(fib[:n])
    
    # Init 5: random odd numbers
    random.seed(42)
    for _ in range(5):
        c = random.sample(range(1, max_val+1, 2), n)
        candidates.append(c)
    
    best = None
    best_score = float('inf')
    
    for init in candidates:
        if max(init) > max_val:
            continue
        score = count_distinct_primes(init)
        if score < best_score:
            best_score = score
            best = init[:]
    
    # Local search from best
    current = best[:]
    current_score = best_score
    best = current[:]
    best_score = current_score
    
    random.seed(123)
    for it in range(30000):
        i = random.randint(0, n-1)
        new_val = random.randint(1, max_val)
        if new_val in current:
            continue
        
        old_val = current[i]
        current[i] = new_val
        new_score = count_distinct_primes(current)
        
        if new_score <= current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best = current[:]
        else:
            current[i] = old_val
    
    return best

# EVOLVE-BLOCK-END
