"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from math import isqrt

# EVOLVE-BLOCK-START
def _prime_factors(n):
    """Return set of prime factors of n."""
    factors = set()
    if n % 2 == 0:
        factors.add(2)
        while n % 2 == 0:
            n //= 2
    f = 3
    while f * f <= n:
        if n % f == 0:
            factors.add(f)
            while n % f == 0:
                n //= f
        f += 2
    if n > 1:
        factors.add(n)
    return factors

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = 12
    max_val = 10000
    rng = random.Random(42)
    
    best_set = None
    best_primes = None
    best_count = float('inf')
    
    def evaluate(arr):
        primes = set()
        for i in range(n):
            for j in range(i + 1, n):
                primes |= _prime_factors(arr[i] + arr[j])
        return primes
    
    # Key insight: if all numbers are multiples of M, all sums are multiples of M.
    # But sums may have additional prime factors.
    # Try structured candidates first.
    
    candidates = []
    
    # Multiples of various numbers
    for m in [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120]:
        vals = list(range(m, max_val + 1, m))
        if len(vals) >= n:
            candidates.append(vals[:n])
    
    # Consecutive small numbers
    candidates.append(list(range(1, n + 1)))
    candidates.append(list(range(2, 2 + 2*n, 2)))
    
    # All ≡ 1 mod small primes
    for mod in [3, 5, 7, 11]:
        vals = [x for x in range(1, max_val + 1) if x % mod == 1]
        if len(vals) >= n:
            candidates.append(vals[:n])
    
    # Random
    for _ in range(10):
        candidates.append(rng.sample(range(1, max_val + 1), n))
    
    # Evaluate all candidates
    for cand in candidates:
        if len(set(cand)) < n:
            continue
        primes = evaluate(cand)
        if len(primes) < best_count:
            best_count = len(primes)
            best_primes = primes
            best_set = list(cand)
    
    # Local search with efficient delta evaluation
    def local_search(arr, primes, count):
        nonlocal best_set, best_primes, best_count
        current = list(arr)
        current_primes = set(primes)
        current_count = count
        
        while time.time() < deadline - 5:
            improved = False
            for i in range(n):
                if time.time() >= deadline - 5:
                    break
                old_val = current[i]
                for trial in range(20):
                    new_val = rng.randint(1, max_val)
                    if new_val == old_val or new_val in set(current[:i]) | set(current[i+1:]):
                        continue
                    
                    new_current = list(current)
                    new_current[i] = new_val
                    
                    # Compute new prime set
                    new_primes = set()
                    for a in range(n):
                        for b in range(a + 1, n):
                            new_primes |= _prime_factors(new_current[a] + new_current[b])
                    
                    if len(new_primes) < current_count:
                        current = new_current
                        current_primes = new_primes
                        current_count = len(new_primes)
                        improved = True
                        if current_count < best_count:
                            best_count = current_count
                            best_primes = current_primes
                            best_set = list(current)
                        break
                    elif len(new_primes) == current_count and rng.random() < 0.05:
                        current = new_current
                        current_primes = new_primes
                        break
            
            if not improved:
                # Try swapping two elements
                swapped = False
                for i in range(n):
                    if time.time() >= deadline - 5:
                        break
                    for j in range(i + 1, n):
                        new_current = list(current)
                        new_current[i], new_current[j] = current[j], current[i]
                        if len(set(new_current)) < n:
                            continue
                        new_primes = set()
                        for a in range(n):
                            for b in range(a + 1, n):
                                new_primes |= _prime_factors(new_current[a] + new_current[b])
                        if len(new_primes) < current_count:
                            current = new_current
                            current_primes = new_primes
                            current_count = len(new_primes)
                            swapped = True
                            if current_count < best_count:
                                best_count = current_count
                                best_primes = current_primes
                                best_set = list(current)
                            break
                    if swapped:
                        break
                if not swapped:
                    break
        
        return current, current_primes, current_count
    
    # Run local search from best candidate
    current, current_primes, current_count = local_search(
        best_set, best_primes, best_count
    )
    
    # More restarts if time remains
    if time.time() < deadline - 10:
        for restart in range(50):
            if time.time() >= deadline - 10:
                break
            cand = rng.sample(range(1, max_val + 1), n)
            primes = evaluate(cand)
            if len(primes) < best_count:
                best_count = len(primes)
                best_primes = primes
                best_set = list(cand)
            else:
                cand2, p2, c2 = local_search(cand, primes, len(primes))
                if c2 < best_count:
                    best_count = c2
                    best_primes = p2
                    best_set = cand2
    
    return best_set
# EVOLVE-BLOCK-END
