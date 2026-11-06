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
    deadline = start_time + 85
    
    n = 12
    max_val = 10000
    rng = random.Random(42)
    
    best_set = None
    best_count = float('inf')
    
    # Precompute prime factor sets for all possible sums (2 to 20000)
    # This is the key speedup: O(1) lookup instead of O(sqrt(n)) per factorization
    max_sum = 2 * max_val
    pf_cache = [None] * (max_sum + 1)
    for s in range(2, max_sum + 1):
        pf_cache[s] = _prime_factors(s)
    
    def evaluate(arr):
        primes = set()
        for i in range(n):
            for j in range(i + 1, n):
                primes |= pf_cache[arr[i] + arr[j]]
        return primes
    
    # Strategy 1: Greedy construction with restricted prime sets
    # Try different small prime sets and greedily build valid number sets
    
    prime_sets_to_try = [
        {2},
        {2, 3},
        {2, 5},
        {2, 7},
        {2, 3, 5},
        {2, 3, 7},
        {2, 5, 7},
        {2, 3, 5, 7},
        {2, 3, 5, 11},
        {2, 3, 5, 13},
        {2, 3, 5, 17},
        {2, 3, 5, 19},
        {2, 3, 7, 11},
        {2, 3, 7, 13},
        {2, 3, 7, 17},
        {2, 5, 7, 11},
        {2, 5, 7, 13},
        {2, 3, 5, 7, 11},
        {2, 3, 5, 7, 13},
        {2, 3, 5, 7, 17},
        {2, 3, 5, 7, 19},
        {2, 3, 5, 11, 13},
        {2, 3, 7, 11, 13},
        {2, 3, 5, 7, 11, 13},
        {2, 3, 5, 7, 11, 17},
        {2, 3, 5, 7, 13, 17},
    ]
    
    def is_smooth(s, allowed_primes):
        """Check if s has only prime factors in allowed_primes."""
        for p in allowed_primes:
            while s % p == 0:
                s //= p
        return s == 1
    
    def greedy_build(allowed_primes, start_vals=None, rng_seed=42):
        """Greedily build a set of n distinct numbers <= max_val where all pair sums are smooth."""
        local_rng = random.Random(rng_seed)
        if start_vals is None:
            # Start with small numbers
            candidates = list(range(1, min(max_val, 500) + 1))
            local_rng.shuffle(candidates)
        else:
            candidates = list(start_vals)
            local_rng.shuffle(candidates)
        
        chosen = []
        for cand in candidates:
            if len(chosen) >= n:
                break
            # Check if cand can be added
            valid = True
            for existing in chosen:
                s = cand + existing
                if not is_smooth(s, allowed_primes):
                    valid = False
                    break
            if valid:
                chosen.append(cand)
        
        if len(chosen) >= n:
            return chosen[:n]
        return None
    
    # Try greedy construction for each prime set
    for ps in prime_sets_to_try:
        if time.time() > deadline - 5:
            break
        for seed in range(20):
            if time.time() > deadline - 5:
                break
            result = greedy_build(ps, rng_seed=seed)
            if result is not None:
                primes = evaluate(result)
                if len(primes) < best_count:
                    best_count = len(primes)
                    best_set = list(result)
    
    # Strategy 2: Structured candidates (multiples of M)
    for m in [2, 4, 8, 16, 3, 6, 12, 24, 5, 10, 20, 40, 7, 14, 28, 56, 11, 22, 44, 88, 13, 26, 52, 104]:
        if time.time() > deadline - 5:
            break
        # Take consecutive multiples
        vals = list(range(m, m + n * m, m))
        if vals[-1] <= max_val:
            primes = evaluate(vals)
            if len(primes) < best_count:
                best_count = len(primes)
                best_set = list(vals)
        
        # Take first n multiples
        vals2 = list(range(m, max_val + 1, m))[:n]
        if len(vals2) == n:
            primes = evaluate(vals2)
            if len(primes) < best_count:
                best_count = len(primes)
                best_set = list(vals2)
    
    # Strategy 3: Local search from best found so far
    if best_set is not None:
        def local_search(arr):
            current = list(arr)
            current_primes = evaluate(current)
            current_count = len(current_primes)
            
            while time.time() < deadline - 3:
                improved = False
                for i in range(n):
                    if time.time() >= deadline - 3:
                        break
                    old_val = current[i]
                    for trial in range(30):
                        new_val = rng.randint(1, max_val)
                        if new_val == old_val:
                            continue
                        if new_val in current:
                            continue
                        
                        new_current = list(current)
                        new_current[i] = new_val
                        
                        # Efficient: only recompute primes for pairs involving index i
                        # Remove old contributions, add new ones
                        new_primes = set()
                        for a in range(n):
                            for b in range(a + 1, n):
                                if a == i or b == i:
                                    new_primes |= pf_cache[new_current[a] + new_current[b]]
                                else:
                                    new_primes |= pf_cache[current[a] + current[b]]
                        
                        if len(new_primes) < current_count:
                            current = new_current
                            current_primes = new_primes
                            current_count = len(new_primes)
                            improved = True
                            if current_count < best_count:
                                best_count = current_count
                                best_set = list(current)
                            break
                        elif len(new_primes) == current_count and rng.random() < 0.03:
                            current = new_current
                            current_primes = new_primes
                            break
                
                if not improved:
                    break
            
            return current, current_count
        
        current, current_count = local_search(best_set)
        if current_count < best_count:
            best_count = current_count
            best_set = current
    
    # Strategy 4: More random restarts with local search
    if time.time() < deadline - 10:
        for restart in range(100):
            if time.time() >= deadline - 10:
                break
            cand = rng.sample(range(1, max_val + 1), n)
            primes = evaluate(cand)
            c = len(primes)
            if c < best_count:
                best_count = c
                best_set = list(cand)
            else:
                # Quick local search
                current = list(cand)
                current_primes = primes
                current_count = c
                for step in range(20):
                    if time.time() >= deadline - 10:
                        break
                    i = rng.randint(0, n - 1)
                    new_val = rng.randint(1, max_val)
                    if new_val in current:
                        continue
                    new_current = list(current)
                    new_current[i] = new_val
                    new_primes = set()
                    for a in range(n):
                        for b in range(a + 1, n):
                            new_primes |= pf_cache[new_current[a] + new_current[b]]
                    if len(new_primes) < current_count:
                        current = new_current
                        current_primes = new_primes
                        current_count = len(new_primes)
                        if current_count < best_count:
                            best_count = current_count
                            best_set = list(current)
    
    return best_set
# EVOLVE-BLOCK-END
