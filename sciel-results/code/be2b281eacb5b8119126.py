"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    # Precompute smallest prime factor for fast factorization
    MAX_VAL = 20000
    spf = list(range(MAX_VAL + 1))
    for i in range(2, int(MAX_VAL**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_VAL + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def factorize(n):
        """Return set of prime factors of n."""
        if n <= 1:
            return set()
        factors = set()
        while n > 1:
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        return factors
    
    def evaluate(arr):
        """Return the number of distinct prime factors of all pairwise sums."""
        primes = set()
        n = len(arr)
        for i in range(n):
            for j in range(i+1, n):
                s = arr[i] + arr[j]
                primes |= factorize(s)
        return len(primes)
    
    def get_prime_set(arr):
        """Return the set of distinct prime factors of all pairwise sums."""
        primes = set()
        n = len(arr)
        for i in range(n):
            for j in range(i+1, n):
                s = arr[i] + arr[j]
                primes |= factorize(s)
        return primes
    
    # Strategy: For each small prime set P, find 12 numbers where all pairwise sums are P-smooth
    # P-smooth means all prime factors are in P
    
    # Try various small prime sets
    prime_sets_to_try = [
        {2},
        {2, 3},
        {2, 3, 5},
        {2, 3, 5, 7},
        {2, 3, 5, 7, 11},
        {2, 3, 5, 7, 11, 13},
    ]
    
    def is_p_smooth(n, P):
        """Check if n is P-smooth (all prime factors in P)."""
        while n > 1:
            p = spf[n]
            if p not in P:
                return False
            while n % p == 0:
                n //= p
        return True
    
    # Precompute P-smooth numbers up to 20000 for each prime set
    smooth_cache = {}
    for P in prime_sets_to_try:
        key = frozenset(P)
        smooth_nums = set()
        for n in range(2, 20001):
            if is_p_smooth(n, P):
                smooth_nums.add(n)
        smooth_cache[key] = smooth_nums
    
    # For each prime set, build a graph and find a clique of size 12
    best = None
    best_score = 999
    best_primes = set()
    
    for P in prime_sets_to_try:
        key = frozenset(P)
        smooth_nums = smooth_cache[key]
        
        # Generate candidate numbers: numbers where all pairwise sums with other candidates are P-smooth
        # Start with a pool of candidates
        candidates = list(range(1, 10001))
        
        # Filter: keep only numbers where n + m is P-smooth for some m in range
        # Actually, let's be smarter: for each candidate n, check how many other candidates m
        # satisfy that n + m is P-smooth
        
        # Greedy clique finding:
        # 1. Sort candidates by "degree" (number of P-smooth sums)
        # 2. Pick the highest-degree vertex
        # 3. Add vertices that are compatible with all current clique members
        
        # Compute degree for each candidate
        degrees = {}
        for n in candidates:
            deg = 0
            for m in candidates:
                if m != n and is_p_smooth(n + m, P):
                    deg += 1
            degrees[n] = deg
        
        # Sort by degree descending
        sorted_candidates = sorted(candidates, key=lambda x: -degrees[x])
        
        # Greedy clique building
        clique = []
        for n in sorted_candidates:
            if len(clique) >= 12:
                break
            # Check if n is compatible with all current clique members
            compatible = True
            for m in clique:
                if not is_p_smooth(n + m, P):
                    compatible = False
                    break
            if compatible:
                clique.append(n)
        
        if len(clique) == 12:
            score = evaluate(clique)
            primes = get_prime_set(clique)
            if score < best_score:
                best = clique
                best_score = score
                best_primes = primes
    
    # If we didn't find a perfect clique, use the best we have
    if best is None:
        # Fallback: use the initial structured set
        best = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]
        best_score = evaluate(best)
        best_primes = get_prime_set(best)
    
    # Local search to improve
    random.seed(42)
    
    # Generate candidate pool
    candidate_pool = []
    for m in [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30]:
        for k in range(1, 10000 // m + 1):
            val = k * m
            if 1 <= val <= 10000:
                candidate_pool.append(val)
    for _ in range(3000):
        candidate_pool.append(random.randint(1, 10000))
    candidate_pool = list(set(candidate_pool))
    random.shuffle(candidate_pool)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(12):
            old_val = best[i]
            for new_val in candidate_pool:
                if time.time() > deadline:
                    break
                if new_val == old_val or new_val in best:
                    continue
                candidate = best[:]
                candidate[i] = new_val
                score = evaluate(candidate)
                if score < best_score:
                    best = candidate
                    best_score = score
                    best_primes = get_prime_set(best)
                    improved = True
                    break
    
    return best
# EVOLVE-BLOCK-END
