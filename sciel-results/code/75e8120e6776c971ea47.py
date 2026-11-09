import time
import math
import random

def _sieve_prime_factors(max_n):
    """Precompute distinct prime factors for all numbers up to max_n using sieve."""
    factors = [set() for _ in range(max_n + 1)]
    for d in range(2, max_n + 1):
        if not factors[d]:
            for m in range(d, max_n + 1, d):
                factors[m].add(d)
    return factors

def _count_distinct_primes(arr, pf):
    """Count distinct primes dividing all pairwise sums using precomputed factors."""
    primes = set()
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            s = arr[i] + arr[j]
            primes |= pf[s]
    return len(primes)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Precompute prime factors for all sums (max sum = 20000)
    pf = _sieve_prime_factors(20000)
    
    best_arr = None
    best_count = float('inf')
    
    def update_best(arr, count):
        nonlocal best_arr, best_count
        if count < best_count:
            best_count = count
            best_arr = arr[:]
            return best_count <= 4
    
    # Phase 1: Try structured families with small prime factor sets
    # Key idea: if all numbers are multiples of M, sums are multiples of M
    # Try various moduli and residue classes
    
    # Strategy A: Multiples of highly composite numbers
    moduli_to_try = [210, 420, 840, 630, 1260, 2520, 30, 60, 120, 240, 480, 960, 1680, 3360]
    
    for M in moduli_to_try:
        if time.time() > deadline - 10:
            break
        
        # Try all residue classes mod M that give at least 12 numbers <= 10000
        for r in range(M):
            if time.time() > deadline - 10:
                break
            
            # Generate numbers: r, r+M, r+2M, ... <= 10000
            # But r must be >= 1
            first = r
            if first < 1:
                first = r + M
                if first > 10000:
                    continue
            
            arr = []
            val = first
            while val <= 10000 and len(arr) < 12:
                arr.append(val)
                val += M
            
            if len(arr) < 12:
                continue
            
            arr = arr[:12]
            count = _count_distinct_primes(arr, pf)
            if update_best(arr, count):
                return best_arr
    
    # Strategy B: Try to find 12 numbers whose pairwise sums are 7-smooth
    # i.e., only have prime factors {2, 3, 5, 7}
    # Precompute which numbers are 7-smooth
    smooth7 = [False] * 20001
    for s in range(2, 20001):
        x = s
        for p in (2, 3, 5, 7):
            while x % p == 0:
                x //= p
        smooth7[s] = (x == 1)
    
    # Try to find 12 numbers a1 < a2 < ... < a12 such that all ai+aj are 7-smooth
    # Use backtracking with pruning
    # First, find candidate numbers: those that can pair with many others to give smooth sums
    
    # For each number x in 1..10000, count how many y in 1..10000, y != x, have x+y smooth
    # This is expensive, so restrict to a subset
    
    # Better approach: try numbers from specific residue classes
    # If x ≡ r mod 2, x+y is even iff y ≡ r mod 2
    # So restrict to same parity
    
    # Try: all even numbers, all numbers ≡ 0 mod 3, etc.
    # Actually, let's try a targeted search
    
    # Candidate pool: numbers that are themselves 7-smooth (increases chance sums are too)
    smooth_numbers = [x for x in range(1, 10001) if smooth7[x]]
    
    if len(smooth_numbers) >= 12:
        # Try greedy: pick numbers that maximize smooth pair count
        # For each pair (i,j) in smooth_numbers, check if sum is smooth
        # This is O(n^2) which might be too slow for large n
        
        # Limit to first 200 smooth numbers for tractability
        pool = smooth_numbers[:200]
        n_pool = len(pool)
        
        # Build adjacency: adj[i] = set of j > i such that pool[i]+pool[j] is smooth
        adj = [set() for _ in range(n_pool)]
        for i in range(n_pool):
            for j in range(i + 1, n_pool):
                if smooth7[pool[i] + pool[j]]:
                    adj[i].add(j)
                    adj[j].add(i)
        
        # Find clique of size 12 using backtracking
        def find_clique(candidates, chosen, depth):
            if time.time() > deadline - 5:
                return None
            if len(chosen) == 12:
                return chosen[:]
            if not candidates:
                return None
            if len(candidates) < 12 - len(chosen):
                return None
            
            # Try each candidate
            for idx in sorted(candidates):
                new_chosen = chosen + [idx]
                # New candidates must be adjacent to all in new_chosen
                new_cands = set()
                for c in candidates:
                    if c <= idx:
                        continue
                    ok = True
                    for ch in new_chosen:
                        if c not in adj[ch]:
                            ok = False
                            break
                    if ok:
                        new_cands.add(c)
                
                if len(new_cands) >= 12 - len(new_chosen) - 1 or len(new_chosen) == 11:
                    result = find_clique(new_cands, new_chosen, depth + 1)
                    if result is not None:
                        return result
            return None
        
        clique = find_clique(set(range(n_pool)), [], 0)
        if clique is not None:
            arr = [pool[i] for i in clique]
            count = _count_distinct_primes(arr, pf)
            if update_best(arr, count):
                return best_arr
    
    # Strategy C: Local search to improve current best
    if best_arr is None:
        best_arr = [210 * i for i in range(1, 13)]
        best_count = _count_distinct_primes(best_arr, pf)
    
    random.seed(42)
    
    for trial in range(30):
        if time.time() > deadline - 5:
            break
        
        if trial == 0:
            arr = best_arr[:]
        else:
            # Random restart with structured initialization
            d = random.choice([2, 3, 4, 5, 6, 7, 10, 12, 14, 15, 20, 21, 24, 28, 30, 35, 40, 42, 60, 70, 84, 105, 120, 140, 168, 210, 420, 840])
            r = random.randint(0, d - 1)
            first = r
            if first < 1:
                first += d
            arr = []
            val = first
            while val <= 10000 and len(arr) < 12:
                arr.append(val)
                val += d
            if len(arr) < 12:
                # Fill with random
                while len(arr) < 12:
                    x = random.randint(1, 10000)
                    if x not in arr:
                        arr.append(x)
            arr = arr[:12]
        
        count = _count_distinct_primes(arr, pf)
        if count < best_count:
            best_count = count
            best_arr = arr[:]
            if best_count <= 4:
                return best_arr
        
        # Local improvement: try replacing each element
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for pos in range(12):
                if time.time() > deadline - 5:
                    break
                old = arr[pos]
                # Try various replacements
                candidates = set()
                for delta in range(-100, 101, 1):
                    c = old + delta
                    if 1 <= c <= 10000 and c not in arr:
                        candidates.add(c)
                # Add some random candidates
                for _ in range(50):
                    c = random.randint(1, 10000)
                    if c not in arr:
                        candidates.add(c)
                
                for c in list(candidates):
                    if time.time() > deadline - 5:
                        break
                    arr[pos] = c
                    new_count = _count_distinct_primes(arr, pf)
                    if new_count < count:
                        count = new_count
                        improved = True
                        if count < best_count:
                            best_count = count
                            best_arr = arr[:]
                            if best_count <= 4:
                                return best_arr
                    else:
                        arr[pos] = old
                
                if improved:
                    break
        
        if time.time() > deadline - 5:
            break
    
    return best_arr
