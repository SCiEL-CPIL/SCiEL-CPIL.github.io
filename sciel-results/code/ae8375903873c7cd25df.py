import time
import math
import random
import sys

_START = time.time()
_HARD_DEADLINE = _START + 80.0  # Keep safety margin

def _init_spf(limit):
    spf = list(range(limit + 1))
    for i in range(2, int(limit**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

# Max sum of two k_i. If B=1, max k=10000, max sum=20000.
SPF_LIMIT = 20000
_spf = _init_spf(SPF_LIMIT)

def _prime_factors(num):
    if num <= 1:
        return frozenset()
    out = set()
    x = num
    while x > 1:
        p = _spf[x]
        out.add(p)
        while x % p == 0:
            x //= p
    return frozenset(out)

def _is_smooth(num, allowed_primes):
    """Check if num is composed only of allowed_primes."""
    if num <= 1:
        return True
    for p in allowed_primes:
        while num % p == 0:
            num //= p
    return num == 1

def _generate_smooth_numbers(max_val, primes):
    """Generate all numbers <= max_val with prime factors in primes."""
    smooth = set([1])
    changed = True
    while changed:
        changed = False
        for x in list(smooth):
            for p in primes:
                nx = x * p
                if nx <= max_val and nx not in smooth:
                    smooth.add(nx)
                    changed = True
    return sorted(smooth)

def _find_clique(candidates, n_target, smooth_check, time_limit):
    """
    Find n_target distinct candidates such that all pairwise sums pass smooth_check.
    Uses a greedy approach with random restarts and backtracking if needed.
    """
    if len(candidates) < n_target:
        return None
    
    # Precompute compatibility matrix?
    # candidates size is small (< 300 for 5-smooth up to 10000).
    # N=300, N^2 = 90,000 checks. Fast.
    N = len(candidates)
    comp = [[False]*N for _ in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            if smooth_check(candidates[i] + candidates[j]):
                comp[i][j] = True
                comp[j][i] = True
    
    # Heuristic: Sort candidates by degree (number of compatible neighbors)
    degrees = [sum(comp[i]) for i in range(N)]
    indices = list(range(N))
    indices.sort(key=lambda i: degrees[i], reverse=True)
    
    best_clique = []
    best_size = 0
    
    # We try multiple restarts with different initial orders or subsets
    num_restarts = 100
    for r in range(num_restarts):
        if time.time() > time_limit:
            break
        
        # Create a shuffled version of the sorted indices to break symmetry
        # But keep high-degree nodes likely to be picked first
        shuffled_indices = indices[:]
        random.shuffle(shuffled_indices)
        
        current_clique = []
        # Greedy addition
        for idx in shuffled_indices:
            # Check if idx is compatible with all in current_clique
            compatible = True
            for c_idx in current_clique:
                if not comp[idx][c_idx]:
                    compatible = False
                    break
            
            if compatible:
                current_clique.append(idx)
                if len(current_clique) == n_target:
                    return [candidates[i] for i in current_clique]
        
        if len(current_clique) > best_size:
            best_size = len(current_clique)
            best_clique = current_clique[:]
            
    # If we didn't find n_target, we might try a simple backtracking on the best clique found?
    # Or just return None and let the caller try a larger prime set.
    # Given the density, if greedy fails, backtracking might be needed, 
    # but usually for smooth numbers, the graph is quite dense.
    
    return None

def construct():
    deadline = time.time() + 80.0
    n = 12
    max_val = 10000
    
    # Try 2 primes: {2, 3}
    primes_2 = [2, 3]
    smooth_2 = _generate_smooth_numbers(max_val, primes_2)
    
    def check_smooth_2(x):
        return _is_smooth(x, primes_2)
        
    clique = _find_clique(smooth_2, n, check_smooth_2, deadline - 5.0)
    if clique:
        return clique

    # Try 3 primes: {2, 3, 5}
    primes_3 = [2, 3, 5]
    smooth_3 = _generate_smooth_numbers(max_val, primes_3)
    
    def check_smooth_3(x):
        return _is_smooth(x, primes_3)
        
    clique = _find_clique(smooth_3, n, check_smooth_3, deadline - 5.0)
    if clique:
        return clique

    # Try 4 primes: {2, 3, 5, 7}
    primes_4 = [2, 3, 5, 7]
    smooth_4 = _generate_smooth_numbers(max_val, primes_4)
    
    def check_smooth_4(x):
        return _is_smooth(x, primes_4)
        
    clique = _find_clique(smooth_4, n, check_smooth_4, deadline - 5.0)
    if clique:
        return clique

    # Fallback: Return a set that minimizes primes heuristically
    # Use small numbers, which tend to have fewer distinct prime factors in sums
    # e.g., powers of 2
    result = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    
    # Verify distinct and limit
    if len(set(result)) == n and all(x <= max_val for x in result):
         return result
    else:
         return list(range(1, 13))
