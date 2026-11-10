"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 40, 'maximum_size': 18}

# Precompute prime factorizations for all numbers in [2, 40]
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def _factorize(num):
    """Return dict of prime->exponent for num."""
    factors = {}
    x = num
    for p in _PRIMES:
        if x % p == 0:
            cnt = 0
            while x % p == 0:
                x //= p
                cnt += 1
            factors[p] = cnt
    return factors

# Precompute factorizations
_FACTS = {n: _factorize(n) for n in range(2, 41)}

def _subset_products_distinct(nums):
    """Check if all 2^k subset products of nums are distinct."""
    k = len(nums)
    if k == 0:
        return True
    products = set()
    for mask in range(1 << k):
        prod = 1
        for i in range(k):
            if mask & (1 << i):
                prod *= nums[i]
        if prod in products:
            return False
        products.add(prod)
    return True

def _greedy_build():
    """Greedy construction: start with primes, add composites that preserve distinctness."""
    current = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    used = set(current)
    
    # Candidates: all composites in [4, 40]
    candidates = [n for n in range(4, 41) if n not in used]
    
    # Sort by a heuristic: prefer numbers with fewer shared prime factors with current set
    def shared_count(n):
        sf = _FACTS[n]
        count = 0
        for p in sf:
            for c in current:
                if p in _FACTS[c]:
                    count += 1
                    break
        return count
    
    candidates.sort(key=shared_count)
    
    for cand in candidates:
        if len(current) >= 18:
            break
        trial = current + [cand]
        if _subset_products_distinct(trial):
            current.append(cand)
            used.add(cand)
    
    return current

def _backtracking_search(target_size=18, deadline=None):
    """Systematic backtracking search for a set of target_size with distinct subset products."""
    all_nums = list(range(2, 41))
    
    # Precompute for speed: for each pair, check if they can coexist
    # We'll use a recursive search with pruning
    
    best_found = []
    
    def _search(start_idx, chosen, remaining_needed):
        if deadline is not None and time.time() > deadline:
            return False
        if remaining_needed == 0:
            best_found.extend(chosen)
            return True
        if start_idx >= len(all_nums):
            return False
        # Pruning: not enough numbers left
        if len(all_nums) - start_idx < remaining_needed:
            return False
        
        # Try including all_nums[start_idx]
        cand = all_nums[start_idx]
        trial = chosen + [cand]
        if _subset_products_distinct(trial):
            if _search(start_idx + 1, trial, remaining_needed - 1):
                return True
        
        # Try excluding
        return _search(start_idx + 1, chosen, remaining_needed)
    
    _search(0, [], target_size)
    return best_found if best_found else None

def construct():
    deadline = time.time() + 50  # 50s safety margin before 60s limit
    
    # First try greedy
    result = _greedy_build()
    
    # If greedy got to 18, we're done
    if len(result) >= 18:
        return [int(x) for x in result[:18]]
    
    # Otherwise, try backtracking to find a set of size 18
    if time.time() < deadline:
        bt_result = _backtracking_search(18, deadline)
        if bt_result and len(bt_result) >= 18:
            return [int(x) for x in bt_result[:18]]
    
    # If we couldn't find 18, return what we have (shouldn't happen, but safety)
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
