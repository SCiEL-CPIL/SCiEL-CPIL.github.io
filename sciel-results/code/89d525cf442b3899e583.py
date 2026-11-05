"""Rewritten program: minimize the largest distinct factor > 12 with product 12!."""
# EVOLVE-BLOCK-START
import math
import time
import heapq
from functools import lru_cache

TARGET = math.factorial(12)
MIN_F = 13
MAX_FACTORS = 12
DEADLINE = 85.0  # safety margin before 90s kill


def _compute_divisors(n):
    """Compute all divisors of n."""
    divs = set([1])
    for p in range(2, int(math.isqrt(n)) + 1):
        if n % p == 0:
            while n % p == 0:
                n //= p
            new_divs = set()
            d = p
            while d <= n * p:
                for old_d in divs:
                    new_divs.add(old_d * d)
                d *= p
            divs |= new_divs
    return sorted(divs)


def _factorize(n):
    """Return prime factorization of n as dict {prime: exponent}."""
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1 if d == 2 else 2
    if n > 1:
        factors[n] = factors.get(n, 1)
    return factors


def construct():
    """Return 1..12 DISTINCT integers all greater than 12, with product exactly 12!.
    Minimize the LARGEST factor."""
    start = time.time()
    
    # Precompute divisors of TARGET
    all_divs = _compute_divisors(TARGET)
    
    # Filter: divisors > MIN_F
    valid_divs = [d for d in all_divs if d > MIN_F]
    valid_divs_set = set(valid_divs)
    n_valid = len(valid_divs)
    
    # Initial solution: [TARGET]
    best_max = TARGET
    best_factors = [TARGET]
    
    # Precompute lower bound products for pruning
    # lb_prod[i][k] = product of k smallest valid divisors starting at index i
    # If this exceeds remaining, we can prune
    max_k = MAX_FACTORS
    lb_prod = [[1] * (max_k + 1) for _ in range(n_valid + 1)]
    for i in range(n_valid, -1, -1):
        lb_prod[i][0] = 1
        for k in range(1, max_k + 1):
            if i + k > n_valid:
                lb_prod[i][k] = float('inf')
            else:
                prev = lb_prod[i][k-1]
                if prev == float('inf'):
                    lb_prod[i][k] = float('inf')
                else:
                    val = prev * valid_divs[i + k - 1]
                    lb_prod[i][k] = val if val <= TARGET else float('inf')
    
    def _can_complete(remaining, min_idx, k):
        """Check if remaining can be factored into k distinct divisors >= valid_divs[min_idx]."""
        if k == 0:
            return remaining == 1
        if remaining < 1:
            return False
        if min_idx >= n_valid:
            return False
        
        # Lower bound check
        if lb_prod[min_idx][k] > remaining:
            return False
        
        if remaining < valid_divs[min_idx]:
            return False
        
        # Check if remaining itself is a valid divisor (for k=1)
        if k == 1:
            return remaining in valid_divs_set and remaining >= valid_divs[min_idx]
        
        return True
    
    # A*-style search using min-heap ordered by current_max
    # Each entry: (current_max, min_idx, remaining, k, tuple_of_factors)
    # We want to find the minimum possible max factor
    
    def a_star_search():
        nonlocal best_max, best_factors
        
        # Start with empty factor set
        # Heap entries: (current_max, min_idx, remaining, k, factors_tuple)
        # current_max starts at MIN_F - 1 (no factors yet)
        heap = []
        heapq.heappush(heap, (MIN_F - 1, 0, TARGET, max_k, ()))
        
        # Also try all k values from 2 to max_k
        for k in range(2, max_k + 1):
            if lb_prod[0][k] <= TARGET:
                heapq.heappush(heap, (MIN_F - 1, 0, TARGET, k, ()))
        
        visited = set()
        
        while heap:
            if time.time() - start > DEADLINE:
                break
            
            cur_max, min_idx, remaining, k, factors = heapq.heappop(heap)
            
            # If we've already found a solution with this or better max, skip
            if cur_max >= best_max:
                break
            
            # State key for deduplication
            state_key = (min_idx, remaining, k)
            if state_key in visited:
                continue
            visited.add(state_key)
            
            if k == 0:
                if remaining == 1:
                    if cur_max < best_max:
                        best_max = cur_max
                        best_factors = list(factors)
                continue
            
            if remaining < 1 or min_idx >= n_valid:
                continue
            
            # Pruning: lower bound
            if lb_prod[min_idx][k] > remaining:
                continue
            
            if remaining < valid_divs[min_idx]:
                continue
            
            # Try each possible next factor
            for i in range(min_idx, n_valid):
                f = valid_divs[i]
                
                if f > remaining:
                    break
                
                if remaining % f != 0:
                    continue
                
                new_max = max(cur_max, f)
                if new_max >= best_max:
                    break
                
                new_remaining = remaining // f
                new_min_idx = i + 1
                new_k = k - 1
                
                if not _can_complete(new_remaining, new_min_idx, new_k):
                    continue
                
                new_factors = factors + (f,)
                heapq.heappush(heap, (new_max, new_min_idx, new_remaining, new_k, new_factors))
                
                # Early termination
                if best_max <= MIN_F + k + 5:
                    return
    
    a_star_search()
    
    return best_factors


# EVOLVE-BLOCK-END
