import time
from itertools import combinations

# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    candidates = list(range(2, 41))
    
    # We want to find the max size k such that there exists a subset of size k
    # with distinct subset products.
    
    # Helper to check if adding x to current set (with products P) is valid
    def is_valid_add(x, P):
        # Check if x * p in P for any p in P
        # If x * p1 == p2, then collision.
        # Note: P contains 1.
        # If x in P, then x*1 = x, so x in P is a collision with 1?
        # No, x*1 = x. If x is in P, then x (from x*1) equals x (in P).
        # So yes, x must not be in P.
        if x in P:
            return False
        for p in P:
            if x * p in P:
                return False
        return True

    def add_product(x, P):
        return P | {x * p for p in P}

    # DFS to find max size
    best_size = 0
    best_set = []
    
    # Heuristic: Try to build the set.
    # Sort candidates? 
    # Primes are "safe". Composites are "risky".
    # Maybe sort by value?
    
    # We can use a recursive search.
    # To speed up, we can try to find a solution of size K, then K+1, etc.
    # Or just DFS and track max.
    
    # Given 39 items, DFS might be slow if not pruned.
    # Pruning: If current_size + remaining_count <= best_size, prune.
    
    def dfs(idx, current_set, current_products):
        nonlocal best_size, best_set
        
        # Deadline check
        if time.time() > deadline:
            return
        
        # Update best
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
        # Pruning
        if idx == len(candidates):
            return
            
        # Option 1: Skip candidates[idx]
        dfs(idx + 1, current_set, current_products)
        
        # Option 2: Include candidates[idx]
        x = candidates[idx]
        if is_valid_add(x, current_products):
            new_products = add_product(x, current_products)
            current_set.append(x)
            dfs(idx + 1, current_set, new_products)
            current_set.pop()

    # To improve search, we can order candidates.
    # Numbers with fewer prime factors (primes) are less likely to conflict?
    # Or numbers that are "large" are less likely to be products of others?
    # Let's try sorting by value descending? Or ascending?
    # Ascending: 2, 3, 4... 2 is very restrictive.
    # Descending: 40, 39... 40 is $2^3 \cdot 5$.
    
    # Let's try a few orderings if time permits, or just one good one.
    # Primes first?
    # If we pick all primes, we get 12.
    # If we pick composites, we might get more.
    
    # Let's sort by number of prime factors (with multiplicity) ascending?
    # Primes have 1.
    # Semi-primes have 2.
    # This might group "simple" numbers together.
    
    def count_prime_factors(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count

    # Sort by number of prime factors, then by value
    candidates.sort(key=lambda x: (count_prime_factors(x), x))
    
    dfs(0, [], {1})
    
    return [int(x) for x in best_set]
# EVOLVE-BLOCK-END
