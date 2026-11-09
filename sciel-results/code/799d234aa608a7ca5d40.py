import time
from math import isqrt

def _is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, isqrt(n) + 1, 2):
        if n % i == 0:
            return False
    return True

def _can_add(x, S_list):
    """Check if x can be added to S_list maintaining validity."""
    n = len(S_list)
    # x as 'a': x ∤ b*c for all distinct b,c in S
    for i in range(n):
        for j in range(i+1, n):
            if (S_list[i] * S_list[j]) % x == 0:
                return False
    # x as 'b' or 'c': a ∤ x*c and a ∤ c*x for all distinct a,c in S
    for i in range(n):
        for j in range(i+1, n):
            if (x * S_list[j]) % S_list[i] == 0:
                return False
            if (x * S_list[i]) % S_list[j] == 0:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    all_nums = list(range(2, 257))
    
    # Heuristics for ordering candidates
    # 1. Descending value (large numbers are less likely to divide products of smaller ones)
    # 2. Ascending value
    # 3. Primes first, then composites descending
    # 4. Primes first, then composites ascending
    
    candidates_orders = [
        sorted(all_nums, reverse=True),
        sorted(all_nums),
        sorted(all_nums, key=lambda x: (0 if _is_prime(x) else 1, -x)),
        sorted(all_nums, key=lambda x: (0 if _is_prime(x) else 1, x)),
    ]
    
    best_set = []
    
    for order in candidates_orders:
        if time.time() > deadline - 5:
            break
            
        S = []
        S_set = set()
        
        for x in order:
            if time.time() > deadline - 5:
                break
            if _can_add(x, S):
                S.append(x)
                S_set.add(x)
        
        if len(S) > len(best_set):
            best_set = S
            # If we found a set larger than the number of primes, we might be able to stop early
            # or continue to see if we can do better.
            # The number of primes in 2..256 is 54.
            if len(best_set) > 54:
                # Try a few more simple local improvements if time permits
                # But for now, just record it.
                pass
    
    # If best_set is empty (should not happen), fall back to primes
    if not best_set:
        best_set = [x for x in all_nums if _is_prime(x)]
        
    return best_set
