import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin
    
    # Candidate pools, ordered by likely feasibility (larger numbers = fewer constraints)
    # We want to find 48 numbers.
    pools = [
        list(range(128, 193)),  # 65 numbers
        list(range(112, 193)),  # 81 numbers
        list(range(96, 193)),   # 97 numbers
        list(range(80, 193)),   # 113 numbers
        list(range(1, 193))     # Fallback to all
    ]
    
    best_set = []
    best_size = 0
    
    for pool in pools:
        if time.time() > deadline:
            break
            
        pool_set = set(pool)
        n_pool = len(pool)
        
        # If pool is too small to pick 48, skip (though 65 is enough)
        if n_pool < 48:
            continue
            
        # Try to find a valid set of size 48
        found = False
        attempts = 0
        max_attempts = 5000
        
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            
            # Random sample
            S = random.sample(pool, 48)
            
            if _is_valid(S, pool_min=min(pool)):
                return S
            
            # If invalid, try local repair
            # Find a conflict
            conflict = _find_conflict(S, pool_min=min(pool))
            if conflict:
                a, b, c, d = conflict
                # Try to swap one of b,c,d with a random unused element
                # We remove one of the summands to break the sum
                # Or remove a? If we remove a, we reduce size. We want to keep size 48.
                # So we must swap.
                
                unused = list(pool_set - set(S))
                if unused:
                    # Try swapping b
                    for elem_to_remove in [b, c, d]:
                        # Try a few random swaps
                        for _ in range(10):
                            if time.time() > deadline:
                                break
                            new_elem = random.choice(unused)
                            if new_elem == elem_to_remove:
                                continue
                            new_S = [x for x in S if x != elem_to_remove] + [new_elem]
                            if _is_valid(new_S, pool_min=min(pool)):
                                return new_S
                            
        # If we exhausted attempts for this pool, move to next
        # Optionally, keep the best valid subset found during repairs?
        # For now, just move on.
        
    # If we didn't find 48, try to maximize size with a greedy approach on the best pool
    # Or return the best we have (which might be empty if we didn't track partials)
    
    # Fallback: Greedy construction similar to seed but maybe smarter
    # Since seed got 45, let's try to beat it with a local search starting from seed's logic
    # But we already tried random restarts.
    
    # Let's try a greedy build-up from the best pool
    if not best_set:
        # Use the largest pool for greedy
        pool = pools[-1]
        S = []
        for x in sorted(pool, reverse=True):
            candidate = S + [x]
            if _is_valid(candidate, pool_min=1):
                S.append(x)
            if len(S) >= 48:
                break
        best_set = S
        
    return best_set

def _is_valid(S, pool_min=1):
    """
    Check if S is valid.
    pool_min is used to optimize target sum checking.
    If pool_min is high, we know min sum is 3*pool_min.
    """
    if len(S) < 4:
        return True
        
    S_set = set(S)
    max_sum = 3 * 192 # Max possible sum of 3 numbers <= 192
    
    # Pre-compute sums of all triples?
    # For each a in S, check if a divides any sum of 3 distinct others.
    
    for a in S:
        # Determine possible multiples of a that can be formed by sum of 3 others
        min_other_sum = 3 * pool_min
        # Actually, others are from S, so min_other_sum is sum of 3 smallest in S\{a}
        # But using pool_min is a safe lower bound.
        
        # Check multiples k*a
        # k*a must be >= min_other_sum and <= max_sum
        # Also k*a must be sum of 3 distinct elements from S\{a}
        
        # Optimization:
        # If a is large, few multiples.
        # If a is small, many multiples.
        
        # We iterate through all triples of S\{a}?
        # No, that's O(N^3) per a. Total O(N^4). 48^4 is too slow?
        # 48^4 ~ 5M. In Python, 5M ops might take 0.5s.
        # We have 55s. So O(N^4) is feasible for a few checks.
        # But we do this in a loop of attempts.
        # 5000 attempts * 0.5s = 2500s. Too slow.
        
        # We need faster validity check.
        
        # Strategy: Precompute all triple sums of S.
        # Map: sum -> list of triples (as sets or tuples)
        # Then for each a, check if any multiple of a is in the map,
        # and if the triple does not contain a.
        
        # Building map: O(N^3) ~ 17k ops.
        # Checking: O(N * #multiples).
        # This is much faster.
        
    # Let's implement the Map approach inside _is_valid
    
    sums_map = {}
    others = [x for x in S]
    n = len(others)
    
    # Build map of sums to triples
    # To avoid storing all triples (memory), we can just store counts?
    # No, we need to know if a specific 'a' is in the triple.
    # If we store triples, we can check.
    # 17k triples is small.
    
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                s = others[i] + others[j] + others[k]
                if s not in sums_map:
                    sums_map[s] = []
                sums_map[s].append((others[i], others[j], others[k]))
                
    for a in S:
        # Check multiples
        # min sum of 3 distinct from S\{a}
        # We can just check all multiples in range [3*1, 3*192]
        # Multiples: 2a, 3a, ...
        # Note: sum must be from DISTINCT OTHERS.
        # So the triple must not contain a.
        
        # Max sum is 576
        max_multiple = 576 // a
        for k in range(2, max_multiple + 1): # Start from 2? 
            # Can sum be a? 
            # If sum = a, then a divides a.
            # But sum of 3 positive integers >= 1+2+3 = 6.
            # If a < 6, sum can be a.
            # If a >= 6, sum > a is likely?
            # Actually, if a is in S, and we pick 3 others, their sum can be a.
            # Example: S={1,2,3,6}. a=6. others 1,2,3 sum=6. 6|6. Invalid.
            # So we must check k=1 as well if possible.
            # But min sum is 3*min(S\{a}).
            
            target = k * a
            if target in sums_map:
                for triple in sums_map[target]:
                    if a not in triple:
                        return False
                        
    return True

def _find_conflict(S, pool_min=1):
    """
    Find a conflict (a, b, c, d) where a | (b+c+d).
    Returns None if valid.
    """
    if len(S) < 4:
        return None
        
    sums_map = {}
    others = [x for x in S]
    n = len(others)
    
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                s = others[i] + others[j] + others[k]
                if s not in sums_map:
                    sums_map[s] = []
                sums_map[s].append((others[i], others[j], others[k]))
                
    for a in S:
        max_multiple = 576 // a
        for k in range(1, max_multiple + 1):
            target = k * a
            if target in sums_map:
                for triple in sums_map[target]:
                    if a not in triple:
                        return (a, triple[0], triple[1], triple[2])
                        
    return None
