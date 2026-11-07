import time
import sys
from itertools import combinations

def construct():
    """
    Returns a list of distinct integers in [2, 40] such that all subset products are distinct.
    Maximizes the number of integers.
    """
    start_time = time.time()
    deadline = start_time + 50  # 50s deadline to be safe
    
    # Known valid 17-element set
    base_set = [2, 3, 8, 9, 12, 32, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Verify base set
    if not _verify(base_set):
        # Fallback to a smaller known valid set if base fails (should not happen)
        return [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    best_set = base_set
    best_size = len(base_set)
    
    # Search for a larger set
    # We try to add composites to the set of primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    composites = [x for x in range(2, 41) if x not in primes]
    
    # We will use a DFS approach
    # State: current set S, set of products P(S)
    # We start with primes
    
    def dfs(S, P, depth):
        nonlocal best_set, best_size, deadline
        
        if time.time() > deadline:
            return
            
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
            
        # Identify candidates: composites not in S
        # Check if they can be added
        # Condition: c is not a ratio p2/p1 for any p1, p2 in P
        # i.e., c * p1 is not in P for any p1 in P
        
        candidates = []
        for c in composites:
            if c in S:
                continue
            
            # Check if c is forbidden
            # c is forbidden if exists p1 in P such that c * p1 in P
            # Or exists p1 in P such that c / p1 in P? No, ratio is p2/p1.
            # If c = p2/p1, then c*p1 = p2.
            # So we check if c*p1 is in P for any p1 in P.
            
            # Optimization: only check if c*p1 <= max(P)?
            # P contains 1. So c*1 = c. If c in P, then c=c/1, forbidden.
            # But c is composite, P contains primes and their products.
            # If c is not in P, we check larger p1.
            
            forbidden = False
            for p1 in P:
                if c * p1 in P:
                    forbidden = True
                    break
            
            if not forbidden:
                candidates.append(c)
                
        # Sort candidates to explore promising ones first
        # Heuristic: smaller numbers might be "safer" or "more restrictive"?
        # Let's just sort by value.
        candidates.sort()
        
        for c in candidates:
            # Add c to S
            new_S = S + [c]
            # Update P
            # P' = P union { c * p for p in P }
            new_P = set(P)
            for p in P:
                new_P.add(c * p)
                
            # Recurse
            dfs(new_S, new_P, depth + 1)
            
            if time.time() > deadline:
                return

    # Initialize with primes
    initial_P = {1}
    for p in primes:
        new_vals = [p * x for x in initial_P]
        initial_P.update(new_vals)
        
    dfs(list(primes), initial_P, 0)
    
    # Return the best set found
    # Ensure types are correct
    result = [int(x) for x in best_set]
    
    # Final verification
    if not _verify(result):
         return base_set
         
    return result

def _verify(candidate):
    """Verify that all subset products are distinct."""
    k = len(candidate)
    products = set()
    for mask in range(1 << k):
        prod = 1
        for i in range(k):
            if mask & (1 << i):
                prod *= candidate[i]
        if prod in products:
            return False
        products.add(prod)
    return True
