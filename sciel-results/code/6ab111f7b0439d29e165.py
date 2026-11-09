import heapq
import math

N = 5000

def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def construct():
    S = set(range(N + 1))
    # We will iteratively remove residue classes.
    # Strategy: Always remove the class that costs the least (smallest count)
    # among all primes that are currently "saturated" (all classes non-empty).
    # To be efficient, we re-evaluate the "cost" for each prime after each removal.
    
    # Optimization: We only need to check primes up to len(S).
    # We use a priority queue to pick the "best" prime to fix next.
    # Key: min_count for that prime. Lower is better (cheaper to fix).
    
    changed = True
    while changed:
        changed = False
        k = len(S)
        if k <= 1:
            break
            
        primes_to_check = _primes(k)
        
        # Find the prime with the minimum "cost" to satisfy admissibility
        # Cost = min number of elements in any residue class.
        # We want to pick the prime where this cost is lowest.
        
        best_p = None
        best_cost = float('inf')
        best_residue = None
        
        for p in primes_to_check:
            # Count elements in each residue class mod p
            classes = [0] * p
            for v in S:
                classes[v % p] += 1
            
            min_count = min(classes)
            
            # If all classes are non-empty, we MUST remove one.
            if min_count > 0:
                # This prime is currently violating the condition.
                # We compare the cost (min_count) to find the cheapest fix.
                if min_count < best_cost:
                    best_cost = min_count
                    best_p = p
                    # Find the residue with this min count
                    # If there are ties, pick the first one (arbitrary)
                    best_residue = classes.index(min_count)
                    
        if best_p is not None:
            # Remove all elements in S that are congruent to best_residue mod best_p
            S = {v for v in S if v % best_p != best_residue}
            changed = True
            
    return list(S)
