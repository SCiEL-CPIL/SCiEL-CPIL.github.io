import time
import math

N = 5000
M = N + 1  # Range is [0, 5000], so size is 5001

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

# Precompute primes
PRIMES = _primes(M)

# Precompute classes for each prime
# classes[p] = list of lists, where classes[p][r] is the set of x in [0, M-1] with x % p == r
CLASSES = {}
for p in PRIMES:
    cls = [[] for _ in range(p)]
    for x in range(M):
        cls[x % p].append(x)
    CLASSES[p] = cls

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    # S is the set of removed elements
    # Using a set for O(1) lookup
    S = set()
    
    # To speed up cost calculation, we can maintain counts of how many elements
    # of each class are already in S.
    # But S changes.
    # Let's just recalculate or update incrementally.
    
    # We need to find the "best" (p, r) to add.
    # Best means minimum |C_{p,r} \ S|.
    
    # Initial state: S is empty.
    # We need to satisfy all primes p <= M.
    
    # Let's iterate.
    while True:
        if time.time() > deadline:
            break
            
        # Determine active primes
        # p is active if p <= M - len(S) and C_{p,r} is not subset of S for any r
        active_primes = []
        for p in PRIMES:
            if p > M - len(S):
                continue
                
            # Check if any class is already covered
            covered = False
            for r in range(p):
                # Check if all x in CLASSES[p][r] are in S
                # This is expensive if done naively.
                # Optimization: if len(S) is large, this is likely true.
                # But we can just check if any x in CLASSES[p][r] is NOT in S.
                # If there is at least one x not in S, the class is not covered.
                # We want to know if ALL are in S.
                is_covered = True
                for x in CLASSES[p][r]:
                    if x not in S:
                        is_covered = False
                        break
                if is_covered:
                    covered = True
                    break
            
            if not covered:
                active_primes.append(p)
        
        if not active_primes:
            break
            
        # For each active prime, find the best residue r
        best_p = None
        best_r = None
        min_cost = float('inf')
        
        for p in active_primes:
            for r in range(p):
                # Cost is number of elements in CLASSES[p][r] not in S
                cost = 0
                for x in CLASSES[p][r]:
                    if x not in S:
                        cost += 1
                        # Early exit if cost exceeds current min
                        if cost >= min_cost:
                            break
                if cost < min_cost:
                    min_cost = cost
                    best_p = p
                    best_r = r
                    # If cost is 0, we can't do better (but it shouldn't be 0 if active)
                    # Actually, if cost is 0, it means the class is covered, which contradicts 'active'.
                    # So cost >= 1.
        
        if best_p is None:
            break
            
        # Add the elements of the best class to S
        for x in CLASSES[best_p][best_r]:
            S.add(x)
            
    # Construct the result
    result = [x for x in range(M) if x not in S]
    return result
