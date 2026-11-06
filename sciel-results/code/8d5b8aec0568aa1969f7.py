"""Rewritten program: Greedy 'Cheapest Constraint First' maximization for admissible tuples in [0, 5000]."""

N = 5000


def _primes(limit):
    if limit < 2:
        return []
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p::p] = bytearray(len(sieve[p * p::p]))
    return [p for p in range(2, limit + 1) if sieve[p]]


def construct():
    primes = _primes(N + 1)
    
    # Start with the full range [0, N]
    points = list(range(N + 1))
    n = len(points)
    
    # Initialize counts for all primes
    # counts[p] is a list of length p, counts[p][r] = number of x in points such that x % p == r
    counts = {}
    for p in primes:
        c = [0] * p
        for x in points:
            c[x % p] += 1
        counts[p] = c
        
    # We will iteratively remove points to satisfy constraints for primes p <= len(points)
    # Strategy: Always fix the constraint that is "cheapest" to fix (smallest min count among full primes)
    
    while True:
        n = len(points)
        if n == 0:
            break
            
        # Find the prime p <= n with the minimum "cost" to fix (min count > 0)
        # Cost is min(counts[p])
        # We want to minimize this cost.
        best_p = None
        best_r = None
        best_cost = float('inf')
        
        for p in primes:
            if p > n:
                break
            min_c = min(counts[p])
            if min_c > 0:
                # This prime is "full" (all residues present)
                if min_c < best_cost:
                    best_cost = min_c
                    best_p = p
                    best_r = counts[p].index(min_c)
                elif min_c == best_cost:
                    # Tie-breaker: Prefer larger prime to remove "sparser" points?
                    # Or just keep the first one. Let's prefer larger p.
                    if best_p is not None and p > best_p:
                        best_p = p
                        best_r = counts[p].index(min_c)

        if best_p is None:
            # All constraints satisfied
            break
        
        # Remove all points x such that x % best_p == best_r
        to_remove = [x for x in points if x % best_p == best_r]
        
        # Update points
        remove_set = set(to_remove)
        points = [x for x in points if x not in remove_set]
        
        # Update counts
        # We only need to update counts for primes q <= new_n
        # Because primes q > new_n will never be checked again.
        new_n = len(points)
        
        for x in to_remove:
            for q in primes:
                if q > new_n:
                    break
                counts[q][x % q] -= 1
                
    return points
