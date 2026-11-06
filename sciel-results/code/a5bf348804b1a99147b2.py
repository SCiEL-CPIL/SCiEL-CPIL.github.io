"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

N = 5000
_DEADLINE = 300.0  # safety margin before 330s kill

# EVOLVE-BLOCK-START
def _primes(limit):
    """Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = 0
    return [p for p in range(2, limit + 1) if sieve[p]]


def construct():
    """
    Construct the largest admissible tuple within [0, N].
    
    Strategy:
    1. Start with the full set [0, N].
    2. Iteratively find a prime p <= current_length such that all residue classes mod p are covered.
    3. If such a prime exists, remove the least-populated residue class for that prime.
       To maximize efficiency and minimize removals, we prioritize the prime that requires
       the fewest removals (i.e., the prime with the smallest minimum class count).
       If multiple primes have the same min count, we might prefer larger primes? 
       Actually, removing a class for a small prime removes ~N/p elements.
       Removing a class for a large prime removes ~N/p elements.
       The cost is the size of the class.
       We want to minimize the total number of elements removed.
       Greedy: pick the (p, class) pair that minimizes the number of elements to remove.
       
    Optimization:
    - Maintain counts of each residue class for each prime.
    - When removing elements, update counts for all primes.
    - Use a priority queue or simply scan for the best candidate each iteration.
    - Since N=5000 and number of primes is ~670, scanning is feasible if done efficiently.
    """
    start_time = time.time()
    
    # Precompute primes up to N+1
    primes = _primes(N + 1)
    
    # Initialize points as a set for O(1) lookup and removal
    # But we need to iterate and check residues. A list is better for iteration, but removal is O(N).
    # Given the potential number of iterations, let's use a list and a set for membership.
    # Or better, just use a list and filter it.
    
    points = list(range(N + 1))
    
    # Precompute residue classes for each prime? 
    # For each prime p, we need to know which points have which residue.
    # Let's maintain a dictionary: prime -> list of counts for each residue.
    # And a mapping: point -> list of (prime, residue) ? No, that's too much.
    
    # Let's use the approach from Inspiration 2's detailed plan:
    # 1. points = list(range(N+1))
    # 2. active = set(points)
    # 3. For each prime p <= N, compute counts of each residue class in active.
    #    counts[p][r] = number of x in active such that x % p == r
    
    # To make updates fast, we need to know for each point x, its residue mod p for all p.
    # This is expensive to precompute for all p.
    # Alternative: When we decide to remove a class (p, r), we iterate through all points,
    # check if x % p == r, and if so, remove it.
    # This is O(N) per removal step.
    # Number of steps? Each step removes at least 1 point. Max N steps.
    # Total O(N^2) = 25M operations. This is fine in Python within 300s?
    # 25M simple operations might take ~2-5 seconds.
    
    # However, we also need to find the "best" prime to fix.
    # This requires checking all primes p <= len(points) to see if they are "full" (all classes covered).
    # And finding the min class count.
    # If we recompute counts from scratch each time, it's O(N * pi(N)) per iteration.
    # pi(5000) ~ 670. N=5000. 3.3M ops per iteration.
    # If we have 100 iterations, that's 330M ops. Might be too slow.
    
    # Let's try to maintain counts incrementally.
    # counts = {p: [0]*p for p in primes}
    # Initialize:
    counts = {}
    for p in primes:
        c = [0] * p
        for x in points:
            c[x % p] += 1
        counts[p] = c
        
    # Now, in each iteration:
    # 1. L = len(points)
    # 2. Find all primes p <= L such that min(counts[p]) > 0 (all classes covered).
    # 3. Among these, find the one with the smallest min(counts[p]).
    #    Let this be p*. Let r* be the index of the min count.
    # 4. If no such prime, break.
    # 5. Remove all x in points such that x % p* == r*.
    # 6. Update points.
    # 7. Update counts: For each removed x, for each prime p in primes:
    #    counts[p][x % p] -= 1
    #    Note: We only need to update counts for primes p <= N (all of them in our list).
    #    But wait, if p > L, we don't care about it? 
    #    The condition is "for every prime p <= len(tuple)".
    #    So if p > L, it's not checked.
    #    However, if we remove points, L decreases.
    #    So primes that were > L might become <= L? No, L decreases, so the set of relevant primes shrinks.
    #    So we only need to track counts for primes that are currently <= L?
    #    Actually, we can just track counts for all primes up to N.
    #    The check is: for p in primes: if p <= L and min(counts[p]) > 0: ...
    
    # Let's refine the update step.
    # When we remove a set of points R, for each x in R, for each p in primes:
    # counts[p][x % p] -= 1
    # This is O(|R| * pi(N)).
    # |R| is the size of the removed class.
    
    # Let's implement this.
    
    active_set = set(points)
    
    changed = True
    while changed:
        if time.time() - start_time > _DEADLINE - 5:
            break
        
        L = len(points)
        best_p = None
        best_r = None
        best_cost = float('inf')
        
        # Find the prime p <= L that is "full" and has the smallest min class count
        for p in primes:
            if p > L:
                break
            # Check if all classes are covered
            min_c = min(counts[p])
            if min_c > 0:
                # This prime is full. Cost to fix is min_c.
                if min_c < best_cost:
                    best_cost = min_c
                    best_p = p
                    # Find the residue class with min count
                    best_r = counts[p].index(min_c)
                    # If there are multiple, index returns the first. That's fine.
        
        if best_p is None:
            # No prime needs fixing
            break
        
        # Remove all points x such that x % best_p == best_r
        # We need to identify these points.
        to_remove = []
        for x in points:
            if x % best_p == best_r:
                to_remove.append(x)
        
        # Update points
        remove_set = set(to_remove)
        points = [x for x in points if x not in remove_set]
        
        # Update counts
        for x in to_remove:
            for p in primes:
                counts[p][x % p] -= 1
        
        changed = True
        # Optional: if best_cost == 0, something is wrong, but we checked min_c > 0.
        
    return points
# EVOLVE-BLOCK-END
