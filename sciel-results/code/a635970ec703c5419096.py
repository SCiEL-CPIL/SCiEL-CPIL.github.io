"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random

N = 5000

# EVOLVE-BLOCK-START

def _primes(limit):
    """Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p::p] = bytearray(len(sieve[p * p::p]))
    return [p for p in range(2, limit + 1) if sieve[p]]


def _is_admissible(points, primes):
    """Check if the tuple of points is admissible w.r.t. the given primes.
    Admissible means: for every prime p in primes, not all residue classes mod p are covered."""
    point_set = set(points)
    for p in primes:
        covered = set()
        for v in points:
            covered.add(v % p)
            if len(covered) == p:
                # All classes covered -> not admissible
                return False
        # If we reach here, not all classes are covered, which is fine
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    # We want the largest admissible tuple in [0, N].
    # The primes to consider are those p <= len(tuple).
    # Since len(tuple) <= N+1, the relevant primes are those up to N+1.
    # However, the condition is "for every prime p <= len(tuple)".
    # As we remove elements, len(tuple) shrinks, so the set of relevant primes shrinks.
    
    # Strategy: Start with all integers [0, N].
    # Iteratively find a prime p <= current_len such that all residue classes mod p are covered.
    # If such a p exists, we must remove at least one element from each "excess" class? No.
    # To make it admissible for p, we need to ensure at least one residue class mod p is empty.
    # This means we must remove ALL elements from at least one residue class mod p.
    # To maximize the final size, we should choose the residue class (for some violating prime p)
    # with the minimum number of elements to remove.
    
    # This is a greedy set cover / hitting set variant.
    # We start with the full set.
    points = list(range(N + 1))
    
    # Precompute primes up to N+1
    primes = _primes(N + 1)
    
    # To speed up, we can maintain the count of elements in each residue class for each prime.
    # But updating this after removals is O(N) per removal if done naively.
    # Given N=5000, and potentially many removals, we need efficiency.
    
    # Let's try a simpler approach first:
    # 1. Start with all points.
    # 2. Loop:
    #    a. Determine current length L.
    #    b. Get relevant primes: p <= L.
    #    c. For each relevant prime p, check if all classes are covered.
    #    d. If no prime violates, we are done. Return points.
    #    e. If multiple primes violate, pick the one that requires the fewest removals to fix.
    #       To fix prime p, we must empty one of its residue classes.
    #       The cost is the size of the smallest residue class for that p.
    #       We pick the (p, class) pair with the minimum size.
    #    f. Remove all points in that class.
    
    # Optimization:
    # Instead of checking all primes from scratch each time, we can be smart.
    # However, with N=5000, the number of primes is ~670.
    # Checking coverage for one prime is O(N).
    # Total cost per iteration ~ 670 * 5000 = 3.3M operations.
    # Number of iterations? Each iteration removes at least 1 element.
    # Worst case 5000 iterations -> 16.5B operations. Too slow in Python.
    
    # We need a faster method.
    
    # Alternative:
    # The condition "all classes covered" is equivalent to the set of points being a "complete residue system" mod p.
    # We want to remove the minimum number of points to break completeness for all small primes.
    
    # Let's use a different perspective.
    # The "seed" solution starts with 1 and primes > sqrt(N).
    # Why? Because for any prime p <= sqrt(N), the set {1, p1, p2, ...} where pi > sqrt(N)
    # does not necessarily cover all classes mod p.
    # Actually, the seed solution is known to be admissible.
    # Let's analyze the seed solution size.
    # Primes > sqrt(5000) ~ 70.7.
    # Number of primes between 71 and 5000 is approx 5000/ln(5000) - 70/ln(70) ~ 5000/8.5 - 70/4.2 ~ 588 - 16 ~ 572.
    # Plus 1. So size ~ 573.
    
    # Can we do better?
    # The maximum size of an admissible tuple in [0, N] is related to the Jacobsthal function.
    # But we are maximizing the size of the subset, not the gap.
    # This is equivalent to finding the maximum size of a set S subset [0, N] such that for all p <= |S|, S is not complete mod p.
    
    # Let's try a greedy removal algorithm with optimization.
    # We maintain a list of active points.
    # We maintain a dictionary: prime -> set of covered classes.
    # When we remove a point x, we update the covered classes for all primes p <= N.
    # This is O(pi(N)) per removal. pi(5000) ~ 670.
    # Total removals: worst case 5000.
    # Total ops: 5000 * 670 = 3.35M. This is fast!
    
    # Algorithm:
    # 1. points = list(range(N+1))
    # 2. active = set(points)
    # 3. For each prime p <= N, compute counts of each residue class in active.
    # 4. Loop:
    #    a. L = len(active)
    #    b. Find all primes p <= L such that all classes are covered (i.e., min count > 0? No, we need to know if ANY class has count 0).
    #       Wait, "all classes covered" means count[c] > 0 for all c in 0..p-1.
    #       So we need to check if min(counts) > 0.
    #    c. If no such p, break.
    #    d. Among violating primes, find the one where we can remove the fewest points to make one class empty.
    #       For a violating prime p, the cost to fix is min(counts for p).
    #       We choose the p with the smallest min(counts).
    #       If tie, choose any.
    #    e. Let c* be the class with min count for p*.
    #    f. Remove all x in active such that x % p* == c*.
    #    g. Update active and counts.
    
    # This should be efficient.
    
    # Implementation details:
    # We need to efficiently find which points to remove.
    # We can store points in a list, and a set for O(1) lookup.
    # To update counts, we iterate over the removed points.
    
    # Let's refine the "find violating prime" step.
    # We need to check all primes p <= L.
    # We can precompute the list of primes.
    # In each iteration, we scan primes from smallest to largest?
    # Actually, we want the one with the smallest min_count.
    # So we can scan all relevant primes, compute min_count, and track the best (p, class) to remove.
    
    # Let's implement this.
    
    primes_list = _primes(N + 1)
    
    # Initialize counts
    # counts[p] is a list of length p, counts[p][c] = number of x in active with x % p == c
    counts = {}
    for p in primes_list:
        counts[p] = [0] * p
        for x in range(N + 1):
            counts[p][x % p] += 1
            
    active = set(range(N + 1))
    # We also need a way to quickly find points to remove.
    # We can just iterate over active if the set is small, but initially it's 5001.
    # Better: maintain a list of points? Or just use the set and filter?
    # Filtering a set of size 5000 is O(5000).
    # Number of iterations is bounded by the number of removals.
    
    # Let's trace the cost.
    # Initialization: O(N * pi(N)) ~ 5000 * 670 = 3.35M. OK.
    # Each iteration:
    #   1. Scan primes p <= L. There are ~ L/ln(L) primes.
    #      For each, compute min(counts[p]). O(p).
    #      Total scan cost: Sum_{p<=L} p ~ L^2 / (2 ln L).
    #      For L=5000, 5000^2 / 17 ~ 1.5M.
    #   2. Remove points. Let k be the number removed.
    #      For each removed point x, update counts for all primes p.
    #      Cost: k * pi(N).
    
    # Total cost might be high if we do this naively in Python.
    # 1.5M operations per iteration * 5000 iterations = 7.5B. Too slow.
    
    # We need to optimize the "find best prime" step.
    # Observation: We only need to consider primes p <= L.
    # Also, if a prime p is not violating (i.e., some class is already empty), we don't care about it.
    # We can maintain a list of "violating primes".
    # Initially, all primes p <= N+1 are likely violating (since we have all numbers 0..N, and N >= p-1, so all classes are covered).
    # Actually, for p > N+1, we can't have all classes covered.
    # For p <= N+1, initially all classes are covered.
    
    # When we remove points, some classes might become empty.
    # If a class becomes empty for prime p, then p is no longer violating.
    # We can maintain a set of violating primes.
    
    # Let's try this approach.
    
    violating = set(p for p in primes_list if p <= N + 1)
    # Actually, all p <= N+1 are violating initially.
    
    # To find the best prime to fix, we need min_count for each violating prime.
    # We can compute min_count on the fly.
    
    # Is there a faster way?
    # Maybe we can just pick the smallest violating prime?
    # The smallest prime p has the smallest p, so the cost to empty a class is likely small?
    # Not necessarily. The distribution of points matters.
    # But usually, smaller primes have more balanced distributions?
    # Let's try picking the violating prime with the smallest p first.
    # This is a heuristic.
    
    # Let's try the heuristic: pick the smallest violating prime p.
    # Then pick the class c with min count for that p.
    # Remove all x in active with x % p == c.
    
    # This simplifies the search to just finding the smallest violating prime.
    # We can maintain a sorted list or just scan from 2 upwards.
    
    # Let's implement this simpler version and see if it's fast enough.
    
    active = set(range(N + 1))
    # We need to update counts efficiently.
    # Instead of updating all counts for all primes, we can just recompute min_count for the smallest violating prime?
    # No, we need to know if it's still violating.
    
    # Let's stick to the full update but optimize the inner loop.
    # Using NumPy might help?
    # "Python standard library, NumPy and SciPy are available."
    
    # Let's try a NumPy-based approach for speed.
    # Represent active as a boolean array of size N+1.
    # For each prime p, we can compute counts using np.bincount or similar.
    # But updating after removals is still tricky.
    
    # Alternative:
    # Since N is small (5000), maybe pure Python is fast enough if we are careful.
    # Let's estimate the number of iterations.
    # Each iteration removes at least 1 point.
    # In practice, it removes many more.
    # For p=2, we remove ~N/2 points.
    # For p=3, we remove ~N/3 points.
    # The number of iterations is likely small (logarithmic in N?).
    
    # Let's try the simple Python implementation with the "smallest violating prime" heuristic.
    
    active = set(range(N + 1))
    # Precompute residues for each point? No, too much memory.
    # Just compute x % p on the fly.
    
    # To speed up "find smallest violating prime":
    # We can iterate p from 2 to L.
    # Check if p is in violating set.
    # If yes, check if it's still violating.
    # To check if it's violating, we need to see if all classes are covered.
    # We can do this by iterating over active points? No, that's O(N).
    # We need the counts.
    
    # Let's maintain counts[p] as a list.
    # Updating counts[p] for all p when removing a set of points R:
    # For x in R:
    #   For p in primes_list:
    #     counts[p][x % p] -= 1
    # This is O(|R| * pi(N)).
    
    # If |R| is large (e.g., 2500), and pi(N) is 670, that's 1.6M updates.
    # If we do this for 10 iterations, that's 16M updates. This is fast in Python.
    
    # The bottleneck is finding the best prime.
    # If we scan all primes p <= L and compute min(counts[p]), that's O(L * avg_p).
    # But we can just check the smallest violating prime.
    # We maintain a sorted list of violating primes?
    # Or just scan from 2 upwards until we find one that is still violating.
    
    # Let's try this.
    
    counts = {}
    for p in primes_list:
        counts[p] = [0] * p
        for x in range(N + 1):
            counts[p][x % p] += 1
            
    # We assume all p <= N+1 are violating initially.
    # We'll maintain a set of primes that are no longer violating.
    non_violating = set()
    
    while True:
        if time.time() > deadline:
            break
            
        L = len(active)
        if L == 0:
            break
            
        # Find the smallest prime p <= L that is violating.
        # A prime p is violating if p not in non_violating AND min(counts[p]) > 0.
        # Wait, if p is in non_violating, it means some class was empty at some point.
        # Since we only remove points, classes remain empty. So non_violating is sticky.
        
        best_p = None
        best_class = None
        best_count = float('inf')
        
        # We want to minimize the number of removals.
        # So we should pick the (p, c) with the smallest counts[p][c] among all violating p.
        # Let's scan all p <= L that are not in non_violating.
        
        for p in primes_list:
            if p > L:
                break
            if p in non_violating:
                continue
            # Check if p is still violating
            # min_count = min(counts[p])
            # If min_count > 0, it is violating.
            # We want to find the global min across all violating p.
            
            # Optimization: if min(counts[p]) >= best_count, we can skip?
            # No, we need the min for this p to know if it's violating.
            # But we can compute min_count and compare.
            
            min_count = min(counts[p])
            if min_count > 0:
                # It is violating
                if min_count < best_count:
                    best_count = min_count
                    best_p = p
                    # Find the class with this min count
                    # There might be multiple, pick the first one
                    best_class = counts[p].index(min_count)
            else:
                # Not violating anymore
                non_violating.add(p)
                
        if best_p is None:
            # No violating primes
            break
            
        # Remove all x in active such that x % best_p == best_class
        to_remove = [x for x in active if x % best_p == best_class]
        
        if not to_remove:
            # Should not happen if best_count > 0
            break
            
        # Update active
        for x in to_remove:
            active.remove(x)
            
        # Update counts
        for x in to_remove:
            for p in primes_list:
                counts[p][x % p] -= 1
                # Check if this makes p non-violating?
                # We can check later in the main loop.
                # But we can also check now: if counts[p][x % p] == 0, then p is non-violating.
                # However, we only care about p <= L.
                # Let's just update the count. The main loop will re-evaluate.
                
        # Optional: update non_violating set immediately for efficiency?
        # We can check for p in primes_list if any count is 0.
        # But that's O(pi(N) * p) which is expensive.
        # Let's rely on the main loop scan.
        
    return list(active)

# EVOLVE-BLOCK-END
