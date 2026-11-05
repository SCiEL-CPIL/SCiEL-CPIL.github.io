"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

N = 5000
DEADLINE = 320.0  # Safety margin before 330s kill

# EVOLVE-BLOCK-START
def _primes(limit):
    """Return list of primes up to limit using sieve."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = 0
    sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = 0
    return [p for p in range(2, limit + 1) if sieve[p]]

def construct():
    start_time = time.time()
    
    # We need primes up to N because the tuple size can approach N
    # and for each prime p <= len(tuple), we must miss a residue class.
    # If len(tuple) becomes large, we need primes up to that size.
    max_k = N
    primes = _primes(max_k)
    
    # Greedy construction
    # We maintain a set of points.
    # For each prime p, we track the counts of each residue class mod p.
    # A point x is valid to add if for all p in primes where p <= len(points) + 1,
    # the residue class x % p is not yet "full" (i.e., count < p).
    # Actually, the condition is: for every prime p <= len(tuple), some residue class is missed.
    # This means we cannot have a tuple that covers ALL residue classes mod p for any p <= len(tuple).
    # So, when adding a new point x to a tuple of size k, the new size is k+1.
    # We must ensure that for all p <= k+1, the new tuple does NOT cover all p residue classes.
    # This is equivalent to saying: for all p <= k+1, there exists at least one residue class mod p 
    # that has count 0 in the new tuple.
    # Before adding x, let counts_p[r] be the count of residue r mod p.
    # After adding x, counts_p[x % p] increases by 1.
    # The condition "some residue class is missed" means min(counts_p) == 0.
    # So, for each p <= k+1, we must check if min(counts_p) == 0 BEFORE adding x, 
    # AND if the residue class x % p was the only one with count 0, then after adding x, 
    # min(counts_p) might become > 0, which is invalid.
    
    # More simply:
    # A state is valid if for all p <= len(points), min(counts_p) == 0.
    # When we add x, len(points) becomes k+1.
    # We need to check validity for all p <= k+1.
    # For p > k, the previous state didn't care about p, but now we do.
    # Wait, the definition says "for every prime p <= len(tuple)".
    # So if len(tuple) is k, we check primes p <= k.
    # If we add a point, len becomes k+1. We must check primes p <= k+1.
    # This includes the new prime p = k+1 (if k+1 is prime) and potentially others if k+1 is not prime but we missed checking a prime p where k < p <= k+1? No, primes are discrete.
    # Basically, we maintain validity for current size k.
    # When adding x to get size k+1:
    # 1. For all primes p <= k: we already had min(counts_p) == 0. Adding x might make min(counts_p) > 0 if x % p was the only 0-count class.
    # 2. For primes p where k < p <= k+1: we need to establish min(counts_p) == 0.
    #    Since we start with empty counts for these p, min is 0. Adding x makes one class 1, others 0. So min is still 0.
    #    So the only risk is for p <= k where the 0-count class is filled.
    
    # Optimization:
    # Maintain counts for primes p <= current_size.
    # When size increases, we don't need to do anything for new primes p > old_size, 
    # because they start with all zeros, so min is 0. They only become dangerous later.
    
    # Let's track:
    # points: list of ints
    # counts: dict mapping prime p to list of counts for residues 0..p-1
    # valid_zero_classes: for each p, the number of residue classes with count 0.
    # If valid_zero_classes[p] > 0, then p is "safe".
    # If valid_zero_classes[p] == 0, then p is "full" (invalid).
    # We must maintain valid_zero_classes[p] > 0 for all p <= len(points).
    
    points = []
    # We will dynamically add primes as needed.
    # But we can pre-initialize for all primes up to N? 
    # N=5000, number of primes is ~669.
    # For each prime p, we have an array of size p.
    # Total memory: sum(p for p in primes) ~ sum of primes up to 5000.
    # Sum of primes up to N is approx N^2 / (2 log N).
    # 5000^2 / (2 * 8.5) ~ 25e6 / 17 ~ 1.5e6. This is fine.
    
    counts = {}
    zero_counts = {}
    
    for p in primes:
        counts[p] = [0] * p
        zero_counts[p] = p
        
    # We only need to consider primes p <= len(points) for the constraint.
    # But we can just check all primes p <= len(points).
    
    # Greedy strategy:
    # Try to add points in increasing order? Or random? Or specific heuristic?
    # Increasing order: 0, 1, 2, ...
    # If we add 0, 1, ..., k-1, then for p=2, residues are 0,1,0,1... 
    # If k is even, counts[2] = [k/2, k/2], zero_counts[2]=0. Invalid!
    # So we can't just take consecutive integers.
    
    # We need to skip numbers that would make a prime "full".
    # A number x is "bad" for current state if there exists a prime p <= len(points) 
    # such that zero_counts[p] == 1 and counts[p][x % p] == 0.
    # Because adding x would make zero_counts[p] become 0.
    
    # Also, if len(points) + 1 is a prime, we don't need to check it for "badness" 
    # because it starts with all zeros, and adding one element leaves p-1 zeros.
    # Wait, if p = len(points)+1, and we add x, counts[p][x%p] becomes 1. 
    # zero_counts[p] becomes p-1. Since p >= 2, p-1 >= 1. So it's safe.
    # So we only need to check primes p <= len(points).
    
    current_size = 0
    # We iterate through candidates 0 to N
    # To be faster, we might want to check candidates in an order that avoids collisions.
    # But let's start with a simple scan and see if it's fast enough.
    # N=5000 is small.
    
    # Precompute residues for each candidate?
    # candidates = list(range(N+1))
    
    # To make it "largest", we should try to include as many as possible.
    # A simple greedy scan might get stuck early.
    # Better: iterate through candidates, but if a candidate is bad, skip it.
    # This is equivalent to finding a maximal independent set in a hypergraph?
    
    # Let's try a randomized greedy approach or a specific ordering.
    # Actually, just scanning 0..N and skipping bad ones is a valid heuristic.
    # Let's see how many we get.
    
    # To improve, we can try multiple random permutations or random starts?
    # Given time limit, let's try a few strategies.
    # Strategy 1: Scan 0..N, skip bad.
    # Strategy 2: Random shuffle, pick valid.
    
    # Let's implement a helper to check if x is valid to add.
    
    def is_valid_to_add(x, k):
        # k is current size
        # Check primes p <= k
        # We need primes <= k.
        # We have list of all primes. We can find the index.
        # Since primes is sorted, we can binary search or just iterate.
        # Iterating over all primes and checking p <= k is O(pi(N)) which is ~600.
        # Doing this for each of 5000 candidates is 3 million ops. Very fast.
        
        for p in primes:
            if p > k:
                break
            if zero_counts[p] == 0:
                # This shouldn't happen if we maintain validity, but just in case
                return False
            if zero_counts[p] == 1:
                # Only one zero class left. If x falls into it, we lose it.
                r = x % p
                if counts[p][r] == 0:
                    return False
        return True

    def add_point(x):
        nonlocal current_size
        for p in primes:
            if p > current_size:
                break
            r = x % p
            if counts[p][r] == 0:
                zero_counts[p] -= 1
            counts[p][r] += 1
        current_size += 1
        points.append(x)

    # Let's try a few random restarts to find a larger set.
    best_points = []
    
    import random
    random.seed(42)
    
    # Number of restarts based on time
    # Each restart scans 5000 candidates.
    # 5000 * 600 checks = 3e6 ops per restart.
    # 3e6 ops is ~0.01s in Python? Maybe 0.1s.
    # So we can do ~100-300 restarts in 30s.
    
    for restart in range(1000):
        if time.time() > DEADLINE - 5:
            break
            
        # Randomize the order of candidates
        candidates = list(range(N + 1))
        random.shuffle(candidates)
        
        # Reset state
        points = []
        current_size = 0
        for p in primes:
            counts[p] = [0] * p
            zero_counts[p] = p
            
        for x in candidates:
            if time.time() > DEADLINE - 5:
                break
            if is_valid_to_add(x, current_size):
                add_point(x)
                
        if len(points) > len(best_points):
            best_points = points[:]
            
    return best_points
# EVOLVE-BLOCK-END
