import time
import math

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0  # 50 seconds margin out of 60s limit
    
    # Max possible cardinality is 9
    best_A = None
    best_B = None
    best_cardinality = 0
    
    # Precompute primes up to 256 (128 + 128)
    max_val = 256
    is_prime = [True] * (max_val + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.sqrt(max_val)) + 1):
        if is_prime[i]:
            for j in range(i*i, max_val + 1, i):
                is_prime[j] = False
    
    def check_pair(A, B):
        sums = []
        for a in A:
            for b in B:
                sums.append(a + b)
        
        distinct_sums = set(sums)
        card = len(distinct_sums)
        
        if card < best_cardinality:
            return False
        
        # Check if all distinct sums are pairwise coprime
        # If they are all distinct primes, they are pairwise coprime.
        # But the requirement is just pairwise coprime.
        # Checking pairwise coprimality is O(n^2) for n distinct sums.
        # n <= 9, so it's very fast.
        
        # Optimization: If card == 9, check if all are prime. If yes, valid.
        # If not all prime, check gcd.
        
        # General check for pairwise coprime
        vals = list(distinct_sums)
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                if math.gcd(vals[i], vals[j]) != 1:
                    return False
        
        return True

    # Search strategy:
    # We want to maximize cardinality. 9 is the max.
    # We search for A, B in [1, 128].
    # To make it fast, we search small numbers first.
    
    # Try exhaustive search over small ranges
    # Range 1 to 30: 30 choose 3 is 4060. 4060^2 is ~16 million.
    # This might be too slow in Python if done naively.
    # Let's try a randomized search with a structured bias towards small numbers.
    
    import random
    
    # First, try a few specific small structured searches
    # If we find a solution with cardinality 9, we can stop immediately.
    
    # Helper to generate random sets
    def gen_set():
        # Bias towards small numbers
        # Use random.sample from 1..128 but weighted?
        # Or just sample from 1..50 first.
        return sorted(random.sample(range(1, 51), 3))
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # Every 1000 attempts, try a structured small search
        # Or just randomize.
        
        A = gen_set()
        B = gen_set()
        
        # Ensure distinct elements in A and B (random.sample guarantees this)
        # Check validity
        if check_pair(A, B):
            card = len(set(a + b for a in A for b in B))
            if card > best_cardinality:
                best_cardinality = card
                best_A = A
                best_B = B
                if card == 9:
                    return [A, B]
        
        # If we've tried many times, expand range
        if attempts % 1000 == 0:
             # Expand search space slightly or change strategy
             pass

    # Fallback: Deterministic search for small numbers if random failed
    # This ensures we don't return empty if random was unlucky
    # But we have best_A, best_B from random.
    # If best_cardinality is 0, we must find a valid one.
    # The seed program returns a valid one with cardinality 5.
    # We should at least match the seed.
    
    if best_A is None:
        # Return a known valid configuration from seed or simple construction
        # Seed: [[1, 7, 13], [4, 10, 16]] -> Sums: 5, 11, 17, 11, 17, 23, 17, 23, 29
        # Distinct: 5, 11, 17, 23, 29. All primes. Pairwise coprime. Card = 5.
        return [[1, 7, 13], [4, 10, 16]]
        
    return [best_A, best_B]
