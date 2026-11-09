import time
import random
from collections import defaultdict

def _sieve_prime_factors(limit):
    """Precompute prime factors for all numbers up to limit."""
    factors = [set() for _ in range(limit + 1)]
    for i in range(2, limit + 1):
        if not factors[i]:
            for j in range(i, limit + 1, i):
                factors[j].add(i)
    return factors

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Precompute prime factors for sums up to 20000
    max_sum = 20000
    pf = _sieve_prime_factors(max_sum)
    
    n = 12
    max_val = 10000
    
    # Initialize with a random set of distinct integers
    A = random.sample(range(1, max_val + 1), n)
    
    # Function to compute prime counts for a given set A
    def get_prime_counts(arr):
        counts = defaultdict(int)
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                s = arr[i] + arr[j]
                for p in pf[s]:
                    counts[p] += 1
        return counts
    
    current_counts = get_prime_counts(A)
    current_distinct = len([p for p, c in current_counts.items() if c > 0])
    
    best_A = A[:]
    best_distinct = current_distinct
    best_counts = current_counts
    
    # Local search
    # We track the current state and try to improve it.
    # If stuck for a while, we can restart or use simulated annealing.
    # Given the small n, simple hill climbing with random restarts might work.
    
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        # Pick a random index to replace
        i = random.randint(0, n - 1)
        old_val = A[i]
        
        # Generate a new candidate value
        # Bias towards smaller numbers? Or random?
        # Smaller numbers might have smoother sums?
        # Let's try random first.
        new_val = random.randint(1, max_val)
        
        # Ensure distinctness
        if new_val in A and new_val != old_val:
            continue
        
        # Calculate delta
        # Remove old_val contributions
        # Add new_val contributions
        
        # To be efficient, we can just recalculate the affected sums.
        # Sums involving A[i] change.
        # There are n-1 such sums.
        
        # Instead of complex delta logic, since n is small (12),
        # and pf lookup is O(1) amortized, we can just compute the new counts
        # for the affected pairs and update the global dict.
        
        # But we need to know if the *number of distinct primes* changes.
        # This requires knowing if a prime count goes from 0 to >0 or >0 to 0.
        
        # Let's do a quick check:
        # 1. Temporarily remove old_val from consideration? No, just compute new state for affected pairs.
        
        # Let's just compute the full new counts if we accept the candidate?
        # No, that's O(n^2) per step. 12*11/2 = 66 sums.
        # 66 lookups in pf. Very fast.
        # We can afford to compute the full new counts for every candidate.
        
        new_A = A[:]
        new_A[i] = new_val
        
        new_counts = get_prime_counts(new_A)
        new_distinct = len([p for p, c in new_counts.items() if c > 0])
        
        if new_distinct < current_distinct:
            A = new_A
            current_counts = new_counts
            current_distinct = new_distinct
            no_improve_count = 0
            
            if current_distinct < best_distinct:
                best_A = A[:]
                best_distinct = current_distinct
                best_counts = current_counts
        else:
            no_improve_count += 1
            # Simulated annealing-like acceptance?
            # If new_distinct == current_distinct, we might accept to explore.
            if new_distinct == current_distinct and random.random() < 0.1:
                A = new_A
                current_counts = new_counts
                # current_distinct stays same
                # Don't update best unless we find lower later
            else:
                # Reject
                pass
        
        if no_improve_count > max_no_improve:
            # Random restart
            A = random.sample(range(1, max_val + 1), n)
            current_counts = get_prime_counts(A)
            current_distinct = len([p for p, c in current_counts.items() if c > 0])
            no_improve_count = 0
    
    return best_A

