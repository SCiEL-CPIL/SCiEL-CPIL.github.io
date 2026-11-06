# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction
import random
import time

def _primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i, v in enumerate(sieve) if v]

PRIMES = _primes_up_to(1000)

def construct():
    # Select a set of primes that fits within budget 1
    # We use {3, 5, 7, 11, 13, 17, 19, 23}
    # Sum of reciprocals is approx 0.994
    selected_primes = [3, 5, 7, 11, 13, 17, 19, 23]
    
    # Verify budget
    budget_sum = sum(Fraction(1, p) for p in selected_primes)
    if budget_sum > 1:
        # Fallback to a safe set if needed, though this set is safe
        selected_primes = [3, 5, 7, 11, 13, 17, 19]
        budget_sum = sum(Fraction(1, p) for p in selected_primes)

    L_max_check = 10000
    # We will search for residues that maximize the covered prefix length
    
    # Initial random residues
    residues = [random.randint(0, p - 1) for p in selected_primes]
    
    def calculate_L(residues):
        covered = np.zeros(L_max_check + 1, dtype=bool)
        for p, r in zip(selected_primes, residues):
            # Indices are 1..L_max_check
            # n % p == r
            # n = r + k*p
            # First n >= 1
            # If r == 0, start at p (since n>=1 and n%p==0 -> n=p, 2p...)
            # If r > 0, start at r
            if r == 0:
                start = p
            else:
                start = r
            
            if start <= L_max_check:
                covered[start:L_max_check+1:p] = True
        
        # Find first False
        # np.argmax returns index of first True. 
        # We want first False in ~covered.
        # If all True, argmax is 0 (or undefined if empty, but size > 0).
        # Careful: if covered[0] is False (it is, we start at 1), argmax(~covered) is 0.
        # We need the first index i >= 1 such that covered[i] is False.
        
        # Let's just find the first False in covered[1:]
        # ~covered[1:] is boolean array
        # If all True, np.argmax returns 0? No, if all False, argmax returns 0.
        # If all True in ~covered[1:], then all covered.
        
        not_covered = ~covered[1:]
        if not any(not_covered):
            return L_max_check
        
        first_gap_idx = np.argmax(not_covered)
        # first_gap_idx is 0-based index in covered[1:]
        # So actual number is first_gap_idx + 1
        # The covered length is (first_gap_idx + 1) - 1 = first_gap_idx
        # Example: covered[1]=True, covered[2]=False.
        # not_covered[0]=False, not_covered[1]=True.
        # argmax is 1.
        # L = 1. (1 is covered, 2 is not).
        # Formula: L = first_gap_idx
        return first_gap_idx

    best_L = calculate_L(residues)
    best_residues = residues[:]
    
    # Simulated Annealing
    start_time = time.time()
    T = 1.0
    cooling = 0.995
    min_T = 0.001
    max_time = 60  # Use 60s to be safe
    
    current_L = best_L
    current_residues = best_residues[:]
    
    while T > min_T and (time.time() - start_time) < max_time:
        # Pick a random prime to change
        idx = random.randint(0, len(selected_primes) - 1)
        p = selected_primes[idx]
        old_r = current_residues[idx]
        new_r = random.randint(0, p - 1)
        
        if new_r == old_r:
            continue
            
        # Try new residue
        trial_residues = current_residues[:]
        trial_residues[idx] = new_r
        
        trial_L = calculate_L(trial_residues)
        
        delta = trial_L - current_L
        
        if delta > 0 or random.random() < np.exp(delta / T):
            current_L = trial_L
            current_residues = trial_residues
            if current_L > best_L:
                best_L = current_L
                best_residues = current_residues[:]
        
        T *= cooling
        
    # Return the best found
    classes = [[p, r] for p, r in zip(selected_primes, best_residues)]
    
    # Optional: Try to extend with remaining budget if we have time and budget
    # But the SA should have found a good configuration for the fixed set.
    # The prompt asks to maximize L.
    # With this set, we expect L to be quite high (maybe 1000+).
    
    return classes

# EVOLVE-BLOCK-END
