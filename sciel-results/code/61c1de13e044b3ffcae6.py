"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from math import isqrt

N = 5000

# EVOLVE-BLOCK-START
def _primes(limit):
    if limit < 2:
        return []
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p*p:limit+1:p] = False
    return np.where(sieve)[0]

def construct():
    start_time = time.time()
    deadline = 300.0  # Safety margin
    
    # Generate all primes up to N
    primes = _primes(N)
    
    # Start with all numbers 0..N
    # Represent S as a boolean mask for efficiency
    S_mask = np.ones(N + 1, dtype=bool)
    
    # Helper to get current size
    def get_size():
        return np.count_nonzero(S_mask)
    
    # Hybrid Strategy:
    # 1. Sieve out small primes (remove residue 0) to satisfy them cheaply in terms of "logic" 
    #    though it removes many points, it prevents the "catastrophic" removal later if we 
    #    had to fix them individually? Actually, fixing small primes individually is 
    #    equivalent to sieving if we pick residue 0.
    #    Let's sieve primes up to a threshold.
    
    # Threshold: Sieve primes <= 100
    small_primes = primes[primes <= 100]
    for p in small_primes:
        # Remove all x in S such that x % p == 0
        # Create a mask for multiples of p
        # Multiples of p are 0, p, 2p...
        # We can use np.arange
        multiples = np.arange(0, N + 1, p)
        S_mask[multiples] = False
        
    # Now iterate to fix remaining primes
    # We need to check primes p such that p <= current_size
    # We don't know current_size exactly without counting, but it decreases.
    
    # To speed up, we can pre-compute residues? 
    # Or just loop.
    
    # List of primes to check. We only care about p <= N.
    # We will iterate through primes in descending order? 
    # Or ascending? 
    # If we fix large primes first, we remove few points.
    # If we fix small primes (already sieved), they are satisfied.
    # So we only need to check primes > 100.
    
    remaining_primes = primes[primes > 100]
    
    # We need to re-check admissibility as S changes.
    # A prime p is "bad" if all p residues are present in S.
    
    # Optimization: 
    # Instead of checking all primes every time, we can just iterate through them.
    # But S changes, so a previously good prime might become bad? 
    # No, removing points only *reduces* coverage. 
    # If a residue was present, it might become absent. 
    # So a prime that was "bad" (all covered) can become "good" (some missing).
    # A prime that was "good" (some missing) remains "good" (still missing).
    # So we only need to worry about primes that are currently "bad".
    
    # However, the condition is p <= len(S).
    # As len(S) decreases, the set of required primes shrinks.
    
    # Strategy:
    # Loop until no bad primes exist among p <= len(S).
    
    # To make it efficient, we can maintain a list of "active" primes.
    # But checking coverage is O(|S|).
    # |S| is at most 5000.
    # Number of primes is ~ 600.
    # 600 * 5000 = 3,000,000 ops per full check.
    # This is fast in Python/NumPy.
    
    # Let's just do a full check in each iteration if needed.
    # But we can be smarter: only check primes that were bad in the last step, 
    # or primes that are <= current size.
    
    # Actually, since we remove points, a prime p is bad only if 
    # np.unique(S_indices % p) has length p.
    
    # Let's iterate.
    
    # To avoid infinite loops or slow convergence, 
    # we pick the "most problematic" prime.
    # What is most problematic? 
    # One where the smallest residue class count is largest? 
    # No, we want to remove the *least* number of points.
    # So for a bad prime p, we remove min(count(r)).
    # We should pick the prime p that minimizes min(count(r))?
    # Or pick the largest p? (Largest p has smallest counts).
    
    # Let's pick the largest bad prime.
    
    changed = True
    while changed:
        if time.time() - start_time > deadline:
            break
            
        current_size = get_size()
        
        # Find primes p <= current_size
        # We only care about p > 100 (since <=100 are handled by sieve)
        # Wait, if current_size < 100, we might need to check primes <= 100 again?
        # No, we removed all multiples of p<=100. So residue 0 is always missing for p<=100.
        # So they are always admissible.
        
        # So we only check p in remaining_primes where p <= current_size.
        bad_primes = []
        
        # Get indices of S for faster modulo ops?
        # S_indices = np.where(S_mask)[0]
        # Doing this every loop is expensive.
        # Let's keep S_indices updated.
        
        # Actually, let's just use S_mask and np.where when needed.
        
        # To speed up, let's maintain S_indices.
        # But S_indices changes when we remove points.
        
        # Let's just compute S_indices once per outer loop iteration?
        # No, S_indices changes inside the inner logic if we remove points.
        
        # Let's restructure:
        # 1. Get current S_indices.
        # 2. Check all candidate primes.
        # 3. If any bad, fix one.
        # 4. Repeat.
        
        S_indices = np.where(S_mask)[0]
        if len(S_indices) == 0:
            break
            
        # Candidate primes: p in remaining_primes and p <= len(S_indices)
        # Also, we should check primes <= 100 if len(S_indices) < 100? 
        # No, they are satisfied by construction (residue 0 missing).
        
        candidates = remaining_primes[remaining_primes <= len(S_indices)]
        
        bad_p = None
        min_removal_count = float('inf')
        best_r = -1
        best_p = -1
        
        for p in candidates:
            # Check if all residues are covered
            residues = S_indices % p
            unique_res = np.unique(residues)
            if len(unique_res) == p:
                # Bad prime. Find least populated residue
                # Count occurrences
                # np.bincount is fast
                counts = np.bincount(residues, minlength=p)
                min_count = np.min(counts)
                # We want to minimize the removal
                if min_count < min_removal_count:
                    min_removal_count = min_count
                    best_r = np.argmin(counts)
                    best_p = p
                    bad_p = p
        
        if bad_p is not None:
            # Remove all x in S such that x % best_p == best_r
            # Find indices in S_indices that match
            # We can compute mask on S_indices
            to_remove_mask = (S_indices % best_p) == best_r
            # Get the actual values to remove
            values_to_remove = S_indices[to_remove_mask]
            # Update S_mask
            S_mask[values_to_remove] = False
            changed = True
        else:
            changed = False
            
    # Return the list of integers
    return np.where(S_mask)[0].tolist()

# EVOLVE-BLOCK-END
