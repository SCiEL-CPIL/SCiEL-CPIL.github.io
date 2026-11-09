import numpy as np
import math

N = 5000

def _primes(limit):
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p::p] = False
    return np.where(sieve)[0]

PRIMES = _primes(N)

def construct():
    # Estimate upper bound for K
    # K ln K <= N * exp(-gamma)
    # exp(-gamma) ~ 0.56146
    # N * 0.56146 ~ 2807
    # Solve K ln K = 2807
    # Try K=450: 450 * 6.109 = 2749
    # Try K=460: 460 * 6.131 = 2820
    # So max K is around 455-460.
    # Start from 500 to be safe.
    
    best_k = 0
    best_set = []
    
    for K in range(500, 0, -1):
        valid = np.ones(N + 1, dtype=bool)
        count = N + 1
        
        # Only process primes <= K
        # PRIMES is a numpy array
        relevant_primes = PRIMES[PRIMES <= K]
        
        for p in relevant_primes:
            if count < K:
                break
            
            # Get indices of valid numbers
            # We need counts of residues for valid numbers
            # valid_indices = np.where(valid)[0]
            # residues = valid_indices % p
            # counts = np.bincount(residues, minlength=p)
            
            # Optimization: 
            # Instead of extracting valid indices every time, 
            # we can just use the valid mask.
            # But bincount needs the values.
            
            valid_indices = np.where(valid)[0]
            if len(valid_indices) == 0:
                break
                
            residues = valid_indices % p
            counts = np.bincount(residues, minlength=p)
            
            min_count = np.min(counts)
            # Find the first residue with min count
            min_res = np.argmin(counts)
            
            # Mark all x in [0, N] with x % p == min_res as invalid
            # Create mask for these positions
            # positions = np.arange(min_res, N + 1, p)
            # valid[positions] = False
            
            # Update count
            # We can just update valid and sum at the end or track count
            # Tracking count is faster.
            # How many were removed?
            # removed = np.sum(valid[min_res::p])
            # count -= removed
            # valid[min_res::p] = False
            
            # Vectorized update
            mask = np.zeros(N + 1, dtype=bool)
            mask[min_res::p] = True
            # Only those that are currently valid
            to_remove = valid & mask
            removed_count = np.sum(to_remove)
            count -= removed_count
            valid[to_remove] = False
            
        if count >= K:
            # Extract K valid numbers
            valid_indices = np.where(valid)[0]
            # Return list of first K
            return valid_indices[:K].tolist()
            
    return []
