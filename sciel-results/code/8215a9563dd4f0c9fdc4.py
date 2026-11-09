import numpy as np
import time
import random

N = 5000

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

PRIMES = _primes(N)

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    best_set = []
    best_size = 0
    
    # Number of restarts to try
    # Each restart is fast, so we can try many
    num_restarts = 100
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        mask = np.ones(N + 1, dtype=bool)
        active_indices = np.arange(N + 1)
        
        # Main loop
        while True:
            K = mask.sum()
            if K == 0:
                break
                
            # Get primes <= K
            # Since K decreases, we can just filter the global list
            # Optimization: precompute primes and use bisect?
            # For N=5000, linear scan of ~669 primes is fine.
            primes_to_check = [p for p in PRIMES if p <= K]
            
            if not primes_to_check:
                break
                
            fixed = False
            # Iterate primes in ascending order
            for p in primes_to_check:
                # Get residues of active elements
                # mask is boolean, active_indices is full range
                # We need residues of only active elements
                # active_indices[mask] gives the values
                active_vals = active_indices[mask]
                
                if len(active_vals) == 0:
                    break
                    
                residues = active_vals % p
                counts = np.bincount(residues, minlength=p)
                
                if counts.min() > 0:
                    # All classes covered, must remove one
                    min_count = counts.min()
                    candidates = np.where(counts == min_count)[0]
                    
                    # Random tie-breaking
                    # If only one candidate, choice is deterministic
                    chosen = random.choice(candidates)
                    
                    # Remove elements in this residue class
                    # Create a new mask
                    # We need to zero out mask where (val % p == chosen)
                    # We can do this vectorized
                    remove_mask = (active_vals % p == chosen)
                    # But we need to apply this to the global mask
                    # active_vals corresponds to active_indices[mask]
                    # So we need to find which indices in active_indices correspond to these
                    # Actually, simpler:
                    # mask[active_indices % p == chosen] = False 
                    # Wait, active_indices is 0..N.
                    # mask is 0..N.
                    # So:
                    mask[active_indices % p == chosen] = False
                    # Note: this might remove elements that were already False?
                    # No, mask is boolean. Setting False to False is fine.
                    # But we only want to remove active ones.
                    # active_indices % p == chosen is a mask over 0..N.
                    # If mask[i] is already False, it stays False.
                    # If mask[i] is True, it becomes False.
                    # This is correct.
                    
                    fixed = True
                    break # Restart outer while loop
            
            if not fixed:
                break
                
        # Check if this result is better
        current_size = mask.sum()
        if current_size > best_size:
            best_size = current_size
            # Store the actual values
            best_set = active_indices[mask].tolist()

    return best_set
