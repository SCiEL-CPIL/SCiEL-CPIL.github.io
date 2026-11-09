"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import bisect

def _sieve_primes(limit):
    """Return numpy array of all primes <= limit."""
    if limit < 2:
        return np.array([], dtype=np.int64)
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return np.where(is_prime)[0]

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    # Get all primes <= 10000
    primes = _sieve_primes(10000)
    n = len(primes)
    
    if n == 0:
        return []
    if n == 1:
        return [int(primes[0])]
    
    primes_list = primes.tolist()
    
    best_len = 0
    best_seq = []
    
    # For each starting prime, simulate the greedy strategy:
    # Always pick the smallest next prime that satisfies the gap constraint.
    # This minimizes the next gap, which is optimal for maximizing the number of future terms.
    
    for i in range(n):
        if time.time() > deadline:
            break
            
        current_val = primes_list[i]
        last_gap = 0
        current_idx = i
        seq = [current_val]
        
        while True:
            # Find the next prime
            # Target value: current_val + last_gap
            target = current_val + last_gap
            
            # We need the first prime > current_val such that prime - current_val >= last_gap
            # This is equivalent to prime >= target
            # Use bisect to find the first prime >= target
            j = bisect.bisect_left(primes_list, target, lo=current_idx + 1)
            
            if j == n:
                break
            
            next_val = primes_list[j]
            # Check if the gap is valid (it should be by construction of target, 
            # but bisect_left finds >= target, so next_val - current_val >= last_gap)
            new_gap = next_val - current_val
            
            # Since we picked the smallest such prime, new_gap is the smallest valid gap.
            # Update state
            seq.append(next_val)
            last_gap = new_gap
            current_val = next_val
            current_idx = j
            
            # Optimization: If the remaining primes are not enough to beat best_len, break?
            # Remaining count = n - current_idx
            # Current len = len(seq)
            # Max possible total len = len(seq) + (n - current_idx - 1)
            # If this max <= best_len, we can break early for this start i.
            if len(seq) + (n - current_idx - 1) <= best_len:
                break
        
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq
    
    return best_seq

# EVOLVE-BLOCK-END
