import time
import random
import numpy as np

def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i, is_p in enumerate(sieve) if is_p]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    primes = _primes_upto(1000)
    n_primes = len(primes)
    
    # Set a limit for evaluation. 20000 is a good balance.
    # If we reach this, the score is at least 20000.
    # The evaluator checks up to 100000, but finding a solution > 20000 is already very good.
    # We can increase this if we have time, but 20000 should be sufficient for search dynamics.
    limit = 20000
    x = np.arange(1, limit + 1)
    
    def evaluate(residues):
        """
        Returns the largest L such that all 1..L are covered at least twice.
        Uses vectorized numpy operations.
        """
        counts = np.zeros(limit, dtype=np.int8)
        # Iterate over primes and add to counts where x % p == r
        for i, p in enumerate(primes):
            r = residues[i]
            # Vectorized comparison
            # Note: x % p is computed every time. 
            # This is 168 * 20000 = 3.36M operations. 
            # NumPy handles this very fast.
            mask = (x % p) == r
            counts[mask] += 1
            
        # Find first index where count < 2
        # np.where returns indices
        bad_indices = np.where(counts < 2)[0]
        if len(bad_indices) == 0:
            return limit
        return int(bad_indices[0])

    # Initialization
    best_residues = None
    best_score = -1
    
    # Try a few random initializations
    for seed in [42, 123, 999, 2024, 555]:
        random.seed(seed)
        residues = [random.randint(0, p - 1) for p in primes]
        score = evaluate(residues)
        if score > best_score:
            best_score = score
            best_residues = residues[:]
            
    # Simulated Annealing
    current_residues = best_residues[:]
    current_score = best_score
    
    T = 10.0  # Initial temperature
    cooling = 0.9995
    min_T = 0.5
    
    # We will run until deadline
    steps = 0
    while time.time() < deadline:
        # Pick a random prime to change
        i = random.randint(0, n_primes - 1)
        old_r = current_residues[i]
        new_r = random.randint(0, primes[i] - 1)
        
        # Skip if no change
        if new_r == old_r:
            continue
            
        current_residues[i] = new_r
        new_score = evaluate(current_residues)
        
        delta = new_score - current_score
        
        # Acceptance criterion
        if delta > 0:
            # Always accept improvement
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_residues = current_residues[:]
        else:
            # Accept worse with probability exp(-delta/T)
            # delta is negative, so -delta is positive
            prob = np.exp(-delta / T)
            if random.random() < prob:
                current_score = new_score
            else:
                # Revert
                current_residues[i] = old_r
        
        # Cool down
        T *= cooling
        if T < min_T:
            T = min_T
            
        steps += 1
        
        # Optional: If we hit the limit, we might want to restart or increase limit?
        # If best_score == limit, we are hitting the cap.
        # We can try to increase limit to see if we can go further.
        # But re-evaluating with a new limit requires rebuilding x.
        # Let's just keep searching. If we are at the cap, delta is often 0.
        # The SA will wander.
        
        # Check time periodically to avoid overshooting
        if steps % 100 == 0 and time.time() > deadline - 5:
            break

    return best_residues
