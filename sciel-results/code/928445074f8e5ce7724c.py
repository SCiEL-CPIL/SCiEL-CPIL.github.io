import numpy as np
import time

def construct():
    # Generate primes <= 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(1000)
    n_primes = len(primes)
    
    # Search range for L
    # The problem mentions L >= 1000. Let's search up to 5000 to be safe.
    # If we find L > 5000, we can extend, but 5000 is likely sufficient for an improvement over 84.
    N = 5000
    
    best_residues = None
    best_score = 0
    
    # Multiple random restarts
    start_time = time.time()
    max_time = 100 # seconds, leaving buffer for overhead
    
    while time.time() - start_time < max_time:
        # Initialize residues
        # Strategy: Random
        residues = np.array([np.random.randint(0, p) for p in primes])
        
        # Calculate initial counts
        counts = np.zeros(N, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            counts[r::p] += 1
            
        # Calculate score
        failures = np.nonzero(counts < 2)[0]
        current_score = failures[0] if len(failures) > 0 else N
        
        if current_score > best_score:
            best_score = current_score
            best_residues = residues.copy()
            
        # Hill Climbing
        # We will perform a number of steps. 
        # If we get stuck (no improvement for many steps), we break to restart.
        stagnation_counter = 0
        max_stagnation = 1000
        
        while time.time() - start_time < max_time:
            # Pick a random prime
            i = np.random.randint(n_primes)
            p = primes[i]
            r_old = residues[i]
            
            # Pick a new random residue
            r_new = np.random.randint(0, p)
            if r_old == r_new:
                continue
                
            # Update counts
            counts[r_old::p] -= 1
            counts[r_new::p] += 1
            
            # Check score
            failures = np.nonzero(counts < 2)[0]
            new_score = failures[0] if len(failures) > 0 else N
            
            if new_score > current_score:
                current_score = new_score
                residues[i] = r_new
                stagnation_counter = 0
                if current_score > best_score:
                    best_score = current_score
                    best_residues = residues.copy()
            else:
                # Revert
                counts[r_new::p] -= 1
                counts[r_old::p] += 1
                stagnation_counter += 1
                if stagnation_counter > max_stagnation:
                    break
                    
    return best_residues.tolist()
