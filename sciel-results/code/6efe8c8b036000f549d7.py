import numpy as np
import time
import math

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
    N = 5000
    
    # Weights for the soft objective function
    # We want to maximize the prefix length, so early indices are more important.
    # Use a geometric decay or harmonic series.
    # Harmonic: 1/x. This is good because if we cover 1..L, the sum is H_L.
    # If we fail at L, we lose 1/L.
    weights = 1.0 / np.arange(1, N + 1, dtype=np.float64)
    
    best_residues = None
    best_score = 0
    
    start_time = time.time()
    max_time = 110 # seconds
    
    def get_soft_score(counts):
        # counts[1:N+1] corresponds to x=1..N
        # min(count, 2) ensures we don't over-reward high counts
        covered = np.minimum(counts[1:N+1], 2)
        return np.dot(covered, weights)
    
    def get_exact_score(counts):
        # Find the first x in 1..N where count[x] < 2
        # counts[1:N+1]
        failures = np.nonzero(counts[1:N+1] < 2)[0]
        if len(failures) == 0:
            return N
        return failures[0] + 1
    
    # Simulated Annealing Parameters
    T = 1.0
    T_min = 0.01
    alpha = 0.999
    
    # Run multiple restarts
    for restart in range(100):
        if time.time() - start_time > max_time:
            break
            
        # Initialize residues
        # Heuristic initialization: set r_p = 1 for all p?
        # Or random?
        # Let's try random but biased towards small residues?
        residues = np.array([np.random.randint(0, p) for p in primes])
        
        # Calculate initial counts
        counts = np.zeros(N + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            counts[r::p] += 1
            
        current_soft = get_soft_score(counts)
        current_exact = get_exact_score(counts)
        
        if current_exact > best_score:
            best_score = current_exact
            best_residues = residues.copy()
            
        # Annealing loop
        T = 1.0
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
            
            new_soft = get_soft_score(counts)
            delta_soft = new_soft - current_soft
            
            # Acceptance probability
            if delta_soft > 0 or np.random.rand() < math.exp(delta_soft / T):
                current_soft = new_soft
                residues[i] = r_new
                
                # Check exact score periodically or if soft score improved significantly
                # To save time, only check exact score if soft score is high or periodically
                if delta_soft > 0.01 or (restart == 0 and i % 10 == 0):
                    new_exact = get_exact_score(counts)
                    if new_exact > best_score:
                        best_score = new_exact
                        best_residues = residues.copy()
            else:
                # Revert
                counts[r_new::p] -= 1
                counts[r_old::p] += 1
                
            # Cool down
            T *= alpha
            if T < T_min:
                T = T_min
                # Restart cooling
                T = 1.0
                break
                
    return best_residues.tolist()
