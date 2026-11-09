import numpy as np
import time
import random

def construct():
    # Generate primes <= 1000
    primes = []
    is_prime = [True] * 1001
    is_prime[0] = is_prime[1] = False
    for i in range(2, 1001):
        if is_prime[i]:
            primes.append(i)
            for j in range(i*i, 1001, i):
                is_prime[j] = False
    
    n_primes = len(primes)
    M = 3000 # Search horizon
    
    # Initial random assignment
    residues = np.array([random.randint(0, p-1) for p in primes])
    
    # Initialize Coverage Array C[1..M]
    C = np.zeros(M + 1, dtype=np.int32)
    for i, p in enumerate(primes):
        r = residues[i]
        if r == 0:
            start = p
        else:
            start = r
        C[start::p] += 1
        
    # Helper to get L
    def get_L(C):
        bad = C[1:M+1] < 2
        if not np.any(bad):
            return M
        first_bad = np.argmax(bad) + 1
        return first_bad - 1

    current_L = get_L(C)
    best_L = current_L
    best_res = residues.copy()
    
    # Simulated Annealing
    start_time = time.time()
    deadline = start_time + 140
    
    T = 20.0
    cooling = 0.9999
    
    # Use local variables for speed
    primes_list = primes
    n_p = n_primes
    
    while time.time() < deadline:
        i = random.randint(0, n_p - 1)
        p = primes_list[i]
        old_r = residues[i]
        
        new_r = random.randint(0, p - 1)
        if new_r == old_r:
            continue
        
        # Update C
        if old_r == 0:
            start_old = p
        else:
            start_old = old_r
        C[start_old::p] -= 1
        
        if new_r == 0:
            start_new = p
        else:
            start_new = new_r
        C[start_new::p] += 1
        
        new_L = get_L(C)
        
        # Acceptance
        delta = new_L - current_L
        if delta >= 0:
            residues[i] = new_r
            current_L = new_L
            if new_L > best_L:
                best_L = new_L
                best_res = residues.copy()
        else:
            # Reject or accept with prob exp(delta / T)
            # delta is negative
            if random.random() < np.exp(delta / T):
                residues[i] = new_r
                current_L = new_L
            else:
                # Revert C
                C[start_new::p] -= 1
                C[start_old::p] += 1
                # residues[i] remains old_r
        
        T *= cooling
        if T < 0.1:
            T = 0.1
            
    return [int(r) for r in best_res]
