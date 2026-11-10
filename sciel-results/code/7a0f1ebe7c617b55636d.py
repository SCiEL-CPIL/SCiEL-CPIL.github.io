import numpy as np
import time
import random

def get_primes(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i:n+1:i] = [False] * len(sieve[i*i:n+1:i])
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def construct():
    primes = get_primes(1000)
    n_primes = len(primes)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # State
    # Residues
    # Try a few initializations? Or just random?
    # Let's try a heuristic init: r_i = i % p_i
    residues = [i % p for i, p in enumerate(primes)]
    
    # Max L to check
    # The problem asks for L >= 1000. Let's aim for 5000 or 10000.
    # If we find L > 1000, that's a success.
    # Let's set L_limit = 20000 to be safe, but scanning 20000 is fine.
    L_limit = 20000
    
    # Coverage array
    cov = np.zeros(L_limit, dtype=np.int32)
    
    # Initialize coverage
    for i, p in enumerate(primes):
        r = residues[i]
        # Indices: r, r+p, ... < L_limit
        # np.arange(r, L_limit, p)
        idxs = np.arange(r, L_limit, p)
        cov[idxs] += 1
        
    # Calculate initial L
    bad = np.where(cov < 2)[0]
    if len(bad) == 0:
        cur_L = L_limit
    else:
        cur_L = bad[0] - 1
        
    best_L = cur_L
    best_residues = residues.copy()
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Pick random prime
        i = random.randint(0, n_primes - 1)
        p = primes[i]
        old_r = residues[i]
        
        # Pick new residue
        # Bias towards small residues? 
        # Uniform is fine, but maybe small residues help cover start.
        new_r = random.randint(0, p - 1)
        
        if new_r == old_r:
            continue
            
        # Update coverage
        old_idxs = np.arange(old_r, L_limit, p)
        new_idxs = np.arange(new_r, L_limit, p)
        
        cov[old_idxs] -= 1
        cov[new_idxs] += 1
        
        # Calculate new L
        bad = np.where(cov < 2)[0]
        if len(bad) == 0:
            new_L = L_limit
        else:
            new_L = bad[0] - 1
            
        # Acceptance
        if new_L >= cur_L:
            cur_L = new_L
            residues[i] = new_r
            if new_L > best_L:
                best_L = new_L
                best_residues = residues.copy()
        else:
            # Delta L is negative
            delta = cur_L - new_L
            # Probability
            # If T is small, reject.
            # Scale delta: L can be in thousands. 
            # Let's normalize by 100?
            prob = np.exp(-delta / (T * 100.0))
            if random.random() < prob:
                cur_L = new_L
                residues[i] = new_r
            else:
                # Revert
                cov[old_idxs] += 1
                cov[new_idxs] -= 1
                residues[i] = old_r # already old_r, but explicit
                
        T *= cooling
        
    return best_residues
