import time
import random
import math
import numpy as np

def get_primes(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

PRIMES = np.array(get_primes(1000))
N_PRIMES = len(PRIMES)
MAX_CHECK = 5000

def compute_L(counts):
    # counts is a numpy array of size MAX_CHECK+1
    # Find first index > 0 where count < 2
    # We care about 1..MAX_CHECK
    # If all are >= 2, L = MAX_CHECK
    # Use np.argmax on boolean array (counts[1:] < 2)
    # If no such element, argmax returns 0, which corresponds to index 1 in original?
    # No, if all False, argmax returns 0.
    # We need to handle the case where all are covered.
    mask = counts[1:] < 2
    if not np.any(mask):
        return MAX_CHECK
    first_fail = np.argmax(mask)
    # first_fail is index in counts[1:], so actual index is first_fail + 1
    # L is first_fail (since 1..first_fail are covered, first_fail+1 is not)
    # Wait.
    # If counts[1] < 2, mask[0] is True. argmax=0. L=0. Correct.
    # If counts[1]>=2, counts[2]<2, mask[0]=False, mask[1]=True. argmax=1. L=1. Correct.
    return first_fail

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Initialization
    residues = np.zeros(N_PRIMES, dtype=int)
    for i, p in enumerate(PRIMES):
        residues[i] = ((i + 2) // 2) % p
        
    counts = np.zeros(MAX_CHECK + 1, dtype=int)
    # Initialize counts
    for i, p in enumerate(PRIMES):
        r = residues[i]
        start = r
        if start < 1:
            # Find smallest x >= 1 such that x == r mod p
            # x = r + k*p >= 1
            # k >= (1 - r) / p
            k = math.ceil((1 - r) / p)
            if k < 0: k = 0 # r is always >= 0, so 1-r <= 1. If r=0, k=0? 0+0*p=0 < 1. k=1? 0+1*p=p>=1.
            # Actually simpler:
            # range(r, MAX_CHECK+1, p) includes 0 if r=0.
            # We want x >= 1.
            # If r == 0, first x is p.
            # If r > 0, first x is r.
            if r == 0:
                start = p
            else:
                start = r
        # Update counts
        if start <= MAX_CHECK:
            counts[start:MAX_CHECK+1:p] += 1

    L_curr = compute_L(counts)
    L_best = L_curr
    best_residues = residues.copy()
    
    T = 10.0
    T_min = 0.1
    alpha = 0.999
    
    while time.time() < deadline:
        move_type = random.random()
        
        if move_type < 0.6:
            # Random single change
            i = random.randint(0, N_PRIMES - 1)
            p = PRIMES[i]
            old_r = residues[i]
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue
            
            # Update counts
            # Decrement old
            start_old = old_r if old_r > 0 else p
            if start_old <= MAX_CHECK:
                counts[start_old:MAX_CHECK+1:p] -= 1
                
            # Increment new
            start_new = new_r if new_r > 0 else p
            if start_new <= MAX_CHECK:
                counts[start_new:MAX_CHECK+1:p] += 1
                
            residues[i] = new_r
            
        else:
            # Targeted move
            target_x = L_curr + 1
            if target_x > MAX_CHECK:
                # Cap reached
                i = random.randint(0, N_PRIMES - 1)
                p = PRIMES[i]
                old_r = residues[i]
                new_r = random.randint(0, p - 1)
                if new_r == old_r:
                    continue
                start_old = old_r if old_r > 0 else p
                if start_old <= MAX_CHECK:
                    counts[start_old:MAX_CHECK+1:p] -= 1
                start_new = new_r if new_r > 0 else p
                if start_new <= MAX_CHECK:
                    counts[start_new:MAX_CHECK+1:p] += 1
                residues[i] = new_r
            else:
                # Find primes that do NOT cover target_x
                # target_x % p != residues[i]
                # Vectorized check
                mods = target_x % PRIMES
                mismatches = mods != residues
                candidate_indices = np.where(mismatches)[0]
                
                if len(candidate_indices) > 0:
                    i = candidate_indices[random.randint(0, len(candidate_indices)-1)]
                    p = PRIMES[i]
                    old_r = residues[i]
                    new_r = target_x % p
                    
                    start_old = old_r if old_r > 0 else p
                    if start_old <= MAX_CHECK:
                        counts[start_old:MAX_CHECK+1:p] -= 1
                    start_new = new_r if new_r > 0 else p
                    if start_new <= MAX_CHECK:
                        counts[start_new:MAX_CHECK+1:p] += 1
                    residues[i] = new_r
                else:
                    # Fallback to random
                    i = random.randint(0, N_PRIMES - 1)
                    p = PRIMES[i]
                    old_r = residues[i]
                    new_r = random.randint(0, p - 1)
                    if new_r == old_r:
                        continue
                    start_old = old_r if old_r > 0 else p
                    if start_old <= MAX_CHECK:
                        counts[start_old:MAX_CHECK+1:p] -= 1
                    start_new = new_r if new_r > 0 else p
                    if start_new <= MAX_CHECK:
                        counts[start_new:MAX_CHECK+1:p] += 1
                    residues[i] = new_r

        L_new = compute_L(counts)
        
        if L_new > L_curr:
            L_curr = L_new
            if L_new > L_best:
                L_best = L_new
                best_residues = residues.copy()
        else:
            delta = L_new - L_curr
            if delta == 0:
                if random.random() < 0.5:
                    L_curr = L_new
            else:
                if random.random() < math.exp(delta / T):
                    L_curr = L_new
        
        T *= alpha
        if T < T_min:
            T = T_min

    return best_residues.tolist()
