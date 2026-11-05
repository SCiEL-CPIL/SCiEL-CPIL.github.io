import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Primes up to 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                sieve[i*i:limit+1:i] = [False] * len(sieve[i*i:limit+1:i])
        return [i for i, is_prime in enumerate(sieve) if is_prime]
        
    primes = get_primes(1000)
    n_primes = len(primes)
    
    MAX_L = 100000
    
    # Initialize residues (using seed logic as a decent start, or random)
    # Let's use the seed logic as it's known to give L=84
    residues = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    
    # Initialize counts
    counts = np.zeros(MAX_L + 1, dtype=np.int32)
    for i, p in enumerate(primes):
        r = residues[i]
        start = r if r > 0 else p
        if start <= MAX_L:
            counts[start::p] += 1
            
    # Calculate initial L
    def calc_L(counts, current_L):
        if current_L > 0:
            # Check if current prefix is still valid
            # We only need to check up to current_L
            # If any count < 2, L drops
            # Optimization: only check if we suspect it dropped? 
            # No, we must check.
            # But if L is small, it's fast.
            prefix = counts[1:current_L+1]
            if np.min(prefix) < 2:
                bad = np.where(prefix < 2)[0]
                return bad[0] # Index in 0-based slice of 1..L corresponds to L_new
                # e.g. counts[1] is bad -> bad[0]=0 -> L=0
                # counts[2] is bad -> bad[0]=1 -> L=1
                # So L_new = bad[0]
            # Else, try to extend
            L = current_L
            if L < MAX_L:
                rest = counts[L+1:]
                bad_idx = np.where(rest < 2)[0]
                if len(bad_idx) == 0:
                    return MAX_L
                else:
                    return L + bad_idx[0]
        else:
            # L is 0, check from 1
            bad_idx = np.where(counts[1:] < 2)[0]
            if len(bad_idx) == 0:
                return MAX_L
            else:
                return bad_idx[0]
                
    current_L = calc_L(counts, 0)
    best_L = current_L
    best_residues = residues.copy()
    
    # Simulated Annealing Parameters
    T = 1.0
    cooling = 0.9999
    
    while time.time() < deadline:
        # Pick random prime
        i = np.random.randint(n_primes)
        p = primes[i]
        old_r = residues[i]
        new_r = np.random.randint(p)
        
        if new_r == old_r:
            continue
            
        # Apply change
        start_old = old_r if old_r > 0 else p
        if start_old <= MAX_L:
            counts[start_old::p] -= 1
            
        start_new = new_r if new_r > 0 else p
        if start_new <= MAX_L:
            counts[start_new::p] += 1
            
        # Calculate new L
        new_L = calc_L(counts, current_L)
        
        # Acceptance
        if new_L >= current_L:
            # Accept
            residues[i] = new_r
            current_L = new_L
            if new_L > best_L:
                best_L = new_L
                best_residues = residues.copy()
        else:
            # Reject or Accept with probability
            # Delta is negative
            delta = new_L - current_L
            prob = np.exp(delta / T)
            if np.random.random() < prob:
                residues[i] = new_r
                current_L = new_L
            else:
                # Revert
                residues[i] = old_r
                # Revert counts
                if start_new <= MAX_L:
                    counts[start_new::p] -= 1
                if start_old <= MAX_L:
                    counts[start_old::p] += 1
                # current_L remains unchanged
                
        T *= cooling
        if T < 0.01:
            T = 0.01
            
    return best_residues
