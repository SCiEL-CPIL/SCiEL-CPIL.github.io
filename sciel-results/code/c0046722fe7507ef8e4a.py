import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 130s limit, 20s margin
    
    # Generate primes <= 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                sieve[i*i::i] = [False] * len(sieve[i*i::i])
        return [i for i, is_p in enumerate(sieve) if is_p]
    
    primes = get_primes(1000)
    n_primes = len(primes)
    assert n_primes == 168
    
    # We only care about L up to a certain limit for optimization speed.
    # The goal is L >= 1000. Let's optimize for L up to 2000 to be safe, 
    # but the bottleneck is usually much smaller if the configuration is bad.
    # Actually, checking up to 5000 is fast enough with vectorization.
    MAX_L_CHECK = 5000
    
    # Precompute the indices for each prime and residue? 
    # No, that's 168 * avg_p memory. 
    # Instead, we update a counts array.
    
    def compute_L(residues, counts_cache=None):
        """
        Compute the largest L such that every x in [1,L] has count >= 2.
        Uses a temporary counts array.
        """
        counts = np.zeros(MAX_L_CHECK + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            if r == 0:
                # Covers p, 2p, 3p, ...
                # np.arange is fast
                idx = np.arange(p, MAX_L_CHECK + 1, p)
            else:
                idx = np.arange(r, MAX_L_CHECK + 1, p)
            if len(idx) > 0:
                counts[idx] += 1
        
        # Find largest L where min(counts[1:L+1]) >= 2
        # We can do this by finding the first index where counts < 2
        # counts[0] is not used.
        # valid = counts >= 2
        # We need the longest prefix of True in valid[1:]
        
        # Efficient check:
        # If any count in [1, MAX_L_CHECK] is < 2, L is the index before it.
        # We want the max L.
        # Let's just loop, it's O(MAX_L_CHECK) which is 5000 iterations. Very fast.
        L = 0
        for x in range(1, MAX_L_CHECK + 1):
            if counts[x] >= 2:
                L = x
            else:
                break
        return L

    # Initial residues: 
    # A good start is to try to cover small numbers.
    # For p=2, r=0 covers evens, r=1 covers odds.
    # Let's start with a greedy assignment for the first few primes.
    residues = np.zeros(n_primes, dtype=np.int64)
    
    # Greedy initialization for first 20 primes
    # For each prime, try all residues and pick the one that maximizes the current L
    # This is expensive if done sequentially for all 168, but we only do it for the first K.
    
    K_ACTIVE = 25  # Number of small primes to actively optimize
    
    # Initialize small primes greedily
    for i in range(K_ACTIVE):
        p = primes[i]
        best_r = 0
        best_L = -1
        for r in range(p):
            residues[i] = r
            # We don't need to recompute full L for all previous primes if we are careful,
            # but for simplicity and correctness, we compute L for the whole set.
            # However, only the first i+1 primes are non-zero (or set). 
            # The rest are 0.
            L = compute_L(residues)
            if L > best_L:
                best_L = L
                best_r = r
        residues[i] = best_r
        
    # Initialize remaining primes to 0 (or 1, doesn't matter much yet)
    for i in range(K_ACTIVE, n_primes):
        residues[i] = 0

    current_L = compute_L(residues)
    best_L = current_L
    best_residues = residues.copy()
    
    # Local Search Phase
    # We iterate over the active primes (first K_ACTIVE) and try to improve by changing one residue.
    # We use a "best response" strategy: for a given prime i, try ALL residues and pick the best.
    
    rng = random.Random(42)
    
    # To avoid getting stuck in local minima, we might shuffle the order or restart occasionally.
    # But best-response is quite strong.
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        # Shuffle the order of active primes to avoid bias
        active_indices = list(range(K_ACTIVE))
        rng.shuffle(active_indices)
        
        for i in active_indices:
            if time.time() > deadline - 10:
                break
            
            p = primes[i]
            old_r = residues[i]
            best_local_L = current_L
            best_local_r = old_r
            
            # Try all residues for this prime
            for r in range(p):
                if r == old_r:
                    continue
                residues[i] = r
                L = compute_L(residues)
                if L > best_local_L:
                    best_local_L = L
                    best_local_r = r
            
            # Apply best
            if best_local_r != old_r:
                residues[i] = best_local_r
                current_L = best_local_L
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues.copy()
                improved = True
    
    # If we have time, try to expand the active set or do random perturbations on the active set
    if time.time() < deadline - 20:
        # Try adding a few more primes to the active set
        K_ACTIVE_2 = 35
        if K_ACTIVE < K_ACTIVE_2:
             # Initialize the new primes greedily
             for i in range(K_ACTIVE, K_ACTIVE_2):
                 p = primes[i]
                 best_r = residues[i]
                 best_L_local = current_L
                 for r in range(p):
                     residues[i] = r
                     L = compute_L(residues)
                     if L > best_L_local:
                         best_L_local = L
                         best_r = r
                 residues[i] = best_r
             
             # Run local search on the expanded set
             improved = True
             while improved and time.time() < deadline - 10:
                 improved = False
                 active_indices = list(range(K_ACTIVE_2))
                 rng.shuffle(active_indices)
                 for i in active_indices:
                     if time.time() > deadline - 10:
                         break
                     p = primes[i]
                     old_r = residues[i]
                     best_local_L = current_L
                     best_local_r = old_r
                     for r in range(p):
                         if r == old_r:
                             continue
                         residues[i] = r
                         L = compute_L(residues)
                         if L > best_local_L:
                             best_local_L = L
                             best_local_r = r
                     if best_local_r != old_r:
                         residues[i] = best_local_r
                         current_L = best_local_L
                         if current_L > best_L:
                             best_L = current_L
                             best_residues = residues.copy()
                         improved = True

    # Final check: ensure we return the best found
    # The best_residues might not be the current residues if we didn't update best_residues in the last step
    # Let's do a final compute on best_residues to be sure
    final_L = compute_L(best_residues)
    
    return [int(r) for r in best_residues]
