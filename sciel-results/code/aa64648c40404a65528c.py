import numpy as np
import time

def _primes_up_to(n):
    """Return list of primes up to n."""
    if n < 2:
        return []
    sieve = np.ones(n + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    return [int(p) for p in np.where(sieve)[0]]

def _find_first_gap(counts):
    """Find the first index i >= 1 such that counts[i] < 2.
    Returns cap if all are covered."""
    bad = np.where(counts[1:] < 2)[0]
    if len(bad) == 0:
        return len(counts) - 1
    return bad[0] + 1

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    primes = _primes_up_to(1000)
    n_primes = len(primes)
    cap = 100000
    
    # Initialize residues to 0
    residues = np.zeros(n_primes, dtype=np.int64)
    
    # Initialize coverage counts
    counts = np.zeros(cap + 1, dtype=np.int32)
    for i, p in enumerate(primes):
        r = residues[i]
        start = r if r > 0 else p
        if start <= cap:
            counts[start::p] += 1
            
    first_gap = _find_first_gap(counts)
    best_L = first_gap
    best_residues = residues.copy()
    
    rng = np.random.RandomState(42)
    stuck_count = 0
    
    while time.time() < deadline:
        if first_gap >= cap:
            break
            
        u = first_gap
        improved = False
        
        # Try to fix u by changing one residue
        # Shuffle primes to avoid deterministic bias
        perm = rng.permutation(n_primes)
        
        for i in perm:
            if time.time() > deadline - 5:
                break
                
            p = primes[i]
            current_r = residues[i]
            target_r = u % p
            
            if current_r == target_r:
                continue
                
            # Apply move
            old_start = current_r if current_r > 0 else p
            new_start = target_r if target_r > 0 else p
            
            if old_start <= cap:
                counts[old_start::p] -= 1
            if new_start <= cap:
                counts[new_start::p] += 1
                
            new_first_gap = _find_first_gap(counts)
            
            if new_first_gap > first_gap:
                # Accept move
                first_gap = new_first_gap
                residues[i] = target_r
                improved = True
                stuck_count = 0
                
                if first_gap > best_L:
                    best_L = first_gap
                    best_residues = residues.copy()
                    
                break # Restart search for the new gap
            else:
                # Undo move
                if new_start <= cap:
                    counts[new_start::p] -= 1
                if old_start <= cap:
                    counts[old_start::p] += 1
        
        if not improved:
            stuck_count += 1
            # If stuck, perform a random kick to escape local optimum
            if stuck_count > 10:
                # Randomly change 5 residues
                for _ in range(5):
                    i = rng.randint(n_primes)
                    p = primes[i]
                    old_r = residues[i]
                    new_r = rng.randint(p)
                    
                    old_start = old_r if old_r > 0 else p
                    new_start = new_r if new_r > 0 else p
                    
                    if old_start <= cap:
                        counts[old_start::p] -= 1
                    if new_start <= cap:
                        counts[new_start::p] += 1
                    residues[i] = new_r
                    
                first_gap = _find_first_gap(counts)
                stuck_count = 0
                
    return [int(r) for r in best_residues]
