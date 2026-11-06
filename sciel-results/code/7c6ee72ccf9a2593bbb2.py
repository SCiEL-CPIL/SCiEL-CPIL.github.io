"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START

def _primes_up_to(n):
    """Generate all primes up to n using sieve of Eratosthenes."""
    if n < 2:
        return []
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]

def _compute_L(residues, primes, max_check=100000):
    """
    Compute the largest L such that every integer 1..L satisfies at least
    two of the congruences x = r_p (mod p).
    """
    n_primes = len(primes)
    # counts[x] = number of primes p where x % p == r_p
    # We need counts[x] >= 2 for all x in 1..L
    # Use numpy for speed
    import numpy as np
    
    x = np.arange(1, max_check + 1, dtype=np.int32)
    counts = np.zeros(max_check, dtype=np.int32)
    
    for i, p in enumerate(primes):
        r = residues[i]
        # x % p == r  =>  x in {r, r+p, r+2p, ...} within [1, max_check]
        # First x >= 1 with x % p == r:
        # if r == 0: first is p (since x>=1)
        # else: first is r (if r>=1)
        if r == 0:
            start = p
        else:
            start = r
        if start < 1:
            start += p
        # Mark all x in [start, max_check] with step p
        if start <= max_check:
            idx = np.arange(start, max_check + 1, p, dtype=np.int32)
            counts[idx - 1] += 1
    
    # Find the largest L such that counts[0:L] are all >= 2
    # i.e., the first index where counts < 2
    deficient = np.where(counts < 2)[0]
    if len(deficient) == 0:
        return max_check
    else:
        return int(deficient[0])  # 0-indexed, so this is the first deficient x

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s out of 150s budget
    
    primes = _primes_up_to(1000)
    n_primes = len(primes)
    
    # Initial solution: the seed
    best_residues = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    best_L = _compute_L(best_residues, primes)
    
    # Local search: try to improve by changing one residue at a time
    # For each prime, try all possible residues and keep the best improvement
    
    random.seed(42)
    
    # Precompute for faster evaluation
    import numpy as np
    
    def compute_L_fast(residues):
        """Faster L computation using numpy vectorization."""
        max_check = 100000
        x = np.arange(1, max_check + 1, dtype=np.int32)
        counts = np.zeros(max_check, dtype=np.int32)
        
        for i, p in enumerate(primes):
            r = residues[i]
            if r == 0:
                start = p
            else:
                start = r
            if start < 1:
                start += p
            if start <= max_check:
                idx = np.arange(start, max_check + 1, p, dtype=np.int32)
                counts[idx - 1] += 1
        
        deficient = np.where(counts < 2)[0]
        if len(deficient) == 0:
            return max_check
        else:
            return int(deficient[0])
    
    # Try to improve using local search
    improved = True
    while improved and (time.time() < deadline - 5):
        improved = False
        for i in range(n_primes):
            if time.time() > deadline - 5:
                break
            p = primes[i]
            current_r = best_residues[i]
            # Try a few random residues, plus neighbors
            candidates = set()
            candidates.add((current_r - 1) % p)
            candidates.add((current_r + 1) % p)
            # Add some random ones
            for _ in range(min(10, p)):
                candidates.add(random.randint(0, p - 1))
            
            for new_r in candidates:
                if new_r == current_r:
                    continue
                trial = best_residues.copy()
                trial[i] = new_r
                trial_L = compute_L_fast(trial)
                if trial_L > best_L:
                    best_residues = trial
                    best_L = trial_L
                    improved = True
                    break
            # If we improved, restart the scan
            if improved:
                break
    
    # Also try a greedy approach: for each position x that's deficient,
    # try to fix it by adjusting residues of primes that could cover it
    if time.time() < deadline - 5:
        # Greedy repair
        for iteration in range(20):
            if time.time() > deadline - 5:
                break
            # Find first deficient x
            max_check = 100000
            x_arr = np.arange(1, max_check + 1, dtype=np.int32)
            counts = np.zeros(max_check, dtype=np.int32)
            for i, p in enumerate(primes):
                r = best_residues[i]
                if r == 0:
                    start = p
                else:
                    start = r
                if start < 1:
                    start += p
                if start <= max_check:
                    idx = np.arange(start, max_check + 1, p, dtype=np.int32)
                    counts[idx - 1] += 1
            
            deficient = np.where(counts < 2)[0]
            if len(deficient) == 0:
                break
            
            x_def = int(deficient[0]) + 1  # 1-indexed
            
            # For each prime, check if changing its residue to cover x_def
            # would help without hurting other positions too much
            for i, p in enumerate(primes):
                if time.time() > deadline - 5:
                    break
                # What residue would cover x_def?
                r_needed = x_def % p
                if r_needed == best_residues[i]:
                    continue
                # Check if this prime currently covers x_def (it doesn't, since it's deficient)
                # Try changing residue
                trial = best_residues.copy()
                trial[i] = r_needed
                trial_L = compute_L_fast(trial)
                if trial_L > best_L:
                    best_residues = trial
                    best_L = trial_L
                    break
    
    return best_residues

# EVOLVE-BLOCK-END
