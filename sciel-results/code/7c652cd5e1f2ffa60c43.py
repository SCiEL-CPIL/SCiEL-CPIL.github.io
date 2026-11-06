"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def _sieve_primes(limit):
    """Sieve of Eratosthenes to get primes up to limit."""
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

def _compute_counts(residues, primes, max_check):
    """
    Compute the number of congruences satisfied by each integer 1..max_check.
    Returns a numpy array of shape (max_check,) where counts[i-1] = number of primes p
    such that i % p == residues[idx] where idx is the index of p in primes.
    """
    counts = np.zeros(max_check, dtype=np.int32)
    for i, (p, r) in enumerate(zip(primes, residues)):
        # For each x in 1..max_check such that x % p == r, increment counts
        # x = r, r+p, r+2p, ... but x >= 1
        # So first x is max(r, 1) if r > 0, else p (since r=0 means x=p, 2p, ...)
        # Actually x % p == r means x = r + k*p for k >= 0, but x >= 1
        # If r == 0: x = p, 2p, 3p, ...
        # If r > 0: x = r, r+p, r+2p, ... but only if r >= 1 (which it is since 0 <= r < p and r > 0)
        
        if r == 0:
            start = p
        else:
            start = r
        
        if start <= max_check:
            # All x = start, start+p, start+2p, ... <= max_check
            # These are at indices start-1, start+p-1, ... in the counts array
            indices = np.arange(start - 1, max_check, p)
            counts[indices] += 1
    
    return counts

def _calculate_score(residues, primes, max_check=100000):
    """
    Calculate the largest L such that every integer 1..L satisfies at least two congruences.
    """
    counts = _compute_counts(residues, primes, max_check)
    # Find the first position where count < 2
    # positions are 1-indexed, so index 0 corresponds to position 1
    below_two = np.where(counts < 2)[0]
    if len(below_two) == 0:
        return max_check
    first_fail = below_two[0] + 1  # position where count < 2
    return first_fail - 1  # L = first_fail - 1

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    primes = _sieve_primes(1000)
    n_primes = len(primes)  # 168
    
    # Initial solution: round-robin style
    residues = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    
    # Compute initial score
    max_check = 100000
    current_score = _calculate_score(residues, primes, max_check)
    best_residues = residues.copy()
    best_score = current_score
    
    # Precompute some structures for speed
    # For each prime index, precompute the positions it covers for each possible residue
    # This is too memory intensive, so we'll do incremental updates
    
    # Local search: try changing one residue at a time
    # Focus on residues of smaller primes first (they affect more positions)
    # Actually, all primes affect different numbers of positions
    
    # Strategy: iterative improvement with restarts
    max_iterations = 50000
    iteration = 0
    
    while iteration < max_iterations:
        if time.time() > deadline:
            break
        
        # Pick a random prime index to modify
        # Bias towards smaller primes? Or uniform?
        # Let's try uniform first, then maybe bias
        idx = random.randint(0, n_primes - 1)
        old_r = residues[idx]
        
        # Try a few random new residues
        improved = False
        for _ in range(5):
            new_r = random.randint(0, primes[idx] - 1)
            if new_r == old_r:
                continue
            
            residues[idx] = new_r
            new_score = _calculate_score(residues, primes, max_check)
            
            if new_score > current_score:
                current_score = new_score
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_residues = residues.copy()
                break
            else:
                residues[idx] = old_r
        
        if not improved:
            # Try a more targeted approach: find the first failure point
            # and try to fix it
            counts = _compute_counts(residues, primes, max_check)
            below_two = np.where(counts < 2)[0]
            if len(below_two) > 0:
                fail_pos = below_two[0] + 1  # 1-indexed position
                
                # Try to find a prime that can cover this position
                # For position fail_pos, we need x % p == r for some p
                # Check which primes currently cover this position
                # and which don't
                
                # Try changing residues of primes that don't cover fail_pos
                # to make them cover it
                candidates = []
                for i, p in enumerate(primes):
                    needed_r = fail_pos % p
                    if residues[i] != needed_r:
                        candidates.append(i)
                
                if candidates:
                    # Try a few random candidates
                    random.shuffle(candidates)
                    for idx2 in candidates[:10]:
                        old_r2 = residues[idx2]
                        needed_r2 = fail_pos % primes[idx2]
                        residues[idx2] = needed_r2
                        new_score = _calculate_score(residues, primes, max_check)
                        
                        if new_score > current_score:
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best_residues = residues.copy()
                            break
                        else:
                            residues[idx2] = old_r2
        
        iteration += 1
    
    return best_residues
# EVOLVE-BLOCK-END
