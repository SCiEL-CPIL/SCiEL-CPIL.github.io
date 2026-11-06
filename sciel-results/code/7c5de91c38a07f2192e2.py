"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = time.time() + 140  # safety margin before 150s kill

def _sieve_primes(limit):
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i, v in enumerate(sieve) if v]

def _compute_L(residues, primes):
    """Compute the largest L such that every integer 1..L is covered by >=2 congruences."""
    n = len(primes)
    max_L = 100000
    # Build coverage array using numpy
    # coverage[x] = number of primes p_i such that x ≡ r_i (mod p_i)
    # We only need to check up to max_L
    # For each prime p_i with residue r_i, the covered numbers in 1..max_L are:
    # r_i, r_i + p_i, r_i + 2*p_i, ... up to max_L
    # But r_i could be 0, meaning multiples of p_i
    # Let's use a vectorized approach: for each prime, mark its coverage
    
    coverage = np.zeros(max_L + 1, dtype=np.int32)
    for i, (p, r) in enumerate(zip(primes, residues)):
        # Numbers x in [1, max_L] where x ≡ r (mod p)
        # If r == 0: x = p, 2p, 3p, ...
        # If r > 0: x = r, r+p, r+2p, ...
        if r == 0:
            start = p
        else:
            start = r
        if start > max_L:
            continue
        # Mark all numbers from start to max_L with step p
        coverage[start::p] += 1
    
    # Find largest L such that coverage[1..L] >= 2
    # Use cumulative check
    valid = coverage[1:] >= 2
    # Find first position where valid is False
    false_positions = np.where(~valid)[0]
    if len(false_positions) == 0:
        return max_L
    first_bad = false_positions[0] + 1  # 1-indexed
    return first_bad - 1

def construct():
    start_time = time.time()
    primes = _sieve_primes(1000)
    n = len(primes)
    
    # Initial assignment: seed-style
    residues = np.array([((i + 2) // 2) % p for i, p in enumerate(primes)], dtype=np.int32)
    
    best_L = _compute_L(residues, primes)
    best_residues = residues.copy()
    
    # Greedy coordinate descent: for each prime, try all residues and pick the best
    # Use numpy to evaluate multiple residues at once for efficiency
    
    # Precompute: for each prime index, we'll try changing its residue
    # To speed up, we can evaluate the effect of changing one residue at a time
    
    # Strategy: iterate over primes, for each prime try a batch of candidate residues
    # and pick the one that maximizes L
    
    def _try_change_prime(residues, prime_idx, candidate_residues):
        """Try changing residue of prime_idx to each value in candidate_residues, return best."""
        old_r = residues[prime_idx]
        p = primes[prime_idx]
        best_local_L = _compute_L(residues, primes)
        best_local_r = old_r
        
        for new_r in candidate_residues:
            if new_r == old_r:
                continue
            residues[prime_idx] = new_r
            L = _compute_L(residues, primes)
            if L > best_local_L:
                best_local_L = L
                best_local_r = new_r
        residues[prime_idx] = best_local_r
        return best_local_L, best_local_r
    
    # Main optimization loop
    no_improve = 0
    max_no_improve = 30
    
    while time.time() < _DEADLINE:
        improved = False
        
        # Pass 1: try all primes in order
        for idx in range(n):
            if time.time() > _DEADLINE:
                break
            p = primes[idx]
            # Try all possible residues for this prime
            # For small primes, try all; for large primes, try a subset
            if p <= 50:
                candidates = list(range(p))
            else:
                # Try current residue plus some random ones
                current = best_residues[idx]
                candidates = [current]
                # Add some structured candidates
                for offset in [1, -1, 2, -2, p//2, (p+1)//2]:
                    c = (current + offset) % p
                    if c not in candidates:
                        candidates.append(c)
                # Add a few random ones
                for _ in range(5):
                    c = np.random.randint(0, p)
                    if c not in candidates:
                        candidates.append(c)
            
            trial_residues = best_residues.copy()
            best_local_L, best_local_r = _try_change_prime(trial_residues, idx, candidates)
            
            if best_local_L > best_L:
                best_L = best_local_L
                best_residues[idx] = best_local_r
                improved = True
                no_improve = 0
            else:
                no_improve += 1
        
        if not improved:
            # Try random perturbations
            for _ in range(20):
                if time.time() > _DEADLINE:
                    break
                idx = np.random.randint(0, n)
                p = primes[idx]
                new_r = np.random.randint(0, p)
                trial_residues = best_residues.copy()
                trial_residues[idx] = new_r
                L = _compute_L(trial_residues, primes)
                if L > best_L:
                    best_L = L
                    best_residues[idx] = new_r
                    improved = True
                    no_improve = 0
        
        if no_improve >= max_no_improve:
            # Reset and try a different initial configuration
            break
    
    return best_residues.tolist()
# EVOLVE-BLOCK-END
