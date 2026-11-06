"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _get_primes(limit):
    """Sieve of Eratosthenes to get primes up to limit."""
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

def _calculate_score(residues, primes, max_check=50000):
    """
    Calculate the largest L such that every integer 1..L satisfies at least two congruences.
    Uses vectorized counting for speed.
    """
    counts = np.zeros(max_check + 1, dtype=np.int32)
    
    for r, p in zip(residues, primes):
        start = r if r > 0 else p
        if start <= max_check:
            counts[start::p] += 1
            
    below_two = np.where(counts[1:] < 2)[0]
    if len(below_two) == 0:
        return max_check
    else:
        return below_two[0]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    primes = _get_primes(1000)
    n_primes = len(primes)
    
    # Better initial heuristic: try to cover small numbers densely
    # For each prime, pick a residue that helps cover positions 1..500
    residues = np.zeros(n_primes, dtype=np.int64)
    for i, p in enumerate(primes):
        # Pick residue to cover a small number: r = (small_target % p)
        # Cycle through small targets
        target = (i * 7) % 100 + 1
        residues[i] = target % p
    
    max_check = 50000
    counts = np.zeros(max_check + 1, dtype=np.int32)
    
    def update_counts(res, prim, delta):
        start = res if res > 0 else prim
        if start <= max_check:
            counts[start::prim] += delta

    def get_score_from_counts():
        below_two = np.where(counts[1:max_check+1] < 2)[0]
        if len(below_two) == 0:
            return max_check
        return below_two[0]
    
    # Initialize counts
    for r, p in zip(residues, primes):
        update_counts(r, p, 1)
        
    current_score = get_score_from_counts()
    best_score = current_score
    best_residues = residues.copy()
    
    no_improve_count = 0
    max_no_improve = 200
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            break
            
        improved = False
        
        # Get the first uncovered position to target
        below_two = np.where(counts[1:max_check+1] < 2)[0]
        if len(below_two) > 0:
            first_gap = below_two[0] + 1  # 1-indexed
        else:
            first_gap = max_check + 1
        
        # Strategy 1: Targeted - find primes that could cover first_gap
        # For prime p, residue r_p = first_gap % p would cover first_gap
        # But we need to be careful: changing r_p affects all positions r_p + k*p
        
        idxs = np.random.permutation(n_primes)
        
        for i in idxs:
            if time.time() > deadline:
                break
                
            p = primes[i]
            old_r = residues[i]
            
            # Generate candidate residues:
            # 1. Residue that covers first_gap: (first_gap % p)
            # 2. Some random residues
            candidates = set()
            
            # Targeted candidate
            if first_gap <= max_check:
                candidates.add(first_gap % p)
            
            # Also try residues near the gap
            for offset in [-2, -1, 0, 1, 2]:
                cand = (first_gap + offset) % p
                if 0 <= cand < p:
                    candidates.add(cand)
            
            # Random candidates
            for _ in range(5):
                candidates.add(np.random.randint(0, p))
            
            candidates = [c for c in candidates if c != old_r]
            
            best_candidate_score = -1
            best_candidate_r = None
            
            for new_r in candidates:
                if new_r == old_r:
                    continue
                
                update_counts(old_r, p, -1)
                residues[i] = new_r
                update_counts(new_r, p, 1)
                
                new_score = get_score_from_counts()
                
                if new_score > best_candidate_score:
                    best_candidate_score = new_score
                    best_candidate_r = new_r
                
                # Revert
                update_counts(new_r, p, -1)
                residues[i] = old_r
                update_counts(old_r, p, 1)
            
            if best_candidate_r is not None and best_candidate_score > best_score:
                # Apply the best candidate
                update_counts(old_r, p, -1)
                residues[i] = best_candidate_r
                update_counts(best_candidate_r, p, 1)
                
                best_score = best_candidate_score
                best_residues = residues.copy()
                improved = True
                no_improve_count = 0
        
        if not improved:
            no_improve_count += 1
        else:
            no_improve_count = 0
            
    return [int(r) for r in best_residues]
# EVOLVE-BLOCK-END
