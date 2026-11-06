import numpy as np
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # 1. Generate Primes
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(n)) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(1000)
    n_primes = len(primes)
    primes_np = np.array(primes)
    
    # The range of integers to check.
    # We check up to 2000 to give some room.
    MAX_CHECK = 2000
    x_values = np.arange(1, MAX_CHECK + 1)
    
    # Precompute x mod p for all x and p.
    # Shape: (MAX_CHECK, n_primes)
    x_mod_p = x_values[:, None] % primes_np[None, :]
    
    # Initial Solution
    # Start with r_p = 1 for all primes.
    current_residues = np.ones(n_primes, dtype=np.int64)
    
    # Helper: Compute the current score (L)
    def compute_score(residues):
        match = (x_mod_p == residues[None, :])
        counts = match.sum(axis=1)
        invalid = counts < 2
        if not np.any(invalid):
            return MAX_CHECK
        else:
            first_invalid_idx = np.argmax(invalid)
            return int(first_invalid_idx)
    
    current_score = compute_score(current_residues)
    best_residues = current_residues.copy()
    best_score = current_score
    
    # Strategy: Greedy Frontier Extension with "Best Single Move"
    # This is the core of the current program's logic, which we are keeping.
    # We iterate x from 1 to MAX_CHECK.
    # For each x, if it is not doubly covered, we find the best prime to change.
    
    while time.time() < deadline - 1.0:
        if current_score >= MAX_CHECK:
            break
            
        gap_x = current_score + 1
        gap_idx = current_score  # index in x_values (0-based, x_values[0]=1)
        
        # Find primes that do NOT currently match gap_x
        match_at_gap = (x_mod_p[gap_idx] == current_residues)
        candidates = np.where(~match_at_gap)[0]
        
        if len(candidates) == 0:
            break
        
        # Vectorized evaluation of all candidate moves
        M = (x_mod_p == current_residues[None, :])
        old_counts = M.sum(axis=1)
        
        C = candidates
        R = x_mod_p[gap_idx, C]  # The new residue for each candidate prime is gap_x % p
        
        x_mod_p_cands = x_mod_p[:, C]
        new_matches_cands = (x_mod_p_cands == R[None, :])
        M_cands = M[:, C]
        
        delta_counts = new_matches_cands.astype(np.int8) - M_cands.astype(np.int8)
        new_counts = old_counts[:, None].astype(np.int8) + delta_counts
        
        invalid = new_counts < 2
        has_gap = invalid.any(axis=0)
        first_gap_idx = np.argmax(invalid, axis=0)
        scores = np.where(has_gap, first_gap_idx, MAX_CHECK)
        
        best_idx_in_scores = np.argmax(scores)
        best_candidate_idx = C[best_idx_in_scores]
        best_new_score = scores[best_idx_in_scores]
        
        if best_new_score > current_score:
            current_residues[best_candidate_idx] = R[best_idx_in_scores]
            current_score = best_new_score
            
            if current_score > best_score:
                best_score = current_score
                best_residues = current_residues.copy()
        else:
            # Stuck. No single move improves the score.
            old_count_at_gap = old_counts[gap_idx]
            
            if old_count_at_gap < 2:
                # Accept move if it keeps score same but increases count at gap?
                if best_new_score >= current_score:
                    current_residues[best_candidate_idx] = R[best_idx_in_scores]
                    current_score = best_new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_residues = current_residues.copy()
                else:
                    # Fallback: random perturbation
                    rand_idx = np.random.randint(0, n_primes)
                    old_r = current_residues[rand_idx]
                    new_r = np.random.randint(0, primes[rand_idx])
                    if new_r != old_r:
                        current_residues[rand_idx] = new_r
                        new_score = compute_score(current_residues)
                        if new_score >= current_score:
                            current_score = new_score
                            if new_score > best_score:
                                best_score = new_score
                                best_residues = current_residues.copy()
                        else:
                            current_residues[rand_idx] = old_r

    return [int(r) for r in best_residues]

# Ensure numpy random seed is set for reproducibility if needed
np.random.seed(42)

if __name__ == "__main__":
    res = construct()
    print(len(res))
