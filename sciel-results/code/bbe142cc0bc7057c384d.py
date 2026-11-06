import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Generate primes <= 1000
    limit = 1000
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    primes = np.where(sieve)[0]
    n_primes = len(primes)
    
    # Seed initialization
    # The i-th prime covers ceil(i/2)
    residues = np.array([((i + 2) // 2) % p for i, p in enumerate(primes)], dtype=int)
    
    # Helper to compute score from counts
    # L is largest integer such that 1..L all have count >= 2
    # We check up to M. If all 1..M have count >= 2, score is M (or higher, but capped).
    # The task says capped at 10^5, but L is likely small.
    # Let's check up to M=5000.
    M = 5000
    
    def compute_score(counts):
        # counts is 0-indexed, we care about 1..M
        # Find first index i in 1..M where counts[i] < 2
        # If none, return M
        mask = counts[1:M+1] < 2
        if not np.any(mask):
            return M
        first_fail = np.argmax(mask) + 1
        return first_fail - 1

    # Initialize counts
    counts = np.zeros(M + 1, dtype=int)
    for i, p in enumerate(primes):
        r = residues[i]
        counts[r::p] += 1
        
    current_score = compute_score(counts)
    best_score = current_score
    best_residues = residues.copy()
    
    N_TRIES = 20 # Number of random candidates to try per prime per pass
    
    while time.time() < deadline:
        improved = False
        # Shuffle order of primes to avoid bias
        order = np.random.permutation(n_primes)
        
        for k in order:
            if time.time() > deadline:
                break
                
            p = primes[k]
            old_r = residues[k]
            
            best_r_in_pass = old_r
            best_score_in_pass = current_score
            best_counts_in_pass = None # We won't store full counts, just revert if needed
            
            # Try N_TRIES random values
            for _ in range(N_TRIES):
                new_r = np.random.randint(0, p)
                if new_r == old_r:
                    continue
                    
                # Update counts
                counts[old_r::p] -= 1
                counts[new_r::p] += 1
                
                new_score = compute_score(counts)
                
                if new_score > best_score_in_pass:
                    best_score_in_pass = new_score
                    best_r_in_pass = new_r
                    # We don't keep counts, we revert at end of trial loop unless accepted
                    # But wait, if we accept, we need the counts to correspond to best_r_in_pass
                    # Since we revert every trial, we must re-apply the best one at the end
                    # Or we can just remember the best_r and re-apply it
                    pass
                else:
                    # Revert
                    counts[new_r::p] -= 1
                    counts[old_r::p] += 1
            
            # If we found a better residue
            if best_r_in_pass != old_r:
                # Apply the best change
                counts[old_r::p] -= 1
                counts[best_r_in_pass::p] += 1
                residues[k] = best_r_in_pass
                current_score = best_score_in_pass
                improved = True
                
                if current_score > best_score:
                    best_score = current_score
                    best_residues = residues.copy()
        
        if not improved:
            # Local optimum reached
            # Try a random kick to escape?
            # Change 5 random residues
            if time.time() < deadline - 5.0:
                kick_indices = np.random.choice(n_primes, 5, replace=False)
                for k in kick_indices:
                    p = primes[k]
                    old_r = residues[k]
                    new_r = np.random.randint(0, p)
                    if new_r != old_r:
                        counts[old_r::p] -= 1
                        counts[new_r::p] += 1
                        residues[k] = new_r
                current_score = compute_score(counts)
                if current_score > best_score:
                    best_score = current_score
                    best_residues = residues.copy()
            else:
                break

    return best_residues.tolist()
