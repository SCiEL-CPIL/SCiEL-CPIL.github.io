"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _sieve_primes(limit):
    """Return list of primes up to limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

def _score(residues, primes, max_check=100000):
    """Compute the largest L such that every integer 1..L satisfies at least two congruences."""
    n = len(primes)
    counts = np.zeros(max_check + 1, dtype=np.int32)
    for r, p in zip(residues, primes):
        start = r if r > 0 else p
        if start <= max_check:
            counts[start::p] += 1
    
    below_two = np.where(counts[1:] < 2)[0]
    if len(below_two) == 0:
        return max_check
    return below_two[0]

def _score_up_to(residues, primes, limit):
    """Compute coverage only up to 'limit' for speed. Returns (L, counts) where L is first gap."""
    counts = np.zeros(limit + 1, dtype=np.int32)
    for r, p in zip(residues, primes):
        start = r if r > 0 else p
        if start <= limit:
            counts[start::p] += 1
    
    below_two = np.where(counts[1:] < 2)[0]
    if len(below_two) == 0:
        return limit, counts
    return below_two[0], counts

def construct():
    primes = _sieve_primes(1000)
    n = len(primes)
    
    # Deadline guard - leave safety margin
    deadline = time.time() + 145
    
    # Start with the seed solution
    best_residues = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    best_score = _score(best_residues, primes)
    
    # Local search: iteratively try to improve coverage
    improved = True
    max_rounds = 200
    round_num = 0
    
    while improved and round_num < max_rounds and time.time() < deadline:
        improved = False
        round_num += 1
        
        # Find the current gap position
        gap_pos = best_score + 1
        if gap_pos > 100000:
            break
        
        # For each prime, try to find a residue that helps cover the gap
        # and doesn't create new gaps before the current best
        for i in range(n):
            if time.time() > deadline:
                break
            
            p = primes[i]
            current_r = best_residues[i]
            
            # Generate strategic candidates:
            # 1. Residues that would hit gap_pos: r ≡ gap_pos (mod p)
            # 2. Neighbors of current residue
            # 3. Some random candidates
            
            candidates = set()
            
            # Candidate that directly hits the gap
            r_hit_gap = gap_pos % p
            candidates.add(r_hit_gap)
            
            # Neighbors
            for delta in range(-3, 4):
                candidates.add((current_r + delta) % p)
            
            # A few random candidates (more for smaller primes where they matter more)
            n_random = max(2, min(8, 50 // p))
            for _ in range(n_random):
                candidates.add(np.random.randint(0, p))
            
            # Limit candidates to keep search fast
            candidate_list = list(candidates)
            if len(candidate_list) > 15:
                # Keep the most strategic ones
                candidate_list = [r_hit_gap] + [c for c in candidate_list if c != r_hit_gap][:14]
            
            for new_r in candidate_list:
                if new_r == current_r:
                    continue
                
                test_residues = list(best_residues)
                test_residues[i] = new_r
                
                # Quick check: does this extend the score?
                # Only need to check up to best_score + some margin
                check_limit = min(best_score + 200, 100000)
                s_test, _ = _score_up_to(test_residues, primes, check_limit)
                
                if s_test > best_score:
                    # Verify with full score
                    s_full = _score(test_residues, primes)
                    if s_full > best_score:
                        best_residues = test_residues
                        best_score = s_full
                        improved = True
                        break  # Move to next i
    
    return best_residues
# EVOLVE-BLOCK-END
