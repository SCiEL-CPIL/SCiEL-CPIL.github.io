"""Rewritten program: local search on residues to maximize doubly-covered prefix."""
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
    # Build a coverage count array for 1..max_check
    # For each prime p with residue r, mark positions r mod p, r+p mod p, etc.
    # We'll use a numpy array of counts.
    
    counts = np.zeros(max_check + 1, dtype=np.int32)
    for r, p in zip(residues, primes):
        # Positions in 1..max_check that are ≡ r (mod p)
        # The first position >= 1 that is ≡ r (mod p):
        # If r == 0, positions are p, 2p, 3p, ...
        # If r > 0, positions are r, r+p, r+2p, ... but need to be >= 1
        # Actually r is in [0, p-1]. The smallest positive integer ≡ r (mod p) is:
        #   if r == 0: p
        #   if r > 0: r (since 1 <= r <= p-1)
        start = p if r == 0 else r
        # Now mark start, start+p, start+2p, ... up to max_check
        if start <= max_check:
            counts[start::p] += 1
    
    # Find the largest L such that counts[1..L] are all >= 2
    # We can find the first position where count < 2
    if counts[1] < 2:
        return 0
    # Use searchsorted or argmax on (counts < 2)
    bad = np.where(counts < 2)[0]
    # We only care about positions >= 1
    bad = bad[bad >= 1]
    if len(bad) == 0:
        return max_check
    return bad[0] - 1

def _initial_residues(primes):
    """Generate initial residues: cover integer ceil(i/2) for the i-th prime."""
    return [((i + 2) // 2) % p for i, p in enumerate(primes)]

def construct():
    primes = _sieve_primes(1000)
    n = len(primes)
    
    # Start with a reasonable initial assignment
    residues = _initial_residues(primes)
    current_score = _score(residues, primes)
    
    deadline = time.time() + 110  # Safety margin before 120s limit
    best_residues = residues[:]
    best_score = current_score
    
    # Local search: hill-climbing on individual residues
    # For each prime, try changing its residue to see if it improves the score
    # We focus on extending the doubly-covered prefix
    
    # Precompute for efficiency
    max_check = 100000
    
    # We'll do multiple passes of local search
    improved = True
    pass_count = 0
    while improved and time.time() < deadline:
        improved = False
        pass_count += 1
        
        for i in range(n):
            if time.time() > deadline:
                break
            p = primes[i]
            old_r = residues[i]
            
            # Try a few candidate residues
            # Strategy: try residues that might help fill gaps in the coverage
            # First, try to find where the current coverage breaks
            # Then try residues that cover those positions
            
            # Quick heuristic: try the old residue and a few others
            # For efficiency, we'll try a limited set of candidates
            candidates = set()
            candidates.add(old_r)
            
            # Try residues that cover the first gap
            # Find where current coverage breaks
            # But recomputing score each time is expensive
            # Instead, let's try a small number of random candidates plus structured ones
            
            # Structured: try to cover positions near the current break point
            # We know current_score, so the break is at current_score + 1
            # Try residues that make prime i cover current_score + 1
            target = best_score + 1
            if target <= max_check:
                candidates.add(target % p)
            
            # Also try a few random residues
            for _ in range(3):
                candidates.add(np.random.randint(0, p))
            
            for new_r in candidates:
                if new_r == old_r:
                    continue
                residues[i] = new_r
                new_score = _score(residues, primes)
                if new_score > best_score:
                    best_score = new_score
                    best_residues = residues[:]
                    improved = True
                    # Don't break; keep trying other candidates for this prime
                else:
                    residues[i] = old_r
                    break  # Only try first non-improving candidate for speed? No, try all candidates
            # Reset to best if we didn't improve
            if residues[i] != best_residues[i]:
                # This can happen if we tried multiple candidates
                # Actually, we reset to old_r if new_r didn't improve
                # But if multiple candidates were tried, we need to be careful
                pass
            residues[i] = old_r  # Ensure we reset after trying candidates
    
    # Restore best
    return best_residues
# EVOLVE-BLOCK-END
