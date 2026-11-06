"""Greedy set-cover heuristic for double-covering prefix [1, L]."""
import time
import math
import numpy as np

# Deadline: 140s wall-clock (10s safety margin before 150s kill)
_DEADLINE = 140.0
_START_TIME = time.time()

def _primes_upto(n):
    """Return list of primes <= n."""
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0:2] = b'\x00\x00'
    for i in range(2, int(math.isqrt(n)) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(n + 1) if sieve[i]]

def _compute_L(residues, primes, max_check=100000):
    """Compute the largest L such that every integer 1..L satisfies at least 2 congruences.
    Returns L (capped at max_check)."""
    n = len(primes)
    # For each prime index i, residue r_i, an integer x is covered by prime i iff x % p_i == r_i
    # We need count of covering primes >= 2 for each x in 1..max_check.
    
    # Efficient approach: for each prime, mark which positions in [1, max_check] it covers.
    # Use a numpy array of counts.
    counts = np.zeros(max_check + 1, dtype=np.int32)
    
    for i, p in enumerate(primes):
        r = residues[i]
        # Positions x in [1, max_check] where x % p == r
        # First such position: if r == 0, it's p, 2p, ...; else r, r+p, r+2p, ...
        # But we need x >= 1.
        if r == 0:
            start = p
        else:
            start = r
        if start > max_check:
            continue
        # Mark all positions start, start+p, start+2p, ... <= max_check
        idx = np.arange(start, max_check + 1, p)
        counts[idx] += 1
    
    # Find the largest L such that counts[1..L] are all >= 2
    # Find first position where count < 2
    deficient = np.where(counts[1:] < 2)[0]
    if len(deficient) == 0:
        return max_check
    first_deficient = deficient[0] + 1  # 1-indexed
    return first_deficient - 1

def _greedy_construct(primes):
    """Greedy set-cover: process primes in increasing order, for each prime choose residue
    that maximizes the number of integers in [1, current_L] that are covered by exactly one
    congruence (among already-processed primes), thereby converting single-covered to double-covered."""
    n = len(primes)
    residues = [0] * n
    
    # We'll track coverage counts incrementally.
    # Start with no primes processed, all counts = 0.
    # We want to extend L as far as possible.
    
    # Strategy: process primes one by one. For each new prime, choose the residue that
    # maximizes the extension of the doubly-covered prefix.
    
    # To evaluate a residue choice for prime p, we need to know how many of the first
    # k primes cover each position. This is expensive to recompute each time.
    
    # Alternative: use a fixed window size and evaluate locally.
    # Let's try a different approach: iterative improvement.
    
    # Start with a reasonable initial guess, then hill-climb.
    
    # Initial: for prime i (0-indexed), set residue to (i+1) % p or similar.
    # Let's use the seed approach as initial: r_i = ceil((i+1)/2) % p
    for i, p in enumerate(primes):
        residues[i] = ((i + 2) // 2) % p
    
    current_L = _compute_L(residues, primes)
    
    # Hill-climbing: try changing each residue to see if L improves.
    improved = True
    while improved and (time.time() - _START_TIME) < _DEADLINE - 5:
        improved = False
        for i, p in enumerate(primes):
            old_r = residues[i]
            best_r = old_r
            best_L = current_L
            
            # Try a few candidate residues: 0, 1, 2, ..., up to min(p, 20) candidates
            # Plus the current one.
            candidates = set()
            candidates.add(old_r)
            for c in range(min(p, 30)):
                candidates.add(c)
            # Also try some random ones if p is large
            if p > 30:
                import random
                for _ in range(10):
                    candidates.add(random.randint(0, p - 1))
            
            for r in candidates:
                if r == old_r:
                    continue
                residues[i] = r
                L = _compute_L(residues, primes)
                if L > best_L:
                    best_L = L
                    best_r = r
                elif L == best_L and r < best_r:
                    # Tie-break: prefer smaller residue (arbitrary)
                    pass
                residues[i] = old_r
            
            if best_r != old_r:
                residues[i] = best_r
                current_L = best_L
                improved = True
    
    return residues, current_L

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    primes = _primes_upto(1000)
    n = len(primes)
    
    # Quick sanity check
    assert n == 168, f"Expected 168 primes, got {n}"
    
    # Run greedy construction
    residues, L = _greedy_construct(primes)
    
    # Verify
    final_L = _compute_L(residues, primes)
    
    # Ensure residues are valid
    for i, p in enumerate(primes):
        assert 0 <= residues[i] < p, f"Invalid residue {residues[i]} for prime {p}"
    
    return residues

# Precompute primes at module level for speed
_PRIMES = _primes_upto(1000)
