"""Structural redesign: greedy construction with local search refinement.

Strategy:
1. Greedily build residues by processing primes in increasing order. For each
   prime p, choose the residue r_p that maximizes the number of x in a target
   window that get an additional (second) coverage from this prime, subject to
   the constraint that we don't create a gap in the current double-covered prefix.
2. After greedy construction, run local search: repeatedly pick a prime and
   try changing its residue to improve the double-covered prefix length.
3. Use time-bounded search with a safety margin.
"""
import time
import random
import math

# Precompute primes <= 1000
def _get_primes(limit=1000):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i in range(limit + 1) if sieve[i]]

PRIMES = _get_primes(1000)
N_PRIMES = len(PRIMES)  # 168

# Maximum L we'll ever need to check (capped at 10^5 per task)
MAX_L = 100000

def _compute_L(residues, primes, max_check=MAX_L):
    """Compute the largest L such that every integer 1..L is covered by >= 2 congruences.
    
    Uses a difference-array approach for efficiency.
    """
    n = len(primes)
    # For each prime, compute which positions in 1..max_check it covers
    # coverage[x] = number of primes p where x % p == r_p
    # We need to find the first x where coverage[x] < 2
    
    # Efficient approach: use an array of counts
    # For each prime p with residue r, it covers positions r, r+p, r+2p, ... (and 0 if r==0)
    # In range 1..max_check, positions are: r, r+p, r+2p, ... <= max_check
    # But also if r == 0, positions are p, 2p, 3p, ... (since x=0 is not in 1..max_check)
    
    coverage = [0] * (max_check + 1)  # 1-indexed, index 0 unused
    
    for i, p in enumerate(primes):
        r = residues[i]
        # Find first position >= 1 that is congruent to r mod p
        if r == 0:
            start = p  # first positive multiple of p
        else:
            start = r
            if start == 0:
                start = p
        # Actually: x ≡ r (mod p), x >= 1
        # If r >= 1: x = r, r+p, r+2p, ...
        # If r == 0: x = p, 2p, 3p, ...
        if r == 0:
            first = p
        else:
            first = r
        for x in range(first, max_check + 1, p):
            coverage[x] += 1
    
    # Find the largest L such that all 1..L have coverage >= 2
    L = 0
    for x in range(1, max_check + 1):
        if coverage[x] < 2:
            break
        L = x
    return L


def _compute_L_fast(residues, primes, max_check=MAX_L):
    """Faster L computation using numpy-style vectorized approach with pure Python lists.
    
    Actually, let's use a smarter incremental approach.
    """
    n = len(primes)
    coverage = [0] * (max_check + 1)
    
    for i, p in enumerate(primes):
        r = residues[i]
        if r == 0:
            first = p
        else:
            first = r
        for x in range(first, max_check + 1, p):
            coverage[x] += 1
    
    L = 0
    for x in range(1, max_check + 1):
        if coverage[x] < 2:
            break
        L = x
    return L


def _greedy_construct(primes, target_window=2000):
    """Greedily construct residues to maximize double coverage in a window."""
    n = len(primes)
    residues = [0] * n
    
    # We'll track coverage in a window [1, target_window]
    # For each prime, choose the residue that adds the most "second coverage"
    # i.e., positions that currently have exactly 1 coverage and will get a second
    
    window = target_window
    coverage = [0] * (window + 1)  # 1-indexed
    
    for i, p in enumerate(primes):
        best_r = 0
        best_score = -1
        best_new_L = -1
        
        # Try each possible residue
        for r in range(p):
            # Compute how many new positions get their second coverage
            if r == 0:
                first = p
            else:
                first = r
            
            # Count positions in [1, window] that currently have coverage == 1
            # and will be covered by this residue
            gain = 0
            for x in range(first, window + 1, p):
                if coverage[x] == 1:
                    gain += 1
                elif coverage[x] >= 2:
                    # Already doubly covered, no harm but no gain
                    pass
                # If coverage[x] == 0, this would make it 1, not 2 yet
            # Also penalize if we break the prefix (positions before current L with coverage 0)
            # Actually we need to ensure no position in 1..current_L drops below 2
            # Since we're only adding, coverage never drops. So the prefix is safe.
            
            if gain > best_score:
                best_score = gain
                best_r = r
        
        residues[i] = best_r
        # Update coverage
        r = best_r
        if r == 0:
            first = p
        else:
            first = r
        for x in range(first, window + 1, p):
            coverage[x] += 1
    
    return residues


def _local_search(residues, primes, time_limit=100.0, max_iterations=50000):
    """Local search: try changing individual residues to improve L."""
    n = len(primes)
    best_residues = list(residues)
    best_L = _compute_L_fast(best_residues, primes, max_check=min(5000, MAX_L))
    
    start_time = time.time()
    rng = random.Random(42)
    
    # Use a smaller check window for speed during search, then verify at the end
    check_window = 2000
    
    for it in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random prime to modify
        i = rng.randrange(n)
        p = primes[i]
        old_r = best_residues[i]
        
        # Try a few random alternative residues
        for _ in range(5):
            new_r = rng.randrange(p)
            if new_r == old_r:
                continue
            
            best_residues[i] = new_r
            new_L = _compute_L_fast(best_residues, primes, max_check=check_window)
            
            if new_L > best_L:
                best_L = new_L
                # Accept and continue
                break
            else:
                best_residues[i] = old_r
        else:
            # No improvement found for this prime in this round
            pass
    
    return best_residues, best_L


def _refine_with_larger_window(residues, primes, time_limit=20.0):
    """Refine using a larger check window for more accurate L estimation."""
    n = len(primes)
    best_residues = list(residues)
    best_L = _compute_L_fast(best_residues, primes, max_check=min(5000, MAX_L))
    
    start_time = time.time()
    rng = random.Random(123)
    check_window = 5000
    
    for it in range(20000):
        if time.time() - start_time > time_limit:
            break
        
        i = rng.randrange(n)
        p = primes[i]
        old_r = best_residues[i]
        
        for _ in range(3):
            new_r = rng.randrange(p)
            if new_r == old_r:
                continue
            
            best_residues[i] = new_r
            new_L = _compute_L_fast(best_residues, primes, max_check=check_window)
            
            if new_L > best_L:
                best_L = new_L
                break
            else:
                best_residues[i] = old_r
    
    return best_residues, best_L


def construct():
    """Construct 168 residues, one for each prime p <= 1000, to maximize double coverage."""
    start_time = time.time()
    time_limit = 120.0  # Safety margin
    
    # Phase 1: Greedy construction
    residues = _greedy_construct(PRIMES, target_window=3000)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > time_limit * 0.5:
        # Not enough time for search, return greedy result
        return residues
    
    # Phase 2: Local search with small window
    residues, L_small = _local_search(residues, PRIMES, time_limit=min(60.0, time_limit - elapsed - 30.0))
    
    elapsed = time.time() - start_time
    if elapsed > time_limit * 0.7:
        # Verify with larger window but don't have time for more search
        final_L = _compute_L_fast(residues, PRIMES, max_check=MAX_L)
        return residues
    
    # Phase 3: Refine with larger window
    residues, L_medium = _refine_with_larger_window(residues, PRIMES, time_limit=min(30.0, time_limit - elapsed - 15.0))
    
    elapsed = time.time() - start_time
    if elapsed < time_limit - 15.0:
        # One more round of local search
        residues, L_final = _local_search(residues, PRIMES, time_limit=time_limit - elapsed - 10.0)
    
    # Final verification (not needed for return, but good to know)
    # final_L = _compute_L_fast(residues, PRIMES, max_check=MAX_L)
    
    return residues
