import numpy as np
import time

def _primes_up_to(n):
    sieve = np.ones(n + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    return [int(p) for p in np.where(sieve)[0]]

def _compute_L(residues, primes, cap=100000):
    """Compute the largest L such that every integer 1..L has coverage >= 2."""
    n = len(primes)
    # For each x in 1..cap, count how many primes cover it
    # x is covered by prime p if x % p == residues[i]
    # Vectorize: for each prime, mark positions
    counts = np.zeros(cap + 1, dtype=np.int32)
    for i, p in enumerate(primes):
        r = residues[i]
        # Positions: r, r+p, r+2p, ... up to cap
        # But we want x >= 1, so start from max(r, 1)
        start = r
        if start < 1:
            start = r + p
        if start > cap:
            continue
        counts[start::p] += 1
    # Find largest L such that counts[1..L] all >= 2
    # Find first position where count < 2
    bad = np.where(counts[1:cap+1] < 2)[0]
    if len(bad) == 0:
        return cap
    return bad[0]  # 0-indexed in the slice, so position is bad[0]+1, but L = bad[0] means positions 1..bad[0] are good

def _score(residues, primes, cap=100000):
    return _compute_L(residues, primes, cap)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin
    
    primes = _primes_up_to(1000)
    n_primes = len(primes)
    
    # Initial: greedy set-cover approach
    # Process primes, for each choose residue that maximizes coverage of under-covered positions
    # Start with all residues as 0, then greedily improve
    
    residues = np.zeros(n_primes, dtype=np.int64)
    
    # Greedy: for each prime, try all residues and pick the one that gives best score
    # But that's too slow. Instead, use a smarter greedy.
    
    # Better: process primes in order, for each prime choose residue that covers
    # the most positions currently with coverage < 2 in range 1..some_limit
    
    # Let's use a different approach: 
    # 1. Start with a reasonable initial assignment
    # 2. Use local search to improve
    
    # Initial: assign each prime to cover a "spread" of positions
    # Use the idea: prime i covers position roughly (i+1) * some_step
    # But let's start with a simple heuristic and refine
    
    # Simple initial: residues[i] = (i * 7 + 3) % primes[i]  (arbitrary spread)
    residues = np.array([(i * 7 + 3) % primes[i] for i in range(n_primes)], dtype=np.int64)
    
    # Local search: iteratively change one residue to improve score
    best_score = _score(residues, primes)
    
    # Fast local search with numpy
    for iteration in range(50000):
        if time.time() > deadline:
            break
        
        # Pick a random prime to modify
        i = np.random.randint(n_primes)
        p = primes[i]
        old_r = residues[i]
        
        # Try a few candidate residues
        # Try: current, and some random ones, and ones that might help
        candidates = set()
        candidates.add(old_r)
        # Add a few random candidates
        for _ in range(5):
            candidates.add(np.random.randint(p))
        # Add candidate that targets the first gap
        # Find first position where coverage < 2
        # This is expensive, so just try random + current
        
        best_new_r = old_r
        best_new_score = best_score
        
        for new_r in candidates:
            if new_r == old_r:
                continue
            residues[i] = new_r
            s = _score(residues, primes)
            if s > best_new_score:
                best_new_score = s
                best_new_r = new_r
            residues[i] = old_r
        
        if best_new_r != old_r:
            residues[i] = best_new_r
            best_score = best_new_score
    
    return [int(r) for r in residues]
