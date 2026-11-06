import numpy as np
import time

def _primes_up_to(n):
    """Return list of primes up to n."""
    if n < 2:
        return []
    sieve = np.ones(n + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    return [int(p) for p in np.where(sieve)[0]]

def _compute_L(residues, primes, cap=100000):
    """Compute the largest L such that every integer 1..L has coverage >= 2."""
    counts = np.zeros(cap + 1, dtype=np.int32)
    for i, p in enumerate(primes):
        r = residues[i]
        start = r
        if start < 1:
            start = r + p
        if start > cap:
            continue
        counts[start::p] += 1
    # Find first position where count < 2
    bad = np.where(counts[1:cap+1] < 2)[0]
    if len(bad) == 0:
        return cap
    return bad[0]

def _score(residues, primes, cap=100000):
    return _compute_L(residues, primes, cap)

def _greedy_init(primes):
    """
    Greedy initialization: for each prime, choose residue to maximize coverage
    of the early range. Use a simple heuristic: try to cover positions 1, 2, 3, ...
    with at least 2 primes.
    """
    n = len(primes)
    residues = np.zeros(n, dtype=np.int64)
    
    # For each position u from 1 upward, ensure it's covered by at least 2 primes.
    # Process primes in order of size (larger first for more flexibility? or smaller first?)
    # Let's try: for each prime p, pick r_p to cover as many of the first M positions as possible.
    # A simpler approach: for each prime, pick r_p = u % p where u is a target.
    
    # Actually, let's use a different greedy:
    # Start with all residues = 0.
    # Then iteratively fix the first gap.
    
    residues[:] = 0
    cap = 10000
    counts = np.zeros(cap + 1, dtype=np.int32)
    for i, p in enumerate(primes):
        r = residues[i]
        start = r
        if start < 1:
            start = r + p
        if start > cap:
            continue
        counts[start::p] += 1
    
    # Now find gaps and fix them
    for u in range(1, cap + 1):
        if counts[u] < 2:
            # Need to add coverage at u
            # Find a prime p where changing r_p to u % p would help most
            best_i = -1
            best_gain = 0
            for i, p in enumerate(primes):
                old_r = residues[i]
                if old_r == u % p:
                    continue  # already covers u
                # If we change to u % p, what's the impact?
                # We gain coverage at u, but lose it at old_r + kp
                # This is complex; for simplicity, just pick the smallest prime not covering u
                # Actually, let's pick the prime with smallest p (most frequent)
                if best_i == -1 or p < primes[best_i]:
                    best_i = i
            
            if best_i >= 0:
                p = primes[best_i]
                old_r = residues[best_i]
                new_r = u % p
                # Update counts
                # Remove old coverage
                start_old = old_r
                if start_old < 1:
                    start_old = old_r + p
                if start_old <= cap:
                    counts[start_old::p] -= 1
                # Add new coverage
                start_new = new_r
                if start_new < 1:
                    start_new = new_r + p
                if start_new <= cap:
                    counts[start_new::p] += 1
                residues[best_i] = new_r
    
    return residues

def _local_search(residues, primes, deadline, rng):
    """
    Local search to improve the score.
    """
    n = len(primes)
    current = residues.copy()
    current_score = _score(current, primes)
    best_score = current_score
    best_residues = current.copy()
    
    no_improve = 0
    max_no_improve = 300
    
    while no_improve < max_no_improve:
        if time.time() > deadline:
            break
        
        # Pick a random prime to modify
        i = rng.randint(n)
        p = primes[i]
        old_r = current[i]
        
        # Generate candidates
        candidates = set()
        candidates.add(old_r)
        
        # Find the first gap
        cap_check = min(10000, _score(current, primes) + 200)
        counts = np.zeros(cap_check + 1, dtype=np.int32)
        for j, pj in enumerate(primes):
            rj = current[j]
            start = rj
            if start < 1:
                start = rj + pj
            if start > cap_check:
                continue
            counts[start::pj] += 1
        bad = np.where(counts[1:cap_check+1] < 2)[0]
        
        if len(bad) > 0:
            u = bad[0] + 1
            candidates.add(u % p)
            for offset in [-3, -2, -1, 1, 2, 3]:
                if 1 <= u + offset <= cap_check:
                    candidates.add((u + offset) % p)
        
        # Add some random candidates
        for _ in range(5):
            candidates.add(rng.randint(p))
        
        best_new_r = old_r
        best_new_score = current_score
        
        for new_r in candidates:
            if new_r == old_r:
                continue
            current[i] = new_r
            s = _score(current, primes)
            if s > best_new_score:
                best_new_score = s
                best_new_r = new_r
            current[i] = old_r
        
        if best_new_r != old_r:
            current[i] = best_new_r
            current_score = best_new_score
            no_improve = 0
            
            if current_score > best_score:
                best_score = current_score
                best_residues = current.copy()
        else:
            no_improve += 1
    
    return best_residues, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    primes = _primes_up_to(1000)
    n_primes = len(primes)
    
    # Greedy initialization
    residues = _greedy_init(primes)
    best_score = _score(residues, primes)
    best_residues = residues.copy()
    
    # Local search with multiple restarts
    rng = np.random.RandomState(42)
    
    for restart in range(30):
        if time.time() > deadline - 15:
            break
        
        if restart == 0:
            current = best_residues.copy()
        else:
            current = best_residues.copy()
            # Perturb 15% of residues
            n_change = max(1, n_primes // 7)
            idx = rng.choice(n_primes, n_change, replace=False)
            for i in idx:
                current[i] = rng.randint(primes[i])
        
        improved_res, improved_score = _local_search(current, primes, deadline - 10, rng)
        
        if improved_score > best_score:
            best_score = improved_score
            best_residues = improved_res
    
    return [int(r) for r in best_residues]
