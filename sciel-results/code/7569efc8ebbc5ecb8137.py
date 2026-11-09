"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 130  # safety margin before 150s kill
    
    # Generate primes up to 1000
    primes = []
    for p in range(2, 1001):
        is_prime = True
        for q in range(2, int(p**0.5) + 1):
            if p % q == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(p)
    
    n_primes = len(primes)
    assert n_primes == 168, f"Expected 168 primes, got {n_primes}"
    
    # Precompute for fast scoring
    # We need to check: for each x in 1..L, count how many primes p have x % p == r_p
    # Score = largest L such that min_{x in 1..L} count(x) >= 2
    
    def score_residues(r):
        """Compute the score (largest L with double coverage) for given residues."""
        # Use numpy for speed
        # For each x from 1 to some limit, count matches
        # We'll compute incrementally
        
        # First, let's compute counts for x = 1..100000 (capped)
        # But we only need up to the point where count drops below 2
        
        # Efficient approach: for each prime p with residue r_p,
        # it covers x where x % p == r_p
        # We want the minimum count over x=1..L to be >= 2
        
        # Let's compute the count array for x = 1..max_L
        # max_L is at most 100000, but we can stop early
        
        max_check = 100000
        counts = np.zeros(max_check + 1, dtype=np.int32)
        
        for i, p in enumerate(primes):
            rp = r[i]
            # x % p == rp means x = rp + k*p for k >= 0
            # x ranges from 1 to max_check
            # So rp + k*p >= 1 and rp + k*p <= max_check
            start = rp
            if start < 1:
                start = 1 + p - (1 - rp) % p  # next x >= 1 with x % p == rp
                # Actually: if rp == 0, x must be multiple of p, so x = p, 2p, ...
                # if rp > 0, x = rp, rp+p, ...
                # Let's just iterate properly
            # x values: rp, rp+p, rp+2p, ... but must be >= 1
            # If rp == 0: x = p, 2p, 3p, ...
            # If rp > 0: x = rp, rp+p, rp+2p, ...
            
            if rp == 0:
                vals = np.arange(p, max_check + 1, p)
            else:
                vals = np.arange(rp, max_check + 1, p)
            
            if len(vals) > 0:
                counts[vals] += 1
        
        # Find largest L such that min(counts[1..L]) >= 2
        # i.e., find the first x where counts[x] < 2, then L = x - 1
        # If all counts[1..max_check] >= 2, then L = max_check (capped)
        
        # Find first index where count < 2
        below2 = np.where(counts[1:] < 2)[0]
        if len(below2) == 0:
            return max_check
        else:
            return below2[0]  # This is the 0-indexed position in counts[1:], so actual x = below2[0] + 1, and L = x - 1 = below2[0]
    
    # Simulated Annealing approach
    # Start with a reasonable initial guess
    # The seed uses r_p = ((i+2)//2) % p
    
    best_r = np.array([((i + 2) // 2) % p for i, p in enumerate(primes)], dtype=np.int64)
    best_score = score_residues(best_r)
    
    # Try some simple heuristics first
    # Heuristic 1: r_p = 0 for all p
    r_test = np.zeros(n_primes, dtype=np.int64)
    s_test = score_residues(r_test)
    if s_test > best_score:
        best_score = s_test
        best_r = r_test.copy()
    
    # Heuristic 2: r_p = 1 for all p
    r_test = np.ones(n_primes, dtype=np.int64)
    s_test = score_residues(r_test)
    if s_test > best_score:
        best_score = s_test
        best_r = r_test.copy()
    
    # Heuristic 3: random
    np.random.seed(42)
    for _ in range(20):
        r_test = np.random.randint(0, 1, n_primes) * np.array(primes, dtype=np.int64)
        r_test = np.random.randint(0, np.array(primes, dtype=np.int64))
        s_test = score_residues(r_test)
        if s_test > best_score:
            best_score = s_test
            best_r = r_test.copy()
    
    # Simulated Annealing
    current_r = best_r.copy()
    current_score = best_score
    
    # Temperature schedule
    T0 = 10.0
    T1 = 0.01
    alpha = 0.9995  # cooling rate
    
    T = T0
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        if T < T1:
            break
        
        # Pick a random prime index to change
        idx = np.random.randint(0, n_primes)
        old_val = current_r[idx]
        new_val = np.random.randint(0, primes[idx])
        
        if new_val == old_val:
            continue
        
        current_r[idx] = new_val
        new_score = score_residues(current_r)
        
        delta = new_score - current_score
        
        if delta > 0:
            # Accept
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_r = current_r.copy()
                no_improve = 0
            else:
                no_improve += 1
        else:
            # Reject with probability exp(-delta/T)
            if delta < 0:
                prob = np.exp(delta / T)
                if np.random.random() < prob:
                    current_score = new_score
                else:
                    current_r[idx] = old_val  # revert
            no_improve += 1
        
        T *= alpha
        
        if no_improve > max_no_improve and T > T1:
            # Restart with higher temperature
            T = max(T * 0.5, T1)
            no_improve = 0
    
    # Local improvement pass
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(n_primes):
            old_val = current_r[idx]
            best_local_val = old_val
            best_local_score = current_score
            
            # Try all possible values for this residue
            for v in range(primes[idx]):
                if v == old_val:
                    continue
                current_r[idx] = v
                s = score_residues(current_r)
                if s > best_local_score:
                    best_local_score = s
                    best_local_val = v
                current_r[idx] = old_val
            
            if best_local_val != old_val:
                current_r[idx] = best_local_val
                current_score = best_local_score
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_r = current_r.copy()
    
    # Final check
    final_score = score_residues(best_r)
    if final_score > best_score:
        best_score = final_score
    
    return [int(x) for x in best_r]

# EVOLVE-BLOCK-END
