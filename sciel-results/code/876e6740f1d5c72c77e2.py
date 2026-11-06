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

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit
    
    primes = _primes_up_to(1000)
    n_primes = len(primes)
    
    # Strategy: Greedy coverage extension
    # We want to maximize the prefix [1, L] where every integer has coverage >= 2.
    # 
    # Key insight: Each prime p covers a residue class r_p mod p.
    # We can choose r_p independently for each prime.
    # 
    # Greedy approach:
    # 1. Start with all residues = 0.
    # 2. Compute coverage counts for 1..cap.
    # 3. Find the first position u where coverage < 2.
    # 4. If u == 1, we need to increase coverage at position 1.
    #    For each prime p, check if changing r_p to 1 % p would increase coverage at 1.
    #    Actually, position 1 is covered by prime p iff 1 % p == r_p, i.e., r_p == 1 (for p > 1).
    #    So we need at least 2 primes with r_p == 1.
    # 5. More generally, at position u, we need coverage >= 2.
    #    Currently, some set of primes cover u. If |S_u| >= 2, we're good at u.
    #    If |S_u| < 2, we need to change some residues to cover u.
    #    For each prime p not currently covering u, we can set r_p = u % p.
    #    This will cover u, but may break coverage at other positions.
    #
    # Better approach: Local search with smart candidate generation.
    #
    # Let's use a different strategy:
    # Start with a heuristic initial assignment, then use simulated annealing or hill climbing.
    
    # Initial assignment: try to spread residues to cover a wide range
    # Use a simple heuristic: residues[i] = i % primes[i]
    residues = np.array([i % primes[i] for i in range(n_primes)], dtype=np.int64)
    
    best_score = _score(residues, primes)
    best_residues = residues.copy()
    
    # Hill climbing with restarts
    rng = np.random.RandomState(42)
    
    for restart in range(20):
        if time.time() > deadline - 10:
            break
        
        # Initialize: random or from best
        if restart == 0:
            current = best_residues.copy()
        else:
            current = best_residues.copy()
            # Perturb: change 10% of residues randomly
            n_change = max(1, n_primes // 10)
            idx = rng.choice(n_primes, n_change, replace=False)
            for i in idx:
                current[i] = rng.randint(primes[i])
        
        current_score = _score(current, primes)
        
        # Local search
        no_improve_count = 0
        max_no_improve = 500
        
        while no_improve_count < max_no_improve:
            if time.time() > deadline - 5:
                break
            
            # Pick a random prime to modify
            i = rng.randint(n_primes)
            p = primes[i]
            old_r = current[i]
            
            # Generate candidates:
            # 1. Current residue
            # 2. A few random residues
            # 3. Residues that target the first gap
            
            candidates = set()
            candidates.add(old_r)
            
            # Add random candidates
            for _ in range(3):
                candidates.add(rng.randint(p))
            
            # Find first gap to target
            # This is expensive, so do it less frequently
            if no_improve_count % 10 == 0:
                # Find first position with coverage < 2
                counts = np.zeros(10000, dtype=np.int32)
                for j, pj in enumerate(primes):
                    rj = current[j]
                    start = rj
                    if start < 1:
                        start = rj + pj
                    if start > 10000:
                        continue
                    counts[start::pj] += 1
                bad = np.where(counts[1:10001] < 2)[0]
                if len(bad) > 0:
                    u = bad[0] + 1  # first uncovered position
                    # For the prime we're modifying, the residue that covers u is u % p
                    candidates.add(u % p)
                    # Also try residues that cover u-1, u+1, etc.
                    for offset in [-2, -1, 1, 2]:
                        if 1 <= u + offset <= 10000:
                            candidates.add((u + offset) % p)
            
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
                no_improve_count = 0
                
                if current_score > best_score:
                    best_score = current_score
                    best_residues = current.copy()
            else:
                no_improve_count += 1
        
        # Update best if improved
        if current_score > best_score:
            best_score = current_score
            best_residues = current.copy()
    
    return [int(r) for r in best_residues]
