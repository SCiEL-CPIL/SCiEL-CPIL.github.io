"""Narrow admissible tuple: fixed-diameter maximization in [0, N]."""
import time
import math
import random

N = 5000
DEADLINE = 315.0  # safety margin before 330s kill

def _primes(limit):
    """Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p::p] = bytearray(len(sieve[p * p::p]))
    return [p for p in range(2, limit + 1) if sieve[p]]

def construct():
    start_time = time.time()
    
    # Get primes up to N+1 (max possible tuple size is N+1)
    all_primes = _primes(N + 1)
    
    # Phase 1: Greedy removal from full set
    points = list(range(N + 1))
    
    while True:
        if time.time() - start_time > DEADLINE:
            break
        
        m = len(points)
        if m == 0:
            break
        
        primes_to_check = [p for p in all_primes if p <= m]
        
        changed = False
        for p in primes_to_check:
            counts = [0] * p
            for v in points:
                counts[v % p] += 1
            
            if min(counts) > 0:
                min_count = min(counts)
                min_residues = [r for r in range(p) if counts[r] == min_count]
                target_residue = min_residues[0]
                new_points = [v for v in points if v % p != target_residue]
                if len(new_points) < len(points):
                    points = new_points
                    changed = True
                    break
        
        if not changed:
            break
    
    # Phase 2: Fast add-back using maintained residue counts
    best_points = list(points)
    best_len = len(points)
    
    # Pre-compute residues for all candidates and all relevant primes
    # This is the key optimization: O(N * num_primes) precomputation
    # then O(1) per add check
    
    # For each prime p, we maintain:
    #   res_count[p][r] = number of elements in current set with value % p == r
    #   missing[p] = number of residues r where res_count[p][r] == 0
    # Adding v is safe iff for all p <= |S|+1, either p was already missing
    # at least one residue (so adding v doesn't complete it), OR v's residue
    # mod p is not the unique missing one.
    # More precisely: adding v violates p iff before adding, exactly one
    # residue was missing mod p AND v % p equals that missing residue.
    
    # We need to handle the case where |S| increases past a prime threshold.
    
    # Let's precompute for each value v in [0, N] and each prime p, v % p
    # Store as a list of dicts or arrays
    
    # Actually, let's just compute on the fly but maintain counts incrementally.
    
    def try_add_back(initial_points, candidates, time_limit):
        """Try to add candidates to initial_points using fast checks."""
        current = list(initial_points)
        current_set = set(current)
        m = len(current)
        
        # Build residue count structures
        # For each prime p <= N+1, maintain count of each residue
        # We only need primes up to N+1 since m can grow to N+1
        max_prime_idx = len(all_primes)
        
        # res_counts[p_idx] = list of length all_primes[p_idx]
        # missing_count[p_idx] = number of residues with count 0
        res_counts = {}
        missing_count = {}
        
        for pi, p in enumerate(all_primes):
            rc = [0] * p
            for v in current:
                rc[v % p] += 1
            mc = sum(1 for r in rc if r == 0)
            res_counts[pi] = rc
            missing_count[pi] = mc
        
        # Also track which primes are "active" (p <= m)
        # A prime p is active if p <= m
        
        # For adding candidate v:
        # 1. New m' = m + 1
        # 2. Check all active primes p <= m: adding v completes residue class
        #    iff missing_count[pi] == 1 and res_counts[pi][v % p] == 0
        # 3. Check if m' is prime: if so, that new prime is now active.
        #    For the new prime p' = m' (if prime), we need missing_count for p'.
        #    But we haven't computed it yet... we need to compute it.
        
        # Optimization: Only check primes that could be violated.
        # A prime p can only be violated by adding v if:
        #   - p <= m (was already active) AND missing_count[pi] == 1 AND res_counts[pi][v%p] == 0
        #   - OR p == m+1 and p is prime (newly active)
        
        # For the newly active prime case, we need to know if after adding v,
        # all residues mod p are covered. That requires computing counts for p.
        
        # Let's precompute which primes are "critical" (missing_count == 1)
        # and for each, what the missing residue is.
        
        critical = {}  # pi -> missing residue
        for pi, p in enumerate(all_primes):
            if missing_count[pi] == 1:
                rc = res_counts[pi]
                for r in range(p):
                    if rc[r] == 0:
                        critical[pi] = r
                        break
        
        # For each candidate, check if it's safe
        added = 0
        for v in candidates:
            if v in current_set:
                continue
            
            m_new = m + 1
            safe = True
            
            # Check active primes where missing_count == 1
            for pi, miss_r in critical.items():
                p = all_primes[pi]
                if p > m:
                    continue
                if v % p == miss_r:
                    safe = False
                    break
            
            # Check if m_new is prime
            if safe:
                # Check if m_new is prime
                is_prime = False
                for p in all_primes:
                    if p == m_new:
                        is_prime = True
                        break
                    if p > m_new:
                        break
                
                if is_prime:
                    # Need to check if adding v makes all residues covered mod m_new
                    # Compute residue counts for m_new
                    p = m_new
                    rc = [0] * p
                    for val in current:
                        rc[val % p] += 1
                    rc[v % p] += 1
                    if min(rc) > 0:
                        safe = False
            
            if safe:
                current.append(v)
                current_set.add(v)
                m = m_new
                added += 1
                
                # Update residue counts for all primes
                for pi, p in enumerate(all_primes):
                    old_count = res_counts[pi][v % p]
                    res_counts[pi][v % p] = old_count + 1
                    if old_count == 0:
                        missing_count[pi] -= 1
                    # Update critical
                    if pi in critical:
                        if missing_count[pi] != 1:
                            del critical[pi]
                    if missing_count[pi] == 1:
                        rc = res_counts[pi]
                        for r in range(p):
                            if rc[r] == 0:
                                critical[pi] = r
                                break
        
        return current
    
    # Get candidates
    point_set = set(best_points)
    candidates = [v for v in range(N + 1) if v not in point_set]
    
    # Try multiple random orders with the fast add-back
    num_trials = 200
    for trial in range(num_trials):
        if time.time() - start_time > DEADLINE - 15:
            break
        
        random.shuffle(candidates)
        
        current = try_add_back(best_points, candidates, DEADLINE - time.time())
        
        if len(current) > best_len:
            best_points = current
            best_len = len(current)
            point_set = set(best_points)
            candidates = [v for v in range(N + 1) if v not in point_set]
    
    return best_points
