<<<<<<< SEARCH
# EVOLVE-BLOCK-START
def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def _is_admissible(points, all_primes):
    """Check if a set of points is admissible."""
    n = len(points)
    for p in all_primes:
        if p > n:
            break
        residues = set(x % p for x in points)
        if len(residues) == p:
            return False
    return True

def construct():
    start_time = time.time()
    
    # 1. Precompute primes
    all_primes = _primes(N + 1)
    prime_set = set(all_primes)
    
    # 2. Initialize sparse set
    # Start with a small set of points that is likely admissible.
    # We start with {0} and add primes > sqrt(N) to create a "spread out" set.
    sqrt_N = int(N ** 0.5) + 1
    points = {0}
    for p in all_primes:
        if p > sqrt_N and p <= N:
            points.add(p)
    
    # Ensure we are within [0, N]
    points = {x for x in points if 0 <= x <= N}
    
    # 3. Helper to update residue counts
    # We maintain counts for all primes p <= N
    # residue_counts[p][r] = count of x in points such that x % p == r
    # missing_count[p] = number of r in [0, p-1] such that residue_counts[p][r] == 0
    
    # Initialize structures
    residue_counts = {}
    missing_counts = {}
    
    for p in all_primes:
        counts = [0] * p
        residue_counts[p] = counts
        missing_counts[p] = p # initially all missing
        
    # Populate initial counts
    for x in points:
        for p in all_primes:
            r = x % p
            if residue_counts[p][r] == 0:
                missing_counts[p] -= 1
            residue_counts[p][r] += 1
            
    # 4. Repair Phase: Ensure admissibility
    # Admissible if for all primes p <= len(points), missing_counts[p] > 0
    # Note: If missing_counts[p] == 0, we must remove elements.
    
    changed = True
    while changed:
        changed = False
        if time.time() - start_time > DEADLINE - 10:
            break
            
        n = len(points)
        # Check primes p <= n
        for p in all_primes:
            if p > n:
                break
            
            if missing_counts[p] == 0:
                # Need to remove elements to create a missing residue
                # Find the residue class with the fewest elements
                min_count = min(residue_counts[p])
                target_residues = [r for r, c in enumerate(residue_counts[p]) if c == min_count]
                # Pick one target residue
                target = target_residues[0]
                
                # Remove all points x such that x % p == target
                to_remove = [x for x in points if x % p == target]
                
                if not to_remove:
                    continue
                    
                for x in to_remove:
                    points.remove(x)
                    # Update counts
                    for pp in all_primes:
                        r = x % pp
                        residue_counts[pp][r] -= 1
                        if residue_counts[pp][r] == 0:
                            missing_counts[pp] += 1
                            
                changed = True
                break # Restart check as points changed significantly
                
    # 5. Optimization Phase: Add back elements
    # Candidates are all x in [0, N] not in points
    candidates = [x for x in range(N + 1) if x not in points]
    random.shuffle(candidates) # Shuffle to avoid bias towards smaller numbers
    
    added_any = True
    while added_any:
        added_any = False
        if time.time() - start_time > DEADLINE - 10:
            break
        
        n = len(points)
        
        # We need to check if adding x is valid.
        # Valid if:
        # 1. For all primes p <= n, if x % p was missing, it was NOT the ONLY missing one.
        #    (i.e., missing_counts[p] > 1 if residue_counts[p][x%p] == 0)
        # 2. If n+1 is prime, adding x does not complete the residue system mod n+1.
        
        new_n = n + 1
        new_prime = None
        if new_n in prime_set:
            new_prime = new_n
            
        for x in candidates:
            if x in points:
                continue
                
            valid = True
            
            # Check 1: Existing primes p <= n
            for p in all_primes:
                if p > n:
                    break
                r = x % p
                if residue_counts[p][r] == 0:
                    # x fills a missing spot
                    if missing_counts[p] == 1:
                        # This was the only missing spot, so adding x makes it complete
                        valid = False
                        break
                        
            if not valid:
                continue
                
            # Check 2: New prime bound
            if new_prime is not None:
                p = new_prime
                # Check if points U {x} covers all residues mod p
                # Since len(points U {x}) == p, this is true iff all residues are distinct.
                # We can check if there is any residue missing in the new set.
                # Current missing_counts[p] tells us how many are missing in 'points'.
                # Let r_x = x % p.
                # If r_x was missing (residue_counts[p][r_x] == 0), then missing_count decreases by 1.
                # If r_x was present, missing_count stays same.
                # We need the final missing count to be > 0.
                
                r_x = x % p
                if residue_counts[p][r_x] == 0:
                    final_missing = missing_counts[p] - 1
                else:
                    final_missing = missing_counts[p]
                    
                if final_missing == 0:
                    valid = False
                    
            if valid:
                # Add x
                points.add(x)
                
                # Update counts
                for pp in all_primes:
                    r = x % pp
                    if residue_counts[pp][r] == 0:
                        missing_counts[pp] -= 1
                    residue_counts[pp][r] += 1
                    
                added_any = True
                # Restart candidate loop to pick up new opportunities?
                # Or just continue?
                # Continuing is faster, but might miss combos.
                # Given the local nature, continuing is usually fine.
                # However, adding x changes n, so new_prime changes.
                # We should probably restart the candidate scan or at least re-evaluate the prime bound.
                # For simplicity and speed, we break and restart the while loop.
                break
                
    return list(points)
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def _is_admissible(points, all_primes):
    """Check if a set of points is admissible."""
    n = len(points)
    for p in all_primes:
        if p > n:
            break
        residues = set(x % p for x in points)
        if len(residues) == p:
            return False
    return True

def _greedy_construction(all_primes, prime_set, rng, max_iter=50000):
    """Run one greedy construction attempt with randomization."""
    start_time = time.time()
    
    # Initialize with a random subset to break symmetry
    # Start with a small random set
    initial_size = rng.randint(50, 200)
    initial_points = set(rng.sample(range(N + 1), initial_size))
    
    # Build residue count structures
    residue_counts = {}
    missing_counts = {}
    
    for p in all_primes:
        counts = [0] * p
        residue_counts[p] = counts
        missing_counts[p] = p
        
    for x in initial_points:
        for p in all_primes:
            r = x % p
            if residue_counts[p][r] == 0:
                missing_counts[p] -= 1
            residue_counts[p][r] += 1
    
    points = initial_points
    
    # Repair phase: ensure admissibility
    changed = True
    while changed:
        changed = False
        if time.time() - start_time > DEADLINE - 15:
            break
            
        n = len(points)
        for p in all_primes:
            if p > n:
                break
            
            if missing_counts[p] == 0:
                # Remove all points in the least-populated residue class
                min_count = min(residue_counts[p])
                target_residues = [r for r, c in enumerate(residue_counts[p]) if c == min_count]
                target = target_residues[rng.randint(0, len(target_residues) - 1)]
                
                to_remove = [x for x in points if x % p == target]
                if not to_remove:
                    continue
                    
                for x in to_remove:
                    points.remove(x)
                    for pp in all_primes:
                        r = x % pp
                        residue_counts[pp][r] -= 1
                        if residue_counts[pp][r] == 0:
                            missing_counts[pp] += 1
                            
                changed = True
                break
                
    # Optimization phase: add elements greedily
    # Precompute candidate list once, shuffle for diversity
    candidates = [x for x in range(N + 1) if x not in points]
    rng.shuffle(candidates)
    
    added_any = True
    iter_count = 0
    while added_any and iter_count < max_iter:
        added_any = False
        iter_count += 1
        if time.time() - start_time > DEADLINE - 15:
            break
        
        n = len(points)
        new_n = n + 1
        new_prime = new_n if new_n in prime_set else None
        
        for x in candidates:
            if x in points:
                continue
                
            valid = True
            
            # Check existing primes
            for p in all_primes:
                if p > n:
                    break
                r = x % p
                if residue_counts[p][r] == 0 and missing_counts[p] == 1:
                    valid = False
                    break
                    
            if not valid:
                continue
                
            # Check new prime bound
            if new_prime is not None:
                p = new_prime
                r_x = x % p
                if residue_counts[p][r_x] == 0:
                    final_missing = missing_counts[p] - 1
                else:
                    final_missing = missing_counts[p]
                if final_missing == 0:
                    valid = False
                    
            if valid:
                points.add(x)
                for pp in all_primes:
                    r = x % pp
                    if residue_counts[pp][r] == 0:
                        missing_counts[pp] -= 1
                    residue_counts[pp][r] += 1
                added_any = True
                break
                
    return list(points)

def construct():
    start_time = time.time()
    
    # Precompute primes
    all_primes = _primes(N + 1)
    prime_set = set(all_primes)
    
    # Multiple randomized restarts
    best_result = []
    rng = random.Random(42)
    
    while time.time() - start_time < DEADLINE - 20:
        result = _greedy_construction(all_primes, prime_set, rng)
        
        if len(result) > len(best_result):
            best_result = result
            
        # If we've found a very large set, we can stop early
        if len(best_result) > 4800:
            break
            
    # Final validation
    if best_result:
        for p in all_primes:
            if p > len(best_result):
                break
            residues = set(x % p for x in best_result)
            if len(residues) == p:
                # Should not happen, but return empty if invalid
                return []
                
    return best_result
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
