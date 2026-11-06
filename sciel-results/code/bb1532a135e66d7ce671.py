"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _get_primes():
    """Return list of primes up to 1000."""
    primes = []
    for p in range(2, 1001):
        is_prime = True
        for q in range(2, int(p ** 0.5) + 1):
            if p % q == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(p)
    return primes

def _compute_L(residues, primes, max_check=100000):
    """Compute the largest L such that every integer 1..L has coverage >= 2."""
    n = len(primes)
    # For efficiency, we can check incrementally
    # But for simplicity and correctness, let's use a direct approach
    # Coverage of x: count of i where x % primes[i] == residues[i]
    
    # We'll scan from 1 upward and track the minimum coverage
    # Actually, we need the largest L where ALL of 1..L have coverage >= 2
    # So we scan until we find an x with coverage < 2
    
    # Optimization: precompute for each prime p, the residues
    # For a given x, coverage = sum(1 for i in range(n) if x % primes[i] == residues[i])
    
    # To speed this up, we can use the fact that for large primes, 
    # the condition x % p == r is satisfied only for specific x in [1, L]
    # But let's just do it directly with some optimization
    
    # Alternative: for each prime i, the numbers covered in [1, L] are 
    # residues[i], residues[i]+p_i, residues[i]+2*p_i, ...
    # We can use a difference array or just count
    
    # For L up to 10^5 and 168 primes, direct computation per x is 168*10^5 = 1.68M ops
    # That's fast enough in Python if we're careful
    
    for L in range(1, max_check + 1):
        # Check if L has coverage >= 2
        count = 0
        for i in range(n):
            if L % primes[i] == residues[i]:
                count += 1
                if count >= 2:
                    break
        if count < 2:
            return L - 1
    return max_check

def _compute_L_fast(residues, primes, max_check=100000):
    """Faster computation using a coverage array."""
    n = len(primes)
    # Create coverage array for 1..max_check
    # For each prime i with residue r_i, mark positions r_i, r_i+p_i, r_i+2*p_i, ...
    
    # We only need to check up to where we find the first gap
    # Let's build coverage incrementally
    
    # For each prime, add 1 to positions that satisfy the congruence
    # Use a list for coverage counts
    # max_check can be up to 100000, which is fine
    
    cov = [0] * (max_check + 1)
    for i in range(n):
        p = primes[i]
        r = residues[i]
        # Positions: r, r+p, r+2p, ... up to max_check
        # Note: x % p == r means x = r + k*p for k >= 0
        # But x must be >= 1, so if r == 0, start at p (since 0 is not in 1..L)
        # Actually x % p == 0 means x is a multiple of p
        # For r > 0: positions are r, r+p, r+2p, ...
        # For r == 0: positions are p, 2p, 3p, ...
        
        if r == 0:
            start = p
        else:
            start = r
        
        for x in range(start, max_check + 1, p):
            cov[x] += 1
    
    # Now find the largest L such that cov[x] >= 2 for all x in 1..L
    for L in range(1, max_check + 1):
        if cov[L] < 2:
            return L - 1
    return max_check

def construct():
    deadline = time.time() + 145.0
    
    primes = _get_primes()
    n_primes = len(primes)
    
    # Initial solution: the seed
    residues = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    best_L = _compute_L_fast(residues, primes)
    best_residues = residues[:]
    
    # Hill climbing: try to improve by changing one residue at a time
    # Focus on fixing the first gap
    
    random.seed(42)
    
    # We'll do multiple passes
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Find the current gap
        current_L = _compute_L_fast(residues, primes)
        
        # The first integer not doubly covered is current_L + 1
        # We want to increase coverage at current_L + 1
        gap_x = current_L + 1
        if gap_x > 100000:
            break
        
        # Try changing each residue to fix the gap
        # For each prime i, what residue would cover gap_x?
        # We need gap_x % primes[i] to be the new residue
        # But we also need to not break coverage of 1..current_L
        
        # Strategy: for each prime, try setting residue to gap_x % p_i
        # Check if this improves L without breaking earlier coverage
        
        candidates = []
        for i in range(n_primes):
            p = primes[i]
            new_r = gap_x % p
            if new_r == residues[i]:
                continue
            # Try this change
            old_r = residues[i]
            residues[i] = new_r
            new_L = _compute_L_fast(residues, primes)
            residues[i] = old_r
            if new_L > current_L:
                candidates.append((new_L, i, new_r))
        
        # Pick the best candidate
        if candidates:
            candidates.sort(reverse=True)
            best_new_L, best_i, best_r = candidates[0]
            residues[best_i] = best_r
            if best_new_L > best_L:
                best_L = best_new_L
                best_residues = residues[:]
            improved = True
        
        # Also try random perturbations
        if time.time() < deadline - 10:
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                # Pick a random prime and random residue
                i = random.randint(0, n_primes - 1)
                old_r = residues[i]
                new_r = random.randint(0, primes[i] - 1)
                residues[i] = new_r
                new_L = _compute_L_fast(residues, primes)
                if new_L > best_L:
                    best_L = new_L
                    best_residues = residues[:]
                    improved = True
                else:
                    residues[i] = old_r
    
    return best_residues
# EVOLVE-BLOCK-END
