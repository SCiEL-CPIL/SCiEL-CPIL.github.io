"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def _primes_up_to(n):
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    primes = _primes_up_to(1000)
    n = len(primes)  # 168
    p_arr = primes  # list of primes

    # Score function: compute L such that every integer 1..L has >= 2 congruences
    def compute_score(residues):
        """Compute the largest L such that every integer 1..L satisfies >= 2 congruences."""
        # We need to check coverage. Use a counting approach for the range we care about.
        # First, find a reasonable upper bound. The seed gives L=84.
        # Let's binary search or just extend until we find a gap.
        # For efficiency, check in chunks.
        
        # Upper bound heuristic: with 168 primes, average coverage per integer in [1, X]
        # is sum(1/p) * X. For X=1000, sum(1/p) for p<=1000 ≈ 2.2, so avg ≈ 2200/1000 ≈ 2.2.
        # But we need >= 2 everywhere. Let's just scan.
        
        # Efficient approach: for each prime p with residue r, the covered positions in [1, MAX]
        # are r, r+p, r+2p, ... (adjusting for 1-indexed: if r==0, positions are p, 2p, ...)
        # Actually x = r mod p means x ≡ r (mod p). If r=0, x is multiple of p.
        
        MAX_CHECK = 20000  # check up to this to find the first gap
        if time.time() > deadline:
            return 0
            
        # Count coverage for positions 1..MAX_CHECK
        # Use a difference array or direct counting
        counts = [0] * (MAX_CHECK + 1)
        
        for p, r in zip(primes, residues):
            # Positions x in [1, MAX_CHECK] where x ≡ r (mod p)
            # x = r + k*p, but we need x >= 1
            # If r == 0: x = p, 2p, 3p, ...
            # If r > 0: x = r, r+p, r+2p, ... (but if r > MAX_CHECK, skip)
            # Actually x ≡ r (mod p) means x = r + k*p for integer k, with x >= 1
            # The smallest x >= 1: if r >= 1, x = r. If r == 0, x = p.
            
            if r == 0:
                first = p
            else:
                first = r
            if first > MAX_CHECK:
                continue
            for x in range(first, MAX_CHECK + 1, p):
                counts[x] += 1
        
        # Find the largest L such that counts[x] >= 2 for all x in 1..L
        L = 0
        for x in range(1, MAX_CHECK + 1):
            if counts[x] >= 2:
                L = x
            else:
                break
        return L

    def compute_score_fast(residues, max_check=5000):
        """Faster score computation for local search."""
        counts = [0] * (max_check + 1)
        for p, r in zip(primes, residues):
            if r == 0:
                first = p
            else:
                first = r
            if first > max_check:
                continue
            for x in range(first, max_check + 1, p):
                counts[x] += 1
        L = 0
        for x in range(1, max_check + 1):
            if counts[x] >= 2:
                L = x
            else:
                break
        return L

    # Initial solution: the seed
    residues = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    best_score = compute_score(residues)
    best_residues = list(residues)
    
    # Local search: try swapping residues of pairs of primes
    # For each prime i, try changing its residue to see if score improves
    random.seed(42)
    
    max_check = 5000
    current_score = best_score
    current_residues = list(best_residues)
    
    # Greedy first-bad-point approach:
    # Find the first position where coverage < 2, then try to fix it
    # by adjusting residues of primes that could cover that position
    
    # Precompute for each prime which positions it covers in [1, max_check]
    # This is expensive, so let's do local search with incremental updates
    
    # Simpler local search: for each prime i, try all residues (or a subset) and see if score improves
    # But 168 primes * up to 1000 residues each is too many.
    
    # Better: for each prime i, only try residues that are "near" the current one or that
    # specifically help cover the first gap.
    
    # Let's do a hill-climbing approach:
    # 1. Find first gap position g (where count < 2)
    # 2. For primes that could cover g (i.e., where changing residue to g % p would help),
    #    try that specific change
    # 3. Accept if score improves
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Find first gap
        counts = [0] * (max_check + 1)
        for p, r in zip(primes, current_residues):
            if r == 0:
                first = p
            else:
                first = r
            if first <= max_check:
                for x in range(first, max_check + 1, p):
                    counts[x] += 1
        
        gap_pos = -1
        for x in range(1, max_check + 1):
            if counts[x] < 2:
                gap_pos = x
                break
        
        if gap_pos == -1:
            # No gap found in [1, max_check], try to extend
            # For now, accept current
            break
        
        # Try to fix gap_pos by changing residues of primes
        # A prime p can cover gap_pos if we set r = gap_pos % p
        # Try each prime in random order
        indices = list(range(n))
        random.shuffle(indices)
        
        for i in indices:
            p = primes[i]
            new_r = gap_pos % p
            if new_r == current_residues[i]:
                continue
            
            old_r = current_residues[i]
            current_residues[i] = new_r
            
            # Recompute score (expensive but necessary for correctness)
            new_score = compute_score_fast(current_residues, max_check)
            
            if new_score > current_score:
                current_score = new_score
                best_score = max(best_score, current_score)
                best_residues = list(current_residues)
                improved = True
                # Don't break; keep trying other primes for this gap
            else:
                current_residues[i] = old_r
        
        # Also try random perturbations
        if not improved and time.time() < deadline:
            # Random swap: pick a random prime and change its residue randomly
            i = random.randint(0, n - 1)
            p = primes[i]
            old_r = current_residues[i]
            new_r = random.randint(0, p - 1)
            if new_r != old_r:
                current_residues[i] = new_r
                new_score = compute_score_fast(current_residues, max_check)
                if new_score > current_score:
                    current_score = new_score
                    best_score = max(best_score, current_score)
                    best_residues = list(current_residues)
                    improved = True
                else:
                    current_residues[i] = old_r
    
    # Final score computation with larger max_check to get accurate score
    if time.time() < deadline - 5:
        final_score = compute_score(best_residues)
        if final_score > best_score:
            best_score = final_score
    
    return best_residues
# EVOLVE-BLOCK-END
