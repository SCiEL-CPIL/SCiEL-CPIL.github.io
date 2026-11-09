"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Generate primes <= 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(math.isqrt(limit)) + 1):
            if sieve[i]:
                for j in range(i * i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]

    primes = get_primes(1000)
    n = len(primes)  # 168

    # Precompute for fast evaluation: for each prime index, precompute residues
    # We'll use a coverage array approach

    def evaluate(residues, max_check=100000):
        """Return the largest L such that every integer in 1..L is covered by at least 2 congruences."""
        # Count coverage for each position 1..max_check
        # Use an array of counts
        counts = [0] * (max_check + 1)
        
        for i, p in enumerate(primes):
            r = residues[i]
            # Positions: r, r+p, r+2p, ... but we only care about 1..max_check
            # If r == 0, positions are p, 2p, ... (since 0 is not in 1..max_check, but r=0 means x ≡ 0 mod p)
            # Actually if r=0, the first positive is p. range(0, max_check+1, p) includes 0, but we ignore index 0.
            # Better: start at r if r >= 1, else start at p (i.e., 0 + p)
            # range(r, max_check+1, p) works if r >= 1. If r == 0, range(0, ...) includes 0.
            # Let's just use range(r, max_check+1, p) and skip index 0 in the sum check.
            for pos in range(r, max_check + 1, p):
                if pos > 0:
                    counts[pos] += 1
        
        # Find largest L where all 1..L have count >= 2
        L = 0
        for x in range(1, max_check + 1):
            if counts[x] >= 2:
                L = x
            else:
                break
        return L

    # Greedy initialization
    residues = [0] * n
    current_L = 0
    
    # Phase 1: Greedy single-pass initialization
    for i in range(n):
        best_r = 0
        best_L = -1
        p = primes[i]
        for r in range(p):
            residues[i] = r
            L = evaluate(residues)
            if L > best_L:
                best_L = L
                best_r = r
            if time.time() > deadline:
                residues[i] = best_r
                return residues
        residues[i] = best_r
        current_L = best_L

    if time.time() > deadline:
        return residues

    # Phase 2: Local search with kicks
    best_residues = residues[:]
    best_L = current_L
    
    random.seed(42)
    stuck_count = 0
    
    while time.time() < deadline:
        # Pick a random index to perturb
        idx = random.randrange(n)
        old_r = residues[idx]
        p = primes[idx]
        
        # Try a random new residue
        new_r = random.randrange(p)
        if new_r == old_r:
            new_r = (new_r + 1) % p
        
        residues[idx] = new_r
        L = evaluate(residues)
        
        if L > best_L:
            best_L = L
            best_residues = residues[:]
            stuck_count = 0
        else:
            # Revert
            residues[idx] = old_r
            stuck_count += 1
        
        # Occasionally kick
        if stuck_count > 2000:
            # Change 5 random residues
            for _ in range(5):
                k = random.randrange(n)
                residues[k] = random.randrange(primes[k])
            L = evaluate(residues)
            if L > best_L:
                best_L = L
                best_residues = residues[:]
            stuck_count = 0
        
        # If time is running low, stop
        if time.time() > deadline - 5:
            break

    return best_residues

# EVOLVE-BLOCK-END
