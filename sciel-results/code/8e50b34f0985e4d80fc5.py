"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    # Deadline guard: evaluator kills after 150s, return best found with safety margin
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    # Generate primes up to 1000
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    n_primes = len(primes)  # 168

    # Helper: compute L (largest prefix where every integer 1..L is covered at least twice)
    def compute_L(residues, max_check=100000):
        # For each integer x from 1 to max_check, count how many primes p satisfy x ≡ r_p (mod p)
        # We need the largest L such that for all x in 1..L, count >= 2
        # Efficient approach: for each prime p with residue r, the covered integers are r, r+p, r+2p, ...
        # Build a coverage count array
        # To be efficient, use a difference array or just count
        # For L up to ~100000, we can use an array of size max_check+1
        # But we only need to find the first gap, so we can stop early
        
        # Use numpy for speed
        import numpy as np
        counts = np.zeros(max_check + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            # Covered: r, r+p, r+2p, ... <= max_check
            # Start from the first positive covered integer
            if r == 0:
                start = p
            else:
                start = r
            # Generate all covered positions
            if start <= max_check:
                positions = np.arange(start, max_check + 1, p)
                counts[positions] += 1
        # Find the largest L such that counts[1..L] >= 2
        # Find first position where count < 2
        mask = counts[1:] < 2
        if not mask.any():
            return max_check
        first_gap = np.argmax(mask) + 1
        return first_gap - 1

    # Initial strategy: assign residues to maximize coverage
    # Start with a reasonable heuristic: try to cover small integers densely
    # Each prime p covers one residue class mod p.
    # For small integers x, x is covered by prime p if x ≡ r_p (mod p).
    # For x < p, this means r_p = x.
    # So for small x, we want many primes p > x to have r_p = x.
    
    # Strategy: For each prime, choose residue to maximize the minimum coverage in the prefix.
    # Greedy: process primes in order, for each prime choose the residue that maximizes coverage.
    
    # Better approach: Hill climbing from a good initial point.
    
    # Initial assignment: For prime i (0-indexed), assign residue to cover integer i+1 if possible
    # Prime p covers integer x if x ≡ r (mod p). For x < p, r = x.
    # For x >= p, r = x % p.
    
    # Let's start with: for the i-th prime, cover integer (i+1) % p
    # But we want every integer covered at least twice.
    
    # Alternative initial: random, then hill-climb.
    # But let's use a structured initial:
    # For prime p, assign r_p such that it covers a "needed" small integer.
    
    # Key insight: integers 1, 2, 3, ... need to be covered by at least 2 primes.
    # Integer x is covered by prime p iff x ≡ r_p (mod p).
    # For p > x: covered iff r_p = x.
    # For p <= x: covered iff r_p = x % p.
    
    # So for small x, many large primes can cover it by setting r_p = x.
    # For x=1: all primes p with r_p = 1 cover it.
    # For x=2: all primes p with r_p = 2 cover it.
    
    # To cover x twice, we need at least 2 primes with the appropriate residue.
    
    # Greedy construction:
    # Process integers x = 1, 2, 3, ...
    # For each x, ensure at least 2 primes cover it.
    # A prime p covers x if r_p ≡ x (mod p).
    
    # Let's do a different approach:
    # 1. Start with all residues = 0
    # 2. For x = 1, 2, 3, ...:
    #    Count how many primes cover x
    #    If < 2, find primes that don't cover x but could be reassigned to cover x
    #    Reassign the prime whose reassignment causes least damage
    
    # This is complex. Let's use hill climbing with a good initial.
    
    # Initial: for each prime p at index i, set r_p = (i+1) % p
    # This covers integer i+1 for prime p (when i+1 < p)
    
    def make_initial():
        residues = []
        for i, p in enumerate(primes):
            # Cover integer i+1
            x = i + 1
            r = x % p
            residues.append(r)
        return residues

    residues = make_initial()
    best_L = compute_L(residues)
    best_residues = list(residues)
    
    # Hill climbing
    random.seed(42)
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        
        # Try each prime, each possible residue
        for i in range(n_primes):
            if time.time() > deadline:
                break
            p = primes[i]
            old_r = residues[i]
            # Try a few random residues plus the old one
            # Focus on residues that might help: small values
            candidates = set()
            candidates.add(old_r)
            # Add some random candidates
            for _ in range(10):
                candidates.add(random.randint(0, p - 1))
            # Also try residues 0, 1, 2, ..., min(20, p-1)
            for r in range(min(30, p)):
                candidates.add(r)
            
            for new_r in candidates:
                if new_r == old_r:
                    continue
                residues[i] = new_r
                L = compute_L(residues)
                if L > best_L:
                    best_L = L
                    best_residues = list(residues)
                    improved = True
                    # Keep this change
                    break
                else:
                    residues[i] = old_r
            else:
                # No improvement for this prime
                residues[i] = old_r
    
    # Second phase: more aggressive hill climbing with larger neighborhood
    if time.time() < deadline:
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
            for i in range(n_primes):
                if time.time() > deadline:
                    break
                p = primes[i]
                old_r = residues[i]
                # Try more candidates
                candidates = set()
                candidates.add(old_r)
                for _ in range(50):
                    candidates.add(random.randint(0, p - 1))
                for r in range(min(50, p)):
                    candidates.add(r)
                
                for new_r in candidates:
                    if new_r == old_r:
                        continue
                    residues[i] = new_r
                    L = compute_L(residues)
                    if L > best_L:
                        best_L = L
                        best_residues = list(residues)
                        improved = True
                        break
                    else:
                        residues[i] = old_r
                else:
                    residues[i] = old_r
    
    # Third phase: simulated annealing-like for escape from local optima
    if time.time() < deadline:
        current_residues = list(best_residues)
        current_L = best_L
        T = 5.0
        for step in range(10000):
            if time.time() > deadline:
                break
            i = random.randint(0, n_primes - 1)
            p = primes[i]
            old_r = current_residues[i]
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue
            current_residues[i] = new_r
            L = compute_L(current_residues)
            if L >= current_L:
                current_L = L
                if L > best_L:
                    best_L = L
                    best_residues = list(current_residues)
            else:
                # Accept with probability exp((L - current_L) / T)
                delta = L - current_L
                if delta < 0 and random.random() < math.exp(delta / T):
                    pass  # accept
                else:
                    current_residues[i] = old_r
            T *= 0.999
    
    return best_residues
# EVOLVE-BLOCK-END
