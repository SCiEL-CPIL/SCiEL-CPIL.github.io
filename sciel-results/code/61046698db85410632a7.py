"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Generate primes up to 1000 using sieve
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(1000)
    n_primes = len(primes)
    
    # Precompute for fast coverage evaluation
    # For each prime p and residue r, the covered integers in [1, L] are r, r+p, r+2p, ...
    # We want to evaluate how many times each integer is covered
    
    def get_L(residues, max_check=10000):
        """Compute the largest L such that every integer 1..L has coverage >= 2."""
        # Use a counting array up to some reasonable bound
        # Start with a moderate bound and expand if needed
        bound = min(max_check, 2000)
        counts = np.zeros(bound + 1, dtype=np.int32)
        
        for i, p in enumerate(primes):
            r = residues[i]
            # Integers covered: r, r+p, r+2p, ... up to bound
            # But we need integers in [1, bound]
            # First integer >= 1 that is ≡ r (mod p)
            if r == 0:
                first = p  # 0, p, 2p, ... but we want >= 1, so p
            else:
                first = r
            if first < 1:
                first += p * ((1 - first + p - 1) // p)
            # Actually simpler: find smallest k such that r + k*p >= 1
            if r >= 1:
                first = r
            else:
                first = r + p  # r=0 case: first positive is p
            
            for x in range(first, bound + 1, p):
                counts[x] += 1
        
        # Find largest L where counts[1..L] are all >= 2
        L = 0
        for x in range(1, bound + 1):
            if counts[x] >= 2:
                L = x
            else:
                break
        return L
    
    def evaluate(residues):
        """Quick evaluation of L for given residues."""
        return get_L(residues)
    
    # Initial assignment: try several strategies
    best_residues = None
    best_L = 0
    
    # Strategy 1: Seed's approach
    residues1 = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    L1 = evaluate(residues1)
    if L1 > best_L:
        best_L = L1
        best_residues = residues1[:]
    
    # Strategy 2: All zeros
    residues2 = [0] * n_primes
    L2 = evaluate(residues2)
    if L2 > best_L:
        best_L = L2
        best_residues = residues2[:]
    
    # Strategy 3: r_p = 1 for all
    residues3 = [1 % p for p in primes]
    L3 = evaluate(residues3)
    if L3 > best_L:
        best_L = L3
        best_residues = residues3[:]
    
    # Strategy 4: Random initializations
    for _ in range(5):
        residues_rand = [random.randint(0, p - 1) for p in primes]
        Lr = evaluate(residues_rand)
        if Lr > best_L:
            best_L = Lr
            best_residues = residues_rand[:]
    
    # Local search / hill climbing
    if best_residues is not None:
        current = best_residues[:]
        current_L = best_L
        
        # Try single-coordinate improvements
        no_improve_count = 0
        max_no_improve = 5000
        random.seed(42)
        
        while time.time() < deadline - 5.0 and no_improve_count < max_no_improve:
            # Pick a random coordinate to try changing
            i = random.randint(0, n_primes - 1)
            p = primes[i]
            old_r = current[i]
            
            # Try a few new values
            improved = False
            for _ in range(3):
                new_r = random.randint(0, p - 1)
                if new_r == old_r:
                    continue
                current[i] = new_r
                new_L = evaluate(current)
                if new_L > current_L:
                    current_L = new_L
                    improved = True
                    break
                else:
                    current[i] = old_r
            
            if improved:
                no_improve_count = 0
                best_L = current_L
                best_residues = current[:]
            else:
                no_improve_count += 1
        
        # More aggressive: try all residues for each prime in order of "importance"
        # Primes with smaller p have more frequent coverage, so they matter more
        if time.time() < deadline - 10.0:
            # Sort primes by size (smaller primes first - they cover more integers)
            order = sorted(range(n_primes), key=lambda i: primes[i])
            
            for _ in range(3):  # Multiple passes
                if time.time() > deadline - 10.0:
                    break
                for i in order:
                    if time.time() > deadline - 10.0:
                        break
                    p = primes[i]
                    old_r = current[i]
                    best_r = old_r
                    best_local_L = current_L
                    
                    # Try a subset of residues (can't try all if p is large)
                    if p <= 50:
                        candidates = list(range(p))
                    else:
                        candidates = [old_r] + [random.randint(0, p-1) for _ in range(20)]
                    
                    for new_r in candidates:
                        if new_r == old_r:
                            continue
                        current[i] = new_r
                        new_L = evaluate(current)
                        if new_L > best_local_L:
                            best_local_L = new_L
                            best_r = new_r
                    
                    current[i] = best_r
                    if best_r != old_r:
                        current_L = best_local_L
                        best_L = current_L
                        best_residues = current[:]
    
    return best_residues if best_residues is not None else [((i + 2) // 2) % p for i, p in enumerate(primes)]
# EVOLVE-BLOCK-END
