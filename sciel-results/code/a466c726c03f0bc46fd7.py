import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i in range(2, n+1) if is_prime[i]]
    
    primes = sieve(1000)
    n_primes = len(primes)
    
    # Initial random assignment
    residues = [random.randint(0, p-1) for p in primes]
    
    def compute_L(residues, max_L=100000):
        """Compute the largest L such that every x in 1..L has coverage >= 2."""
        # We'll compute coverage up to some bound, then find the first gap
        # Start with a reasonable bound, expand if needed
        L = 1000
        while True:
            if L > max_L:
                return max_L
            cov = np.zeros(L + 1, dtype=np.int32)
            for i, p in enumerate(primes):
                r = residues[i]
                # Positions x in 1..L where x % p == r
                if r == 0:
                    # positions: p, 2p, 3p, ...
                    if p <= L:
                        cov[p::p] += 1
                else:
                    # positions: r, r+p, r+2p, ...
                    if r <= L:
                        cov[r::p] += 1
            # Find first position where cov < 2
            # Check positions 1..L
            if L > 0:
                min_cov = np.min(cov[1:L+1]) if L >= 1 else 0
                if min_cov >= 2:
                    # All positions 1..L have coverage >= 2
                    if L >= max_L:
                        return max_L
                    L = min(L * 2, max_L + 1)
                    continue
                else:
                    # Find first position with coverage < 2
                    bad = np.where(cov[1:L+1] < 2)[0]
                    if len(bad) > 0:
                        return bad[0]  # 0-indexed in the slice, so actual position is bad[0]+1
                    else:
                        return L
            else:
                return 0
    
    # Optimize compute_L: just compute up to a fixed bound and find the gap
    def compute_L_fast(residues, bound=100000):
        cov = np.zeros(bound + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            if r == 0:
                if p <= bound:
                    cov[p::p] += 1
            else:
                if r <= bound:
                    cov[r::p] += 1
        # Find first x in 1..bound where cov[x] < 2
        # Use numpy to find first index where cov[1:bound+1] < 2
        bad_mask = cov[1:bound+1] < 2
        if np.any(bad_mask):
            first_bad = np.argmax(bad_mask) + 1  # +1 because we're looking at cov[1:]
            return first_bad
        return bound
    
    # Initial score
    best_residues = residues.copy()
    best_score = compute_L_fast(best_residues)
    
    # Local search: try changing one residue at a time
    # For each prime, try all possible residues and pick the best
    # This is O(n_primes * avg_p * cost_of_eval) which might be too slow
    # Instead, do random restarts + hill climbing
    
    def hill_climb(residues):
        current = residues.copy()
        current_score = compute_L_fast(current)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try each prime in random order
            order = list(range(n_primes))
            random.shuffle(order)
            for i in order:
                if time.time() >= deadline - 5:
                    break
                p = primes[i]
                old_r = current[i]
                # Try a random new residue
                new_r = random.randint(0, p-1)
                if new_r == old_r:
                    continue
                current[i] = new_r
                new_score = compute_L_fast(current)
                if new_score > current_score:
                    current_score = new_score
                    improved = True
                else:
                    current[i] = old_r
        return current, current_score
    
    # Multiple restarts
    best_residues = residues.copy()
    best_score = compute_L_fast(best_residues)
    
    for restart in range(50):
        if time.time() >= deadline - 10:
            break
        # Random restart
        current = [random.randint(0, p-1) for p in primes]
        current_score = compute_L_fast(current)
        
        # Hill climb
        current, current_score = hill_climb(current)
        
        if current_score > best_score:
            best_score = current_score
            best_residues = current.copy()
    
    return best_residues
