import numpy as np
import time

def construct():
    # Generate primes from 13 to 997
    def is_prime(n):
        if n < 2:
            return False
        if n % 2 == 0:
            return n == 2
        for i in range(3, int(n ** 0.5) + 1, 2):
            if n % i == 0:
                return False
        return True

    primes = [p for p in range(13, 1001) if is_prime(p)]
    num_primes = len(primes)
    
    # We want to maximize L. 
    # Set a reasonably high upper bound for search.
    L_MAX = 5000
    
    best_L = 0
    best_residues = [0] * num_primes
    
    deadline = time.time() + 110
    
    # Helper to update counts array for a prime/residue
    def update_counts(counts, p, r, delta):
        if r == 0:
            # Multiples of p: p, 2p, ...
            if p <= L_MAX:
                idx = np.arange(p, L_MAX + 1, p)
                if len(idx) > 0:
                    counts[idx] += delta
        else:
            # r, r+p, ...
            if r <= L_MAX:
                idx = np.arange(r, L_MAX + 1, p)
                if len(idx) > 0:
                    counts[idx] += delta

    def find_gap(counts):
        # Find smallest n >= 1 such that counts[n] == 0
        # counts[0] is ignored
        uncovered = np.where(counts[1:] == 0)[0]
        if len(uncovered) == 0:
            return L_MAX + 1
        return uncovered[0] + 1

    # Multi-start local search
    while time.time() < deadline:
        # Random initialization
        residues = [np.random.randint(p) for p in primes]
        counts = np.zeros(L_MAX + 1, dtype=np.int8)
        
        for p, r in zip(primes, residues):
            update_counts(counts, p, r, 1)
            
        current_L = find_gap(counts)
        if current_L > best_L:
            best_L = current_L
            best_residues = residues.copy()
            
        # Local search: fix the current gap
        improved = True
        while improved and time.time() < deadline:
            improved = False
            G = find_gap(counts)
            if G > best_L:
                best_L = G
                best_residues = residues.copy()
                
            # Try to find a prime that can cover G by changing its residue
            # We only consider the residue r = G % p for each prime p
            for i, p in enumerate(primes):
                r_cand = G % p
                r_curr = residues[i]
                
                if r_cand == r_curr:
                    continue
                
                # Remove current residue
                update_counts(counts, p, r_curr, -1)
                # Add candidate residue
                update_counts(counts, p, r_cand, 1)
                
                G_new = find_gap(counts)
                
                if G_new > G:
                    # Accept the move
                    residues[i] = r_cand
                    improved = True
                    break # Restart the loop with the new gap G_new
                else:
                    # Revert
                    update_counts(counts, p, r_cand, -1)
                    update_counts(counts, p, r_curr, 1)
                    
        # If we are stuck, the outer loop will try a new random initialization
        # We can also add a small random perturbation here to escape local minima
        # within the same initialization if we have time, but multi-start is simpler.
        
    return [[p, r] for p, r in zip(primes, best_residues)]
