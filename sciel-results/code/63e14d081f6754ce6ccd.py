"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145  # 145s safety margin
    
    # Generate primes up to 1000
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                sieve[i*i::i] = [False] * len(sieve[i*i::i])
        return [i for i, is_p in enumerate(sieve) if is_p]
    
    primes = get_primes(1000)
    n_primes = len(primes)  # 168
    
    # Initial assignment: same as seed
    residues = np.array([((i + 2) // 2) % p for i, p in enumerate(primes)], dtype=np.int64)
    
    def compute_counts(residues, max_L):
        """Compute coverage count for each integer in 1..max_L."""
        counts = np.zeros(max_L + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            if r == 0:
                # covers p, 2p, 3p, ...
                start = p
            else:
                start = r
            if start <= max_L:
                # Add 1 to positions start, start+p, start+2p, ...
                counts[start::p] += 1
        return counts
    
    def get_L(residues, max_check=10000):
        """Find the largest L such that all integers 1..L have count >= 2."""
        # Check in chunks for efficiency
        # First find a rough upper bound
        # Try increasing L
        L = 0
        # Check up to max_check
        # We'll compute counts incrementally
        # For efficiency, compute counts for a range and find first position with count < 2
        
        # Compute counts for 1..max_check
        counts = compute_counts(residues, max_check)
        # Find first position where count < 2
        # positions 1..max_check
        bad = np.where(counts[1:max_check+1] < 2)[0]
        if len(bad) == 0:
            return max_check
        return bad[0]  # first bad position, so L = bad[0] - 1 + 1 = bad[0]
        # Wait: if counts[1] < 2, then L = 0 (no integer satisfies)
        # If counts[1] >= 2 but counts[2] < 2, then L = 1
        # bad[0] is the index in counts[1:max_check+1], so actual position is bad[0] + 1
        # L = bad[0] (since positions 1..bad[0] are good, position bad[0]+1 is bad)
        # Actually: bad[0] = 0 means position 1 is bad, L = 0
        # bad[0] = 1 means position 2 is bad, positions 1 is good, L = 1
        # So L = bad[0]
    
    # Let me fix get_L
    def get_L_fixed(residues, max_check=10000):
        counts = compute_counts(residues, max_check)
        # Check positions 1 through max_check
        # Find first position j in 1..max_check where counts[j] < 2
        sub = counts[1:max_check+1]
        bad_indices = np.where(sub < 2)[0]
        if len(bad_indices) == 0:
            return max_check
        # bad_indices[0] is 0-indexed in sub, so actual position is bad_indices[0] + 1
        # All positions 1..bad_indices[0] have count >= 2
        # So L = bad_indices[0]
        return bad_indices[0]
    
    # Initial L
    current_L = get_L_fixed(residues, 10000)
    best_residues = residues.copy()
    best_L = current_L
    
    # Local search
    # For each prime, try changing its residue and see if L improves
    # Use simulated annealing or simple hill climbing
    
    # To speed up, we can precompute which positions each prime covers
    # For a given prime p and residue r, the covered positions in 1..max_check are:
    #   if r == 0: p, 2p, 3p, ...
    #   else: r, r+p, r+2p, ...
    
    # Key optimization: when we change residue for prime i, only the counts at
    # positions covered by prime i change. So we can update counts incrementally.
    
    max_check = 10000
    counts = compute_counts(residues, max_check)
    
    def try_change(residues, counts, i, new_r, max_check):
        """Try changing residue of prime i to new_r, return new counts and new L."""
        p = primes[i]
        old_r = residues[i]
        if old_r == new_r:
            return counts, get_L_fixed(residues, max_check)
        
        # Remove old coverage
        if old_r == 0:
            old_start = p
        else:
            old_start = old_r
        if old_start <= max_check:
            counts[old_start::p] -= 1
        
        # Add new coverage
        if new_r == 0:
            new_start = p
        else:
            new_start = new_r
        if new_start <= max_check:
            counts[new_start::p] += 1
        
        residues[i] = new_r
        L = get_L_fixed(residues, max_check)
        return counts, L
    
    # Simple hill climbing with multiple passes
    import random
    random.seed(42)
    
    for iteration in range(200):  # up to 200 iterations
        if time.time() > deadline - 5:
            break
        
        improved = False
        # Try each prime in random order
        order = list(range(n_primes))
        random.shuffle(order)
        
        for i in order:
            if time.time() > deadline - 5:
                break
            
            p = primes[i]
            current_r = residues[i]
            current_L_local = get_L_fixed(residues, max_check)
            
            # Try a few random new residues
            for _ in range(3):
                new_r = random.randint(0, p - 1)
                if new_r == current_r:
                    continue
                
                # Save old state
                old_counts = counts.copy()
                old_r_saved = residues[i]
                
                # Apply change
                if old_r_saved == 0:
                    old_start = p
                else:
                    old_start = old_r_saved
                if old_start <= max_check:
                    counts[old_start::p] -= 1
                
                if new_r == 0:
                    new_start = p
                else:
                    new_start = new_r
                if new_start <= max_check:
                    counts[new_start::p] += 1
                
                residues[i] = new_r
                new_L = get_L_fixed(residues, max_check)
                
                if new_L > current_L_local:
                    current_L_local = new_L
                    improved = True
                    if new_L > best_L:
                        best_L = new_L
                        best_residues = residues.copy()
                    break
                else:
                    # Revert
                    counts = old_counts
                    residues[i] = old_r_saved
        
        if not improved:
            # Try a more aggressive search: swap two residues
            if iteration < 50:  # limit swap attempts
                for _ in range(10):
                    if time.time() > deadline - 5:
                        break
                    i1 = random.randint(0, n_primes - 1)
                    i2 = random.randint(0, n_primes - 1)
                    if i1 == i2:
                        continue
                    
                    old_r1 = residues[i1]
                    old_r2 = residues[i2]
                    p1 = primes[i1]
                    p2 = primes[i2]
                    
                    new_r1 = random.randint(0, p1 - 1)
                    new_r2 = random.randint(0, p2 - 1)
                    
                    if new_r1 == old_r1 and new_r2 == old_r2:
                        continue
                    
                    old_counts = counts.copy()
                    
                    # Apply changes
                    for (idx, p, old_r, new_r) in [(i1, p1, old_r1, new_r1), (i2, p2, old_r2, new_r2)]:
                        if old_r == 0:
                            s = p
                        else:
                            s = old_r
                        if s <= max_check:
                            counts[s::p] -= 1
                        if new_r == 0:
                            s = p
                        else:
                            s = new_r
                        if s <= max_check:
                            counts[s::p] += 1
                        residues[idx] = new_r
                    
                    new_L = get_L_fixed(residues, max_check)
                    if new_L > best_L:
                        best_L = new_L
                        best_residues = residues.copy()
                    else:
                        counts = old_counts
                        residues[i1] = old_r1
                        residues[i2] = old_r2
    
    # Return best residues found
    result = [int(r) for r in best_residues]
    return result
# EVOLVE-BLOCK-END
