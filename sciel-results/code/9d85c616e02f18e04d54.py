import numpy as np
import time

def construct():
    # Primes <= 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]

    primes = sieve(1000)
    n_primes = len(primes)
    M = 10000
    
    # State
    residues = np.zeros(n_primes, dtype=int)
    counts = np.zeros(M + 1, dtype=int)
    # counts[0] is unused, we care about 1..M
    # Initialize counts[0] to high value so it's never a failure
    counts[0] = 100
    
    def update_counts(p_idx, old_r, new_r):
        p = primes[p_idx]
        # Subtract old
        if old_r != -1: # -1 means uninit? No, we init properly.
            counts[old_r::p] -= 1
        # Add new
        counts[new_r::p] += 1

    def get_L():
        # Find first index >= 1 where counts < 2
        # Use np.argmax on boolean array
        # If all >= 2, argmax returns 0 (if mask[0] is False? No, argmax returns first True)
        # If no True, returns 0.
        # So we check if mask[0] is True? No, we set counts[0]=100, so mask[0] is False.
        # If all 1..M are >= 2, mask is all False, argmax returns 0.
        # We need to distinguish "failure at 0" (impossible) from "no failure".
        # If no failure, L = M.
        mask = counts[1:] < 2
        if not np.any(mask):
            return M
        idx = np.argmax(mask)
        return idx + 1

    # 1. Greedy Initialization
    # We want to ensure 1..M are covered.
    # Iterate x from 1 to M.
    # If counts[x] < 2, fix it.
    # To fix, pick a prime p where residues[i] != x % p.
    # Randomize the order of primes to avoid bias?
    
    # Start with random residues to break symmetry
    residues = np.random.randint(0, 1000, n_primes)
    for i, p in enumerate(primes):
        residues[i] %= p
        counts[residues[i]::p] += 1
    
    # Now repair
    for x in range(1, M + 1):
        if counts[x] < 2:
            # Find a prime to change
            # Candidates: i where residues[i] != x % primes[i]
            # We can just pick a random one
            # To be efficient, pre-compute? No, just loop.
            # Since n_primes=168, loop is fine.
            changed = False
            # Shuffle indices for randomness
            indices = np.random.permutation(n_primes)
            for i in indices:
                p = primes[i]
                target_r = x % p
                if residues[i] != target_r:
                    # Change it
                    old_r = residues[i]
                    residues[i] = target_r
                    update_counts(i, old_r, target_r)
                    changed = True
                    break
            if not changed:
                # This shouldn't happen if M is small enough and we have enough primes
                # But if we are stuck, we might break
                break

    current_L = get_L()
    
    start_time = time.time()
    deadline = start_time + 115 # 115s to leave margin
    
    # 2. Local Search
    best_L = current_L
    best_residues = residues.copy()
    
    while time.time() < deadline:
        current_L = get_L()
        if current_L >= M:
            # We hit the limit, maybe increase M?
            # For now, break.
            break
        
        x_fail = current_L
        
        # Find candidates
        # Primes i where residues[i] != x_fail % primes[i]
        candidates = []
        for i in range(n_primes):
            p = primes[i]
            if residues[i] != x_fail % p:
                candidates.append(i)
        
        if not candidates:
            # Stuck, perturb
            for _ in range(5):
                i = np.random.randint(n_primes)
                old_r = residues[i]
                new_r = np.random.randint(primes[i])
                residues[i] = new_r
                update_counts(i, old_r, new_r)
            continue
        
        best_move = None
        best_new_L = current_L
        
        # Try a subset of candidates if too many? 168 is small.
        # Shuffle candidates to randomize tie-breaking
        np.random.shuffle(candidates)
        
        for i in candidates:
            p = primes[i]
            old_r = residues[i]
            new_r = x_fail % p
            
            # Apply
            update_counts(i, old_r, new_r)
            
            # Evaluate
            new_L = get_L()
            
            # Revert
            update_counts(i, new_r, old_r)
            
            if new_L > best_new_L:
                best_new_L = new_L
                best_move = (i, old_r, new_r)
                
        if best_move:
            i, old_r, new_r = best_move
            residues[i] = new_r
            update_counts(i, old_r, new_r)
            current_L = best_new_L
            if current_L > best_L:
                best_L = current_L
                best_residues = residues.copy()
        else:
            # No improvement, perturb
            # Change 3 random residues
            for _ in range(3):
                i = np.random.randint(n_primes)
                old_r = residues[i]
                new_r = np.random.randint(primes[i])
                residues[i] = new_r
                update_counts(i, old_r, new_r)
                
    return best_residues.tolist()
