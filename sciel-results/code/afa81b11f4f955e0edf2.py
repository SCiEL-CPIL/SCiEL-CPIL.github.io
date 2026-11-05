import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Generate primes <= 1000
    primes = []
    for p in range(2, 1001):
        is_prime = True
        for q in range(2, int(p ** 0.5) + 1):
            if p % q == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(p)
            
    n_primes = len(primes)
    L_LIMIT = 100000
    
    # Initialize residues
    residues = np.array([((i + 2) // 2) % p for i, p in enumerate(primes)], dtype=int)
    
    # Initialize coverage array
    cov = np.zeros(L_LIMIT + 1, dtype=np.int32)
    
    def update_cov(p_idx, old_r, new_r):
        p = primes[p_idx]
        # Remove old
        start_old = old_r
        if start_old == 0:
            start_old = p
        if start_old <= L_LIMIT:
            cov[start_old:L_LIMIT+1:p] -= 1
            
        # Add new
        start_new = new_r
        if start_new == 0:
            start_new = p
        if start_new <= L_LIMIT:
            cov[start_new:L_LIMIT+1:p] += 1
            
    # Build initial cov
    for i, p in enumerate(primes):
        r = residues[i]
        start = r
        if start == 0:
            start = p
        if start <= L_LIMIT:
            cov[start:L_LIMIT+1:p] += 1
            
    def get_L():
        # Find first index where cov < 2
        # np.where(cov < 2)
        # Note: cov[0] is unused.
        bad_indices = np.where(cov[1:] < 2)[0]
        if len(bad_indices) == 0:
            return L_LIMIT
        return bad_indices[0] # This is x-1
        
    current_L = get_L()
    best_L = current_L
    best_residues = residues.copy()
    
    while time.time() < deadline:
        x_fail = current_L + 1
        
        if x_fail > L_LIMIT:
            break
            
        # Find candidates for primes that don't cover x_fail
        # x_fail % p != residues[i]
        
        # Vectorized check for candidates
        # x_fail_mods = [x_fail % p for p in primes]
        # candidates = np.where(x_fail_mods != residues)[0]
        
        # However, we also want to check safety.
        # Safety: No y < x_fail with y % p == residues[i] and cov[y] == 2.
        
        # This safety check is hard to vectorize easily for all p at once.
        # Let's iterate candidates.
        
        # To speed up, we can precompute the "bad" y's for each prime?
        # No, cov changes.
        
        # Let's just iterate candidates in order of prime size (largest first)
        # and check safety.
        
        # Get list of indices where x_fail % p != residues[i]
        # We can compute x_fail % primes vectorized
        x_fail_mods = np.array([x_fail % p for p in primes])
        candidate_indices = np.where(x_fail_mods != residues)[0]
        
        # Sort by prime size descending
        candidate_indices = candidate_indices[np.argsort(primes[candidate_indices])[::-1]]
        
        improved = False
        for i in candidate_indices:
            p = primes[i]
            r_old = residues[i]
            r_new = x_fail % p
            
            # Check safety: 
            # Are there any y in 1..x_fail-1 such that y % p == r_old and cov[y] == 2?
            
            # Generate y's
            start_y = r_old
            if start_y == 0:
                start_y = p
            if start_y < 1:
                start_y = 1
                
            # We only care about y < x_fail
            y_indices = np.arange(start_y, x_fail, p)
            
            if len(y_indices) > 0:
                # Check if any of these have cov == 2
                if np.any(cov[y_indices] == 2):
                    continue # Unsafe
            
            # Safe! Apply move
            update_cov(i, r_old, r_new)
            residues[i] = r_new
            
            new_L = get_L()
            if new_L > current_L:
                current_L = new_L
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues.copy()
                improved = True
                break # Restart loop with new gap
            else:
                # Revert
                update_cov(i, r_new, r_old)
                residues[i] = r_old
                
        if not improved:
            # Try a random move
            i_rand = random.randint(0, n_primes - 1)
            p_rand = primes[i_rand]
            r_old_rand = residues[i_rand]
            r_new_rand = random.randint(0, p_rand - 1)
            
            if r_old_rand != r_new_rand:
                update_cov(i_rand, r_old_rand, r_new_rand)
                residues[i_rand] = r_new_rand
                
                new_L_rand = get_L()
                if new_L_rand >= current_L:
                    current_L = new_L_rand
                    if current_L > best_L:
                        best_L = current_L
                        best_residues = residues.copy()
                else:
                    update_cov(i_rand, r_new_rand, r_old_rand)
                    residues[i_rand] = r_old_rand

    return best_residues.tolist()

# EVOLVE-BLOCK-END
