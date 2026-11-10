import numpy as np
import time

def construct():
    # Generate primes up to 1000
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    n_primes = len(primes)
    
    best_L = 0
    best_residues = None
    
    start_time = time.time()
    time_limit = 115  # seconds, leave margin for 120s limit
    
    current_L_MAX = 1000
    
    for restart in range(200):
        if time.time() - start_time > time_limit:
            break
            
        # Random initialization
        residues = np.array([np.random.randint(p) for p in primes])
        
        # Build coverage array
        cov = np.zeros(current_L_MAX + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            cov[r::p] += 1
            
        # Evaluate current L
        fails = np.where(cov[1:] < 2)[0]
        if len(fails) == 0:
            L = current_L_MAX
        else:
            L = fails[0]
            
        if L > best_L:
            best_L = L
            best_residues = residues.copy()
            
        # Repair loop
        stuck = False
        while not stuck:
            if time.time() - start_time > time_limit:
                break
                
            fails = np.where(cov[1:] < 2)[0]
            if len(fails) == 0:
                L = current_L_MAX
                if L > best_L:
                    best_L = L
                    best_residues = residues.copy()
                
                # If we reached the limit, extend it
                if L >= 1000:
                    # We can stop if we reached 1000, as that's the target.
                    # But let's try to go a bit higher if time permits.
                    # For now, break to restart or next restart.
                    break
                else:
                    # Extend L_MAX
                    current_L_MAX += 500
                    cov = np.zeros(current_L_MAX + 1, dtype=np.int32)
                    for i, p in enumerate(primes):
                        r = residues[i]
                        cov[r::p] += 1
                    continue
            
            L = fails[0]
            x_fail = L + 1
            
            # Find a safe prime to change
            # Iterate from largest prime to smallest to find safe moves more easily
            found_safe = False
            for i in range(n_primes - 1, -1, -1):
                p = primes[i]
                r_old = residues[i]
                
                # If this prime already covers x_fail, changing it won't increase coverage at x_fail
                if x_fail % p == r_old:
                    continue
                
                r_new = x_fail % p
                
                # Check if changing r_old to r_new breaks the prefix [1, x_fail-1]
                # Points y < x_fail covered by r_old but not r_new will lose 1 coverage.
                # We need cov[y] >= 3 for all such y.
                
                y = r_old
                if y < 1:
                    y += p
                    
                safe = True
                while y < x_fail:
                    if cov[y] < 3:
                        safe = False
                        break
                    y += p
                
                if safe:
                    # Apply change
                    cov[r_old::p] -= 1
                    cov[r_new::p] += 1
                    residues[i] = r_new
                    found_safe = True
                    break
            
            if not found_safe:
                # Stuck. Break and restart.
                stuck = True

    return list(best_residues)
