import time
import random
import numpy as np

def construct():
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    n_primes = len(primes)
    MAX_L = 20000
    
    # Use a fixed seed for reproducibility in tests, but random for search?
    # The task doesn't specify deterministic output, just maximize score.
    # We will use random.
    
    residues = [random.randint(0, p - 1) for p in primes]
    
    best_L = 0
    best_residues = residues[:]
    
    start_time = time.time()
    timeout = 140.0
    
    def get_coverage(residues):
        diff = np.zeros(MAX_L + 2, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            start = r if r > 0 else p
            if start > MAX_L:
                continue
            # Vectorized increment
            # Indices: start, start+p, ...
            idx = np.arange(start, MAX_L + 1, p)
            diff[idx] += 1
        cov = np.cumsum(diff)
        return cov[1:MAX_L+1]
        
    def get_L(cov):
        if len(cov) == 0: return 0
        mask = cov >= 2
        if not mask.any():
            return 0
        if (~mask).any():
            return np.argmax(~mask)
        else:
            return MAX_L

    cov = get_coverage(residues)
    current_L = get_L(cov)
    if current_L > best_L:
        best_L = current_L
        best_residues = residues[:]

    while time.time() - start_time < timeout:
        if current_L >= MAX_L:
            break
            
        x_fail = current_L + 1
        
        moved = False
        candidates = []
        for i, p in enumerate(primes):
            new_r = x_fail % p
            if new_r == residues[i]:
                continue
            candidates.append(i)
            
        # Sort by p descending to prefer large primes (safer moves)
        candidates.sort(key=lambda i: primes[i], reverse=True)
        
        # Try top 20 candidates
        for i in candidates[:20]:
            p = primes[i]
            old_r = residues[i]
            new_r = x_fail % p
            
            # Check validity using NumPy
            # Old hits < x_fail
            if old_r == 0:
                start_y = p
            else:
                start_y = old_r
                
            if start_y >= x_fail:
                # No old hits in prefix
                valid = True
            else:
                old_hits = np.arange(start_y, x_fail, p)
                # Filter out those covered by new_r
                # y % p == new_r
                mask_new = (old_hits % p) == new_r
                check_hits = old_hits[~mask_new]
                
                if len(check_hits) > 0:
                    # Check if any of these have cov == 2
                    # cov is 0-indexed for 1..MAX_L
                    # check_hits are 1-indexed
                    check_indices = check_hits - 1
                    if np.any(cov[check_indices] == 2):
                        valid = False
                    else:
                        valid = True
                else:
                    valid = True
            
            if valid:
                residues[i] = new_r
                moved = True
                break
        
        if not moved:
            # Random perturbation
            # Bias towards primes that cover x_fail?
            # Just random
            i = random.randint(0, n_primes - 1)
            p = primes[i]
            residues[i] = random.randint(0, p - 1)
            
        cov = get_coverage(residues)
        current_L = get_L(cov)
        if current_L > best_L:
            best_L = current_L
            best_residues = residues[:]

    return best_residues
