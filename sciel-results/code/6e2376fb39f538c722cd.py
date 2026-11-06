import time
import random
import numpy as np

def construct():
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                is_prime[i*i:n+1:i] = [False] * len(is_prime[i*i:n+1:i])
        return [i for i, prime in enumerate(is_prime) if prime]

    primes = sieve(1000)
    n_primes = len(primes)
    
    residues = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    
    MAX_L_CHECK = 5000
    
    counts = np.zeros(MAX_L_CHECK + 1, dtype=np.int8)
    
    def update_counts(r_p, p, delta):
        start = r_p
        if start == 0:
            start = p
        # NumPy slice assignment is fast
        counts[start:MAX_L_CHECK+1:p] += delta

    for i, p in enumerate(primes):
        update_counts(residues[i], p, 1)
        
    def compute_L_from_counts():
        L = 0
        # Find first index where count < 2
        # np.argmax on a boolean array?
        # Or just loop. Loop is fine for 5000.
        for x in range(1, MAX_L_CHECK + 1):
            if counts[x] >= 2:
                L = x
            else:
                break
        return L

    current_L = compute_L_from_counts()
    best_L = current_L
    best_residues = residues.copy()
    
    start_time = time.time()
    TIME_LIMIT = 110.0
    
    while time.time() - start_time < TIME_LIMIT:
        improved = False
        for i, p in enumerate(primes):
            r_old = residues[i]
            
            candidates = []
            for _ in range(10):
                r_new = random.randint(0, p - 1)
                if r_new != r_old:
                    candidates.append(r_new)
            for delta in [-2, -1, 1, 2]:
                r_new = (r_old + delta) % p
                if r_new != r_old and r_new not in candidates:
                    candidates.append(r_new)
            
            best_candidate_L = current_L
            best_candidate_r = r_old
            
            for r_new in candidates:
                # Check validity
                # Points where count decreases: x % p == r_old
                # If r_old == 0, x = p, 2p...
                # If r_old != 0, x = r_old, r_old+p...
                
                valid = True
                start_check = r_old
                if start_check == 0:
                    start_check = p
                else:
                    # Ensure start_check >= 1
                    if start_check < 1:
                        start_check = 1
                        
                # Check if any count in this class is 1
                # We can use numpy:
                # indices = np.arange(start_check, current_L + 1, p)
                # if np.any(counts[indices] == 1):
                #    valid = False
                
                # This creates an array, which has overhead.
                # For small L, loop is better.
                # Let's use loop for now, but optimized.
                
                x = start_check
                while x <= current_L:
                    if counts[x] == 1:
                        valid = False
                        break
                    x += p
                
                if not valid:
                    continue
                
                # Apply change
                update_counts(r_old, p, -1)
                update_counts(r_new, p, 1)
                
                # Compute new L
                new_L = current_L
                x = current_L + 1
                while x <= MAX_L_CHECK and counts[x] >= 2:
                    new_L = x
                    x += 1
                
                if new_L > best_candidate_L:
                    best_candidate_L = new_L
                    best_candidate_r = r_new
                    
                # Revert
                update_counts(r_new, p, -1)
                update_counts(r_old, p, 1)
                
            if best_candidate_L > current_L:
                update_counts(r_old, p, -1)
                update_counts(best_candidate_r, p, 1)
                residues[i] = best_candidate_r
                current_L = best_candidate_L
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues.copy()
                improved = True
                
        if not improved:
            # Plateau walk
            i = random.randint(0, n_primes - 1)
            p = primes[i]
            r_old = residues[i]
            
            found = False
            for _ in range(50):
                r_new = random.randint(0, p - 1)
                if r_new == r_old:
                    continue
                
                valid = True
                start_check = r_old
                if start_check == 0:
                    start_check = p
                x = start_check
                while x <= current_L:
                    if counts[x] == 1:
                        valid = False
                        break
                    x += p
                    
                if valid:
                    update_counts(r_old, p, -1)
                    update_counts(r_new, p, 1)
                    residues[i] = r_new
                    found = True
                    break
            
            if not found:
                # If we can't move, we might be stuck.
                # To avoid infinite loop, we can just break.
                break

    return best_residues
