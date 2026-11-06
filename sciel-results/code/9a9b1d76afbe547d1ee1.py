import time
import random

def construct():
    start_time = time.time()
    time_limit = 115
    
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if sieve[i]:
                for j in range(i * i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(1000)
    P = len(primes)
    
    # Initial residues
    residues = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    
    L_MAX = 100000
    counts = [0] * (L_MAX + 1)
    
    def apply_change(idx, old_r, new_r):
        p = primes[idx]
        # Remove old
        if old_r == 0:
            start_x = p
        else:
            start_x = old_r
        for x in range(start_x, L_MAX + 1, p):
            counts[x] -= 1
            
        # Add new
        if new_r == 0:
            start_x = p
        else:
            start_x = new_r
        for x in range(start_x, L_MAX + 1, p):
            counts[x] += 1
            
    # Build initial counts
    for i, p in enumerate(primes):
        r = residues[i]
        if r == 0:
            start_x = p
        else:
            start_x = r
        for x in range(start_x, L_MAX + 1, p):
            counts[x] += 1
            
    def get_score():
        for x in range(1, L_MAX + 1):
            if counts[x] < 2:
                return x - 1
        return L_MAX
        
    current_score = get_score()
    best_score = current_score
    best_residues = residues[:]
    
    # Local Search
    while time.time() - start_time < time_limit:
        i = random.randint(0, P - 1)
        p = primes[i]
        old_r = residues[i]
        new_r = random.randint(0, p - 1)
        
        if new_r == old_r:
            continue
            
        # Apply
        apply_change(i, old_r, new_r)
        residues[i] = new_r
        
        new_score = get_score()
        
        if new_score > current_score:
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_residues = residues[:]
        else:
            # Revert
            apply_change(i, new_r, old_r)
            residues[i] = old_r
            
    return best_residues
