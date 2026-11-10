import time
import random
import math

def construct():
    start_time = time.time()
    deadline = 110  # seconds
    
    # Generate primes <= 1000
    primes = []
    for p in range(2, 1001):
        is_prime = True
        for q in range(2, int(p**0.5) + 1):
            if p % q == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(p)
            
    n_primes = len(primes)
    # Initial residues: seed strategy
    residues = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    
    MAX_L = 2000
    cover = [0] * (MAX_L + 1)
    
    # Initialize cover array
    for i, p in enumerate(primes):
        r = residues[i]
        for x in range(r, MAX_L + 1, p):
            if x > 0:
                cover[x] += 1
            else:
                # x=0 is not in 1..L, but r can be 0
                # if r=0, x=p, 2p, ...
                pass
        # Correct loop: x = r, r+p, ...
        # If r=0, x=p, 2p...
        # If r>0, x=r, r+p...
        
    def compute_L(cover):
        L = 0
        for x in range(1, MAX_L + 1):
            if cover[x] >= 2:
                L = x
            else:
                break
        return L
        
    # Re-init cover correctly
    cover = [0] * (MAX_L + 1)
    for i, p in enumerate(primes):
        r = residues[i]
        if r == 0:
            start = p
        else:
            start = r
        for x in range(start, MAX_L + 1, p):
            cover[x] += 1
            
    best_L = compute_L(cover)
    best_residues = residues[:]
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    alpha = 0.9999
    current_L = best_L
    current_residues = residues[:]
    current_cover = cover[:]
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Pick random prime
        i = random.randint(0, n_primes - 1)
        p = primes[i]
        old_r = current_residues[i]
        
        # Pick new random residue
        new_r = random.randint(0, p - 1)
        if new_r == old_r:
            continue
            
        # Update cover
        # Remove old_r
        if old_r == 0:
            start_old = p
        else:
            start_old = old_r
        for x in range(start_old, MAX_L + 1, p):
            current_cover[x] -= 1
            
        # Add new_r
        if new_r == 0:
            start_new = p
        else:
            start_new = new_r
        for x in range(start_new, MAX_L + 1, p):
            current_cover[x] += 1
            
        new_L = compute_L(current_cover)
        
        # Acceptance
        if new_L >= current_L:
            current_L = new_L
            current_residues[i] = new_r
            if new_L > best_L:
                best_L = new_L
                best_residues = current_residues[:]
        else:
            # Delta is negative
            delta = current_L - new_L
            prob = math.exp(-delta / T)
            if random.random() < prob:
                current_L = new_L
                current_residues[i] = new_r
                
        T *= alpha
        
    return best_residues
