"""Rewritten program: Simulated Annealing for residue optimization."""
# EVOLVE-BLOCK-START
import math
import random
import time

def construct():
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
    L_max = 5000
    
    def find_L(coverage, L_max):
        for x in range(1, L_max + 1):
            if coverage[x] < 2:
                return x - 1
        return L_max

    # Phase 1: Greedy Initialization
    residues = [0] * n_primes
    coverage = [0] * (L_max + 1)
    
    for i, p in enumerate(primes):
        best_r = 0
        best_score = -1
        for r in range(p):
            score = 0
            x = r if r > 0 else p
            while x <= 500:
                if coverage[x] < 2:
                    score += 1
                x += p
            if score > best_score:
                best_score = score
                best_r = r
        residues[i] = best_r
        x = best_r if best_r > 0 else p
        while x <= L_max:
            coverage[x] += 1
            x += p

    current_L = find_L(coverage, L_max)
    best_L = current_L
    best_residues = residues[:]
    
    # Phase 2: Simulated Annealing
    start_time = time.time()
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.99
    
    # Helper to apply a move and return delta L
    def apply_move(i, r_new):
        p = primes[i]
        r_old = residues[i]
        if r_old == r_new:
            return 0, 0
        
        # Update coverage
        x = r_old
        if x == 0: x = p
        while x <= L_max:
            coverage[x] -= 1
            x += p
            
        x = r_new
        if x == 0: x = p
        while x <= L_max:
            coverage[x] += 1
            x += p
            
        residues[i] = r_new
        
        # Calculate new L
        new_L = find_L(coverage, L_max)
        delta = new_L - current_L
        return new_L, delta

    # Helper to revert a move
    def revert_move(i, r_old):
        p = primes[i]
        r_new = residues[i]
        if r_old == r_new:
            return
        
        # Update coverage back
        x = r_new
        if x == 0: x = p
        while x <= L_max:
            coverage[x] -= 1
            x += p
            
        x = r_old
        if x == 0: x = p
        while x <= L_max:
            coverage[x] += 1
            x += p
            
        residues[i] = r_old

    while time.time() - start_time < 115:
        if T < T_min:
            break
            
        # Pick a random prime
        i = random.randint(0, n_primes - 1)
        p = primes[i]
        r_old = residues[i]
        r_new = random.randint(0, p - 1)
        
        if r_old == r_new:
            continue
            
        # Apply move
        new_L, delta = apply_move(i, r_new)
        
        # Acceptance criterion
        accepted = False
        if delta >= 0:
            accepted = True
        else:
            # delta is negative
            # Probability = exp(delta / T)
            # Since delta is integer and negative, this is small
            if random.random() < math.exp(delta / T):
                accepted = True
        
        if accepted:
            current_L = new_L
            if current_L > best_L:
                best_L = current_L
                best_residues = residues[:]
        else:
            revert_move(i, r_old)
            # current_L remains unchanged
            
        T *= cooling_rate
        
    return best_residues
# EVOLVE-BLOCK-END
