import random
import math
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
    L_target = 1000
    
    # Initialize residues randomly
    residues = [random.randint(0, p - 1) for p in primes]
    
    # Initialize coverage
    coverage = [0] * (L_target + 1)
    
    def update_coverage(p, r, delta):
        # delta is +1 or -1
        x = r if r > 0 else p
        while x <= L_target:
            coverage[x] += delta
            x += p

    # Apply initial residues
    for i, p in enumerate(primes):
        update_coverage(p, residues[i], 1)
        
    def get_score():
        s = 0
        for x in range(1, L_target + 1):
            s += min(coverage[x], 2)
        return s

    current_score = get_score()
    
    # Simulated Annealing / Local Search
    T = 5.0
    best_score = current_score
    best_residues = residues[:]
    
    start_time = time.time()
    max_time = 110 # Leave 10s buffer
    
    while time.time() - start_time < max_time:
        # Pick a random prime
        i = random.randint(0, n_primes - 1)
        p = primes[i]
        old_r = residues[i]
        
        # Pick a new random residue
        new_r = random.randint(0, p - 1)
        if new_r == old_r:
            continue
            
        # Calculate delta score
        # Identify affected positions
        # We can just iterate and track changes
        delta_score = 0
        
        # Positions to decrease
        # We need to know which positions are affected by old_r and new_r
        # To avoid double counting if sets overlap, we use a set or just careful logic.
        # Since p is prime and old_r != new_r, the sets {x | x % p == old_r} and {x | x % p == new_r} are disjoint.
        # So we can just sum the deltas independently.
        
        # Decrease old_r
        x = old_r if old_r > 0 else p
        while x <= L_target:
            old_c = coverage[x]
            new_c = old_c - 1
            delta_score += min(new_c, 2) - min(old_c, 2)
            x += p
            
        # Increase new_r
        x = new_r if new_r > 0 else p
        while x <= L_target:
            old_c = coverage[x]
            new_c = old_c + 1
            delta_score += min(new_c, 2) - min(old_c, 2)
            x += p
            
        # Acceptance criterion
        if delta_score > 0:
            accept = True
        else:
            # Accept with probability exp(delta/T)
            if T > 0.01:
                prob = math.exp(delta_score / T)
                accept = random.random() < prob
            else:
                accept = False
                
        if accept:
            # Apply move
            update_coverage(p, old_r, -1)
            update_coverage(p, new_r, 1)
            residues[i] = new_r
            current_score += delta_score
            
            if current_score > best_score:
                best_score = current_score
                best_residues = residues[:]
                
        # Cool down
        T *= 0.999
        if T < 0.1:
            T = 0.1 # Floor to allow some exploration
            
    # Final verification and return
    # We return best_residues
    return best_residues
