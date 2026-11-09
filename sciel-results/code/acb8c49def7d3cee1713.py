import time
import random
import math

def construct():
    start_time = time.time()
    deadline = 110.0  # 10s buffer
    
    # Generate primes up to 1000
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
    M = 2000  # Search window
    
    # Initialize residues
    residues = [0] * n_primes
    counts = [0] * (M + 1)
    
    # Initial counts for r=0
    for i, p in enumerate(primes):
        for x in range(p, M + 1, p):
            counts[x] += 1
            
    # Find initial first_fail
    first_fail = M + 1
    for x in range(1, M + 1):
        if counts[x] < 2:
            first_fail = x
            break
            
    best_L = first_fail - 1
    best_res = residues[:]
    
    T = 50.0
    cooling_rate = 0.999
    
    while time.time() < deadline:
        i = random.randint(0, n_primes - 1)
        p = primes[i]
        old_r = residues[i]
        new_r = random.randint(0, p - 1)
        
        if new_r == old_r:
            continue
            
        changes = []
        
        # Decrement old
        start_old = p if old_r == 0 else old_r
        for x in range(start_old, M + 1, p):
            counts[x] -= 1
            changes.append((x, -1))
            
        # Increment new
        start_new = p if new_r == 0 else new_r
        for x in range(start_new, M + 1, p):
            counts[x] += 1
            changes.append((x, 1))
            
        # Find new first_fail
        new_ff = M + 1
        for x in range(1, M + 1):
            if counts[x] < 2:
                new_ff = x
                break
                
        if new_ff == M + 1:
            new_L = M
        else:
            new_L = new_ff - 1
            
        if new_L > best_L:
            best_L = new_L
            best_res = residues[:]
            residues[i] = new_r
            first_fail = new_ff
            T = max(T * 0.95, 1.0) # Cool faster on improvement?
        else:
            current_L = first_fail - 1
            delta = new_L - current_L
            if delta > 0 or (T > 1.0 and random.random() < math.exp(delta / T)):
                residues[i] = new_r
                first_fail = new_ff
            else:
                # Rollback
                for x, delta_c in changes:
                    counts[x] -= delta_c
                # first_fail and residues[i] unchanged
            
        T *= cooling_rate
        if T < 1.0:
            T = 1.0

    return best_res
