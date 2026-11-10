"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

# EVOLVE-BLOCK-START
def construct():
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    n_primes = len(primes)
    M = 5000  # Max L to check
    
    # Initialize with seed
    residues = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    
    # Initialize counts
    counts = [0] * (M + 1)
    for i, p in enumerate(primes):
        r = residues[i]
        first = p if r == 0 else r
        for x in range(first, M + 1, p):
            counts[x] += 1
            
    def get_L(counts):
        for x in range(1, M + 1):
            if counts[x] < 2:
                return x - 1
        return M
        
    best_L = get_L(counts)
    best_residues = residues.copy()
    
    start_time = time.time()
    deadline = start_time + 110
    
    last_improvement_time = start_time
    
    while time.time() < deadline:
        # Pick a random prime to change
        idx = random.randint(0, n_primes - 1)
        p = primes[idx]
        old_r = residues[idx]
        
        # Pick a new residue
        new_r = random.randint(0, p - 1)
        
        if new_r == old_r:
            continue
            
        # Apply change
        residues[idx] = new_r
        
        # Update counts
        first_old = p if old_r == 0 else old_r
        for x in range(first_old, M + 1, p):
            counts[x] -= 1
            
        first_new = p if new_r == 0 else new_r
        for x in range(first_new, M + 1, p):
            counts[x] += 1
            
        new_L = get_L(counts)
        
        if new_L > best_L:
            best_L = new_L
            best_residues = residues.copy()
            last_improvement_time = time.time()
        else:
            # Revert
            residues[idx] = old_r
            for x in range(first_new, M + 1, p):
                counts[x] -= 1
            for x in range(first_old, M + 1, p):
                counts[x] += 1
                
        # Check for stagnation
        if time.time() - last_improvement_time > 5.0:
            # Reset to best and add noise
            residues = best_residues.copy()
            # Recalculate counts from best_residues to be safe
            counts = [0] * (M + 1)
            for i, p in enumerate(primes):
                r = residues[i]
                first = p if r == 0 else r
                for x in range(first, M + 1, p):
                    counts[x] += 1
            # Perturb a few random residues
            for _ in range(10):
                j = random.randint(0, n_primes - 1)
                p_j = primes[j]
                residues[j] = random.randint(0, p_j - 1)
                # Update counts for these perturbations
                # This is getting complex, let's just restart the loop with new random residues
                # if stuck.
                
            # Simpler restart:
            residues = [random.randint(0, p - 1) for p in primes]
            counts = [0] * (M + 1)
            for i, p in enumerate(primes):
                r = residues[i]
                first = p if r == 0 else r
                for x in range(first, M + 1, p):
                    counts[x] += 1
            last_improvement_time = time.time()
            
    return best_residues
# EVOLVE-BLOCK-END
