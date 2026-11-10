"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    # Generate primes up to 1000
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    n_primes = len(primes)
    
    # Initial solution: use the seed approach
    residues = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    
    MAX_X = 5000  # Coverage array size
    
    def compute_coverage(res):
        """Compute coverage count for each x in 1..MAX_X."""
        coverage = [0] * (MAX_X + 1)
        for i, p in enumerate(primes):
            r = res[i]
            start = r if r >= 1 else p
            for x in range(start, MAX_X + 1, p):
                coverage[x] += 1
        return coverage
    
    def score_from_coverage(coverage):
        """Find largest L where all x in 1..L have coverage >= 2."""
        for x in range(1, MAX_X + 1):
            if coverage[x] < 2:
                return x - 1
        return MAX_X
    
    # Initial coverage and score
    coverage = compute_coverage(residues)
    best_score = score_from_coverage(coverage)
    best_residues = residues[:]
    
    # Simulated annealing
    current_residues = residues[:]
    current_score = best_score
    T = 1.0
    cooling = 0.9995
    min_T = 0.005
    
    while T > min_T:
        if time.time() > deadline:
            break
        
        # Pick a random prime to change
        i = random.randint(0, n_primes - 1)
        p = primes[i]
        old_r = current_residues[i]
        new_r = random.randint(0, p - 1)
        
        if new_r == old_r:
            continue
        
        # Update coverage incrementally
        start_old = old_r if old_r >= 1 else p
        start_new = new_r if new_r >= 1 else p
        
        for x in range(start_old, MAX_X + 1, p):
            coverage[x] -= 1
        for x in range(start_new, MAX_X + 1, p):
            coverage[x] += 1
        
        new_score = score_from_coverage(coverage)
        
        # Accept or reject
        if new_score >= current_score:
            current_residues[i] = new_r
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_residues = current_residues[:]
        else:
            delta = current_score - new_score
            if random.random() < math.exp(-delta / T):
                current_residues[i] = new_r
                current_score = new_score
            else:
                # Revert
                for x in range(start_new, MAX_X + 1, p):
                    coverage[x] -= 1
                for x in range(start_old, MAX_X + 1, p):
                    coverage[x] += 1
        
        T *= cooling
    
    return best_residues
# EVOLVE-BLOCK-END
