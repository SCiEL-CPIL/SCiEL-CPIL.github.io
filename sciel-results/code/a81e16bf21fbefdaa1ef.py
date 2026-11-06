import time
import random
import math

N = 51975

def get_divisors(n):
    divs = []
    for i in range(2, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    divs.append(n)
    return sorted(divs)

MODULI = get_divisors(N)
K = len(MODULI)

def initial_coverage(residues):
    counts = [0] * N
    score = N
    for m, r in zip(MODULI, residues):
        for x in range(r, N, m):
            if counts[x] == 0:
                score -= 1
            counts[x] += 1
    return counts, score

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    # Initial state
    current_residues = [random.randint(0, m - 1) for m in MODULI]
    counts, current_score = initial_coverage(current_residues)
    
    best_residues = current_residues[:]
    best_score = current_score
    best_counts = counts[:] # Store best state too? Or just residues? 
                            # We only need to return residues. 
                            # But to continue SA from best, we need counts.
                            # Actually, we can just keep the current state of SA.
                            # If we find a new best, we save residues.
                            # We don't need to revert to best state to continue, 
                            # SA usually works on the current state.
                            # However, if we get stuck, we might want to restart.
                            # Let's just stick to standard SA on current state.
                            
    T = 1.0
    cooling_rate = 0.9995
    min_T = 1e-5
    
    while time.time() - start_time < time_limit:
        i = random.randint(0, K - 1)
        m = MODULI[i]
        
        # Generate new residue
        # To make moves more effective, we can bias towards changing residues 
        # that are "critical" or just random. Random is fine.
        old_r = current_residues[i]
        new_r = random.randint(0, m - 1)
        
        if new_r == old_r:
            continue
            
        # Calculate delta
        # S_old: old_r, old_r+m, ...
        # S_new: new_r, new_r+m, ...
        
        # We need to iterate through the union of S_old and S_new
        # But they are disjoint if (new_r - old_r) % m != 0.
        # If (new_r - old_r) % m == 0, they are the same set (which is impossible if new_r != old_r and 0<=r<m).
        # So S_old and S_new are disjoint.
        
        delta = 0
        
        # Process S_old (removing coverage)
        for x in range(old_r, N, m):
            c = counts[x]
            if c == 1:
                delta += 1 # Becomes uncovered
            counts[x] = c - 1
            
        # Process S_new (adding coverage)
        for x in range(new_r, N, m):
            c = counts[x]
            if c == 0:
                delta -= 1 # Becomes covered
            counts[x] = c + 1
            
        new_score = current_score + delta
        
        # Acceptance
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_residues[i] = new_r
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_residues = current_residues[:]
        else:
            # Revert
            # Process S_old (adding back)
            for x in range(old_r, N, m):
                counts[x] += 1
            # Process S_new (removing back)
            for x in range(new_r, N, m):
                counts[x] -= 1
            current_residues[i] = old_r
            # current_score remains unchanged
            
        T *= cooling_rate
        if T < min_T:
            T = min_T
            
    classes = [[m, r] for m, r in zip(MODULI, best_residues)]
    return classes
