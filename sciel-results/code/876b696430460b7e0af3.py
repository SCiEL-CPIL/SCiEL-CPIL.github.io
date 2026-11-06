import numpy as np
import time
import random

N = 5000

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    primes = _primes(N + 1)
    
    # Precompute bitmasks for all residues of all primes
    # masks[p][r] is an integer bitmask
    masks = {}
    for p in primes:
        masks[p] = []
        for r in range(p):
            m = 0
            for i in range(r, N + 1, p):
                m |= (1 << i)
            masks[p].append(m)
            
    best_size = 0
    best_residues = {}
    
    def evaluate_greedy(randomize_ties=False, seed=None):
        if seed is not None:
            random.seed(seed)
        
        R_mask = 0
        current_size = N + 1
        residues = {}
        
        # Process primes in ascending order
        for p in primes:
            if p > current_size:
                break
                
            min_new = N + 1
            best_candidates = []
            
            for r in range(p):
                new_count = (masks[p][r] & ~R_mask).bit_count()
                
                if new_count < min_new:
                    min_new = new_count
                    best_candidates = [r]
                elif new_count == min_new:
                    best_candidates.append(r)
            
            if randomize_ties and best_candidates:
                r = random.choice(best_candidates)
            else:
                r = best_candidates[0] if best_candidates else 0
                
            residues[p] = r
            R_mask |= masks[p][r]
            current_size -= min_new
            
        return current_size, residues

    # 1. Deterministic Greedy
    size, res = evaluate_greedy(randomize_ties=False)
    if size > best_size:
        best_size = size
        best_residues = res.copy()
        
    # 2. Randomized Greedy with multiple seeds
    trials = 0
    while time.time() < deadline - 60: # Leave 60s for local search/SA
        size, res = evaluate_greedy(randomize_ties=True)
        trials += 1
        if size > best_size:
            best_size = size
            best_residues = res.copy()
            
    # 3. Local Search / Simulated Annealing on the best solution
    if best_residues and time.time() < deadline - 10:
        current_size = best_size
        current_residues = best_residues.copy()
        
        # Reconstruct R_mask
        R_mask = 0
        for p, r in current_residues.items():
            R_mask |= masks[p][r]
        
        # Simulated Annealing
        T = 10.0
        T_min = 0.01
        cooling = 0.999
        max_iter = 10000
        sa_iter = 0
        
        while T > T_min and time.time() < deadline - 10 and sa_iter < max_iter:
            sa_iter += 1
            # Pick a random prime in the current set
            if not current_residues:
                break
            p = random.choice(list(current_residues.keys()))
            old_r = current_residues[p]
            
            # Pick a new random residue
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue
                
            old_mask = masks[p][old_r]
            new_mask = masks[p][new_r]
            
            # Calculate new R_mask
            # R_mask = (R_mask & ~old_mask) | new_mask
            # But we need to be careful: elements covered by old_mask but not new_mask 
            # are "added back" (if not covered by other primes).
            # Elements covered by new_mask but not old_mask are "removed".
            
            # Efficient way:
            # Removed elements: new_mask & ~R_mask
            # Added elements: old_mask & R_mask & ~new_mask  (Wait, this is not quite right if other primes cover them)
            
            # Let's just compute the new R_mask and count bits.
            # This is O(1) for bit operations on Python ints.
            new_R_mask = (R_mask & ~old_mask) | new_mask
            new_size = N + 1 - new_R_mask.bit_count()
            
            delta = new_size - current_size
            
            if delta > 0 or random.random() < np.exp(delta / T):
                # Accept
                current_size = new_size
                current_residues[p] = new_r
                R_mask = new_R_mask
                
                if current_size > best_size:
                    best_size = current_size
                    best_residues = current_residues.copy()
            
            T *= cooling
            
    # Construct final list
    R_mask = 0
    for p, r in best_residues.items():
        R_mask |= masks[p][r]
        
    result = []
    for i in range(N + 1):
        if not (R_mask >> i) & 1:
            result.append(i)
            
    return result
