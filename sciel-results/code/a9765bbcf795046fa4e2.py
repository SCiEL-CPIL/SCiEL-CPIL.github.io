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
    deadline = start_time + 290
    
    primes = _primes(N + 1)
    
    # Precompute bitmasks for all residues of all primes
    # masks[p][r] is an integer bitmask
    masks = {}
    for p in primes:
        masks[p] = []
        for r in range(p):
            m = 0
            # Set bits at r, r+p, r+2p, ...
            # Using a loop is fine for N=5000
            for i in range(r, N + 1, p):
                m |= (1 << i)
            masks[p].append(m)
            
    best_size = 0
    best_residues = {} # p -> r
    
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
                
            # Find best residue
            min_new = N + 1
            best_candidates = []
            
            for r in range(p):
                # Calculate new elements removed
                # new_elements = masks[p][r] & ~R_mask
                # count = new_elements.bit_count()
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
        best_residues = res
        
    # 2. Randomized Greedy with multiple seeds
    trials = 0
    while time.time() < deadline - 30: # Leave 30s for local search
        size, res = evaluate_greedy(randomize_ties=True)
        trials += 1
        if size > best_size:
            best_size = size
            best_residues = res
            # Optional: break if we hit a very good size?
            
    # 3. Local Search (Hill Climbing) on the best solution
    if best_residues and time.time() < deadline - 5:
        current_size = best_size
        current_residues = best_residues.copy()
        
        # Reconstruct R_mask for current solution
        R_mask = 0
        for p, r in current_residues.items():
            R_mask |= masks[p][r]
            
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try changing residue for each prime
            for p in list(current_residues.keys()):
                old_r = current_residues[p]
                old_mask = masks[p][old_r]
                
                # Try other residues
                for new_r in range(p):
                    if new_r == old_r:
                        continue
                    
                    new_mask = masks[p][new_r]
                    # new_R = (R_mask & ~old_mask) | new_mask
                    new_R = (R_mask & ~old_mask) | new_mask
                    new_size = N + 1 - new_R.bit_count()
                    
                    if new_size > current_size:
                        current_size = new_size
                        current_residues[p] = new_r
                        R_mask = new_R
                        improved = True
                        break # Restart loop with new state
                
    # Construct final list
    # Reconstruct R_mask for best_residues
    R_mask = 0
    for p, r in best_residues.items():
        R_mask |= masks[p][r]
        
    # Get indices where bit is 0
    result = []
    for i in range(N + 1):
        if not (R_mask >> i) & 1:
            result.append(i)
            
    return result
