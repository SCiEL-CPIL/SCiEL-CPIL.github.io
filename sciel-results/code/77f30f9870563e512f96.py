import numpy as np
import time
import random

def construct():
    base = 51975
    
    # Generate all divisors of base greater than 1
    divisors = []
    for i in range(2, base + 1):
        if base % i == 0:
            divisors.append(i)
    
    # Sort divisors. Order doesn't strictly matter for the final set, 
    # but for local search, we might want to process smaller moduli first 
    # or in random order.
    
    # Precompute indices for each modulus and residue
    # idx_cache[m] is a list of numpy arrays
    idx_cache = {}
    for m in divisors:
        idx_cache[m] = []
        for r in range(m):
            idx_cache[m].append(np.arange(r, base, m))
            
    # Initialize counts and score
    counts = np.zeros(base, dtype=np.int16)
    
    # Greedy Initialization
    # Start with all residues uncovered.
    # Assign residues to minimize uncovered count.
    # Greedy: for each m, pick r that covers most uncovered.
    
    assigned = {}
    
    # Sort moduli by size ascending (small m covers more, so it's more "important" to place well?)
    # Or just iterate.
    moduli_sorted = sorted(divisors)
    
    for m in moduli_sorted:
        best_r = 0
        max_covered = -1
        for r in range(m):
            idx = idx_cache[m][r]
            covered_now = np.sum(counts[idx] == 0)
            if covered_now > max_covered:
                max_covered = covered_now
                best_r = r
        assigned[m] = best_r
        idx = idx_cache[m][best_r]
        counts[idx] += 1
        
    current_score = np.sum(counts == 0)
    
    # Local Search
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    # We will perform steepest descent local search.
    # In each pass, we iterate over all moduli and try to improve the residue assignment.
    
    improved = True
    pass_count = 0
    no_improve_count = 0
    
    while time.time() < deadline:
        improved = False
        # Shuffle moduli to break symmetry
        moduli_list = divisors.copy()
        random.shuffle(moduli_list)
        
        for m in moduli_list:
            if time.time() > deadline:
                break
            
            current_r = assigned[m]
            
            # Calculate 'lost' for current_r
            # lost = number of residues in S_{m, current_r} that are covered ONLY by this class
            old_idx = idx_cache[m][current_r]
            lost = np.sum(counts[old_idx] == 1)
            
            best_r = current_r
            best_gain = -1 # We want to maximize gain (which reduces score)
            # Score change = lost - gain. We want to minimize this.
            # So we want to maximize gain.
            
            for r in range(m):
                if r == current_r:
                    continue
                new_idx = idx_cache[m][r]
                # gain = number of residues in S_{m, r} that are currently UNCOVERED
                gain = np.sum(counts[new_idx] == 0)
                
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            
            if best_r != current_r:
                # Apply move
                old_idx = idx_cache[m][current_r]
                new_idx = idx_cache[m][best_r]
                
                # Update counts
                counts[old_idx] -= 1
                counts[new_idx] += 1
                
                # Update score
                # new_score = current_score - best_gain + lost
                current_score = current_score - best_gain + lost
                assigned[m] = best_r
                improved = True
        
        if improved:
            no_improve_count = 0
        else:
            no_improve_count += 1
            # If stuck, try a random perturbation to escape local minimum
            if no_improve_count >= 3 and no_improve_count % 3 == 0:
                # Randomly change a few residues
                num_changes = min(5, len(divisors))
                for _ in range(num_changes):
                    m = random.choice(divisors)
                    old_r = assigned[m]
                    new_r = random.randint(0, m - 1)
                    if new_r != old_r:
                        old_idx = idx_cache[m][old_r]
                        new_idx = idx_cache[m][new_r]
                        
                        # Calculate score change
                        lost = np.sum(counts[old_idx] == 1)
                        gain = np.sum(counts[new_idx] == 0)
                        
                        counts[old_idx] -= 1
                        counts[new_idx] += 1
                        current_score = current_score - gain + lost
                        assigned[m] = new_r
                        
        pass_count += 1

    # Construct result
    classes = []
    for m in divisors:
        classes.append([m, assigned[m]])
        
    return classes
