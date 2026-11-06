import numpy as np
import time

def construct():
    base = 51975
    # Find all divisors of base greater than 1
    divisors = []
    for i in range(2, base + 1):
        if base % i == 0:
            divisors.append(i)
    
    # Sort divisors. 
    # Smaller moduli cover more residues per class (N/m is larger).
    # Greedy usually benefits from placing strong (small m) moduli first.
    # But local search doesn't care about order of addition, only final assignment.
    # We will use all divisors.
    
    # Precompute index arrays for each modulus and residue
    # idx_cache[m] is a list of numpy arrays, idx_cache[m][r] contains indices i in [0, base) s.t. i % m == r
    idx_cache = {}
    for m in divisors:
        idx_cache[m] = []
        # We can generate these efficiently
        # for r in range(m):
        #    indices = np.arange(r, base, m)
        # This is fine.
        for r in range(m):
            idx_cache[m].append(np.arange(r, base, m))
            
    # Initialize counts and score
    counts = np.zeros(base, dtype=np.int16)
    current_score = base  # Initially all uncovered
    
    # Greedy Initialization
    # Sort moduli by "power" (N/m) descending? i.e. small m first.
    # Small m has large N/m, so it covers more.
    # Let's sort by m ascending.
    moduli_sorted = sorted(divisors)
    
    assigned = {} # m -> r
    
    for m in moduli_sorted:
        best_r = 0
        best_score = base
        # Evaluate all residues for this m
        # We want to minimize the new score.
        # Current score is 'current_score'.
        # If we pick r, we cover indices idx_cache[m][r].
        # Some are already covered (counts > 0), some are not (counts == 0).
        # New covered = sum(counts[idx] == 0)
        # New score = current_score - new_covered
        # We want to maximize new_covered.
        
        # Optimization: 
        # max_covered = max( sum(counts[idx_cache[m][r]] == 0) for r in range(m) )
        
        max_covered = -1
        for r in range(m):
            idx = idx_cache[m][r]
            # Number of currently uncovered in this class
            covered_now = np.sum(counts[idx] == 0)
            if covered_now > max_covered:
                max_covered = covered_now
                best_r = r
        
        # Apply best_r
        assigned[m] = best_r
        idx = idx_cache[m][best_r]
        counts[idx] += 1
        current_score -= max_covered
        
    # Local Search (Hill Climbing)
    start_time = time.time()
    deadline = start_time + 140.0
    
    # We will iterate until no improvement or deadline.
    # To avoid getting stuck in a cycle, we only accept improvements.
    # We can randomize the order of moduli to escape shallow local minima? 
    # Hill climbing with fixed order might get stuck. Random order is better.
    
    # Create a list of moduli to iterate over
    moduli_list = list(divisors)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle moduli to break symmetry
        np.random.shuffle(moduli_list)
        
        for m in moduli_list:
            if time.time() > deadline:
                break
            
            current_r = assigned[m]
            best_r = current_r
            best_score = current_score
            
            # Try all other residues
            # To save time, we can skip if m is very large and N/m is 1?
            # If N/m == 1, changing r moves the single covered element.
            
            # We calculate the delta for each r
            # delta = (lost) - (gained)
            # lost = sum(counts[old_idx] == 1)
            # gained = sum(counts[new_idx] == 0)
            # Note: old_idx and new_idx are disjoint.
            
            old_idx = idx_cache[m][current_r]
            # Pre-calculate lost for current_r? 
            # lost is constant for a given m and current state, regardless of new_r.
            lost = np.sum(counts[old_idx] == 1)
            
            for r in range(m):
                if r == current_r:
                    continue
                
                new_idx = idx_cache[m][r]
                gained = np.sum(counts[new_idx] == 0)
                
                delta = lost - gained
                new_score = current_score + delta
                
                if new_score < best_score:
                    best_score = new_score
                    best_r = r
            
            if best_r != current_r:
                # Apply move
                old_idx = idx_cache[m][current_r]
                new_idx = idx_cache[m][best_r]
                
                # Update counts
                counts[old_idx] -= 1
                counts[new_idx] += 1
                
                # Update score
                current_score = best_score
                assigned[m] = best_r
                improved = True
                
    # Construct result
    classes = []
    for m in divisors:
        classes.append([m, assigned[m]])
        
    return classes
