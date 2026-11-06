import numpy as np
import time

def construct():
    base = 51975
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Find all divisors of base > 1
    divisors = []
    for d in range(2, base + 1):
        if base % d == 0:
            divisors.append(d)
    # Sort divisors (optional, but for deterministic behavior)
    divisors.sort()
    
    # Precompute mod_arrays: for each m, array of x % m for x in range(base)
    mod_arrays = {}
    for m in divisors:
        mod_arrays[m] = np.arange(base) % m
    
    # Initialize uncovered
    uncovered = np.ones(base, dtype=bool)
    used = set()
    classes = []
    
    # Greedy loop
    while time.time() < deadline:
        best_gain = 0
        best_m = None
        best_r = None
        for m in divisors:
            if m in used:
                continue
            # Compute histogram of residues for uncovered elements
            if not np.any(uncovered):
                break
            # Get the residues of uncovered elements modulo m
            residues = mod_arrays[m][uncovered]
            if len(residues) == 0:
                continue
            hist = np.bincount(residues, minlength=m)
            gain = np.max(hist)
            if gain > best_gain:
                best_gain = gain
                best_m = m
                best_r = int(np.argmax(hist))
        
        if best_m is None or best_gain == 0:
            break
        
        classes.append([best_m, best_r])
        used.add(best_m)
        # Update uncovered
        uncovered[mod_arrays[best_m] == best_r] = False
    
    return classes
