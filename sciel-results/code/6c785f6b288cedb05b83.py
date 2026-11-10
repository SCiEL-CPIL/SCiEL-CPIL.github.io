"""Improved program using exact greedy set cover for odd covering systems."""
import numpy as np

def construct():
    base = 51975
    
    # Generate all odd divisors of 51975 greater than 1
    # 51975 = 3^3 * 5^2 * 7 * 11
    divisors = set()
    for a in range(4):  # 3^a for a in 0..3
        for b in range(3):  # 5^b for b in 0..2
            for c in range(2):  # 7^c for c in 0..1
                for d in range(2):  # 11^d for d in 0..1
                    val = (3 ** a) * (5 ** b) * (7 ** c) * (11 ** d)
                    if val > 1:
                        divisors.add(val)
    
    # Sort divisors in increasing order
    sorted_divs = sorted(divisors)
    
    classes = []
    used_moduli = set()
    
    # Use numpy array for efficient coverage tracking
    uncovered = np.ones(base, dtype=np.uint8)
    
    num_moduli = len(sorted_divs)
    
    for _ in range(num_moduli):
        best_m = None
        best_r = None
        best_count = -1
        
        for m in sorted_divs:
            if m in used_moduli:
                continue
            
            # Find the exact best residue for this modulus
            # We iterate through all residues to ensure optimality for this step
            best_r_local = 0
            best_count_local = -1
            
            # Optimization: If m is large, the slice is small.
            # If m is small, the slice is large, but there are few residues.
            # We can just loop through all r.
            for r in range(m):
                count = np.sum(uncovered[r::m])
                if count > best_count_local:
                    best_count_local = count
                    best_r_local = r
            
            if best_count_local > best_count:
                best_count = best_count_local
                best_m = m
                best_r = best_r_local
        
        if best_m is None or best_count <= 0:
            break
        
        used_moduli.add(best_m)
        classes.append([best_m, best_r])
        
        # Update uncovered array
        uncovered[best_r::best_m] = 0
    
    return classes
