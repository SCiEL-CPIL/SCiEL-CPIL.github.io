"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    base = 51975
    
    # Generate all divisors of base greater than 1
    divisors = []
    for d in range(2, base + 1):
        if base % d == 0:
            divisors.append(d)
            
    # Sort divisors for deterministic processing, though order doesn't strictly matter for global greedy
    divisors.sort()
    
    cover_count = np.zeros(base, dtype=np.int32)
    selected_classes = []
    used_moduli = set()
    
    # Precompute indices for each divisor to speed up access?
    # indices[m] = list of arrays for each r? Or just calculate on fly.
    # Calculating on fly is fine for 48 moduli.
    
    # Global Greedy Phase
    while True:
        best_gain = 0
        best_m = -1
        best_r = -1
        
        for m in divisors:
            if m in used_moduli:
                continue
                
            # Find best residue r for this m
            # We want to maximize the number of NEWLY covered residues
            # Newly covered = cover_count[indices] == 0
            
            # Optimization: If the total possible coverage of m (which is base//m) is <= best_gain, 
            # we might still want to check if it hits uncovered ones.
            # But base//m is the max possible gain.
            if base // m <= best_gain:
                continue
                
            for r in range(m):
                indices = np.arange(r, base, m)
                gain = np.sum(cover_count[indices] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_m = m
                    best_r = r
        
        if best_m == -1:
            break
            
        selected_classes.append([best_m, best_r])
        used_moduli.add(best_m)
        indices = np.arange(best_r, base, best_m)
        cover_count[indices] += 1
        
    # Local Search Phase
    # We optimize residues for the selected moduli.
    # We try to change r for a specific m to improve total coverage.
    
    max_iterations = 50
    for _ in range(max_iterations):
        improved = False
        current_covered = np.sum(cover_count > 0)
        
        # Shuffle to avoid bias? Or just iterate.
        # Let's iterate.
        for i in range(len(selected_classes)):
            m, r = selected_classes[i]
            
            # Remove current class
            indices_old = np.arange(r, base, m)
            cover_count[indices_old] -= 1
            
            best_r_new = r
            best_covered_new = current_covered - np.sum(cover_count[indices_old] == 0) # This logic is slightly off, let's just recalc
            
            # Actually, easier:
            # Temporarily removed.
            # Try all r'
            for r_new in range(m):
                if r_new == r:
                    continue
                indices_new = np.arange(r_new, base, m)
                cover_count[indices_new] += 1
                new_covered = np.sum(cover_count > 0)
                
                if new_covered > best_covered_new:
                    best_covered_new = new_covered
                    best_r_new = r_new
                else:
                    cover_count[indices_new] -= 1 # Revert this specific test
            
            # Apply best_r_new
            if best_r_new != r:
                indices_best = np.arange(best_r_new, base, m)
                cover_count[indices_best] += 1
                selected_classes[i] = [m, best_r_new]
                improved = True
                current_covered = best_covered_new
            else:
                # Revert to original
                cover_count[indices_old] += 1
                # current_covered remains same
                
        if not improved:
            break
            
    return selected_classes
# EVOLVE-BLOCK-END
