"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    base = 51975
    
    # Get all odd divisors of 51975 greater than 1
    divisors = []
    for d in range(3, base + 1, 2):
        if base % d == 0:
            divisors.append(d)
    
    # Sort by size (smaller moduli cover more residues)
    divisors.sort()
    
    # Track coverage
    covered = bytearray(base)
    classes = []
    used_moduli = set()
    
    # Phase 1: Greedy - process moduli in increasing order
    # For each modulus, pick the residue that covers the most uncovered residues
    for m in divisors:
        if time.time() > deadline - 5:
            break
        
        # Find best residue for this modulus
        best_r = 0
        best_count = -1
        
        # For small moduli, try all residues
        # For large moduli, sample or use heuristic
        if m <= 200:
            for r in range(m):
                # Count uncovered residues in this class
                count = 0
                for pos in range(r, base, m):
                    if not covered[pos]:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_r = r
        else:
            # For large moduli, sample residues
            # Try a few promising residues
            candidates = [0, 1, 2, m//2, m//3, 2*m//3]
            candidates = [c % m for c in candidates if c < m]
            for r in candidates:
                count = 0
                for pos in range(r, base, m):
                    if not covered[pos]:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_r = r
            # Also try a few random samples
            for _ in range(10):
                r = (base * 7919 + len(classes) * 104729) % m
                count = 0
                for pos in range(r, base, m):
                    if not covered[pos]:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_r = r
        
        classes.append([m, best_r])
        used_moduli.add(m)
        
        # Update coverage
        for pos in range(best_r, base, m):
            covered[pos] = 1
    
    # Phase 2: Local search - try to improve by swapping residues
    if time.time() < deadline - 10:
        # Try swapping residues for pairs of classes
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            
            # Recompute coverage from scratch for current classes
            covered_new = bytearray(base)
            for m, r in classes:
                for pos in range(r, base, m):
                    covered_new[pos] = 1
            
            uncovered_count = sum(1 for c in covered_new if not c)
            
            # Try improving individual classes
            for i, (m, r) in enumerate(classes):
                if time.time() > deadline - 10:
                    break
                
                # Temporarily remove this class
                covered_temp = bytearray(base)
                for j, (m2, r2) in enumerate(classes):
                    if j != i:
                        for pos in range(r2, base, m2):
                            covered_temp[pos] = 1
                
                # Find best residue for this modulus given others
                best_r2 = r
                best_uncovered = sum(1 for c in covered_temp if not c)
                
                if m <= 100:
                    for r2 in range(m):
                        # Check if adding this class reduces uncovered
                        covered_check = bytearray(covered_temp)
                        for pos in range(r2, base, m):
                            covered_check[pos] = 1
                        unc = sum(1 for c in covered_check if not c)
                        if unc < best_uncovered:
                            best_uncovered = unc
                            best_r2 = r2
                else:
                    # Sample for large moduli
                    for r2 in [0, 1, 2, m//2, m//3]:
                        r2 = r2 % m
                        covered_check = bytearray(covered_temp)
                        for pos in range(r2, base, m):
                            covered_check[pos] = 1
                        unc = sum(1 for c in covered_check if not c)
                        if unc < best_uncovered:
                            best_uncovered = unc
                            best_r2 = r2
                
                if best_r2 != r:
                    classes[i] = [m, best_r2]
                    improved = True
    
    # Final coverage calculation
    covered_final = bytearray(base)
    for m, r in classes:
        for pos in range(r, base, m):
            covered_final[pos] = 1
    
    return classes
# EVOLVE-BLOCK-END
