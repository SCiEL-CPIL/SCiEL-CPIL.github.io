import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    base = 51975
    moduli = [d for d in range(2, base + 1) if base % d == 0]
    moduli_sorted = sorted(moduli)
    
    count_cover = np.zeros(base, dtype=np.int32)
    residues = {}
    
    # Greedy init - fast version
    for m in moduli_sorted:
        if m > 5000:
            r = 0
        else:
            # Try all residues for small moduli
            best_r = 0
            best_s = -1
            for r in range(m):
                s = np.sum(count_cover[r::m] == 0)
                if s > best_s:
                    best_s = s
                    best_r = r
            r = best_r
        residues[m] = r
        count_cover[r::m] += 1
    
    best_uncovered = int(np.sum(count_cover == 0))
    best_residues = residues.copy()
    
    # Local search - very fast version
    # Only try to improve moduli where it's likely to help
    # Focus on small moduli first since they cover more
    
    small_moduli = [m for m in moduli_sorted if m <= 200]
    med_moduli = [m for m in moduli_sorted if 200 < m <= 2000]
    large_moduli = [m for m in moduli_sorted if m > 2000]
    
    search_order = small_moduli + med_moduli + large_moduli
    
    for iteration in range(200):
        if time.time() > deadline - 1:
            break
        
        improved = False
        
        for m in search_order:
            if time.time() > deadline - 0.1:
                break
            
            r_old = residues[m]
            
            # Try a few candidate residues
            if m <= 100:
                candidates = range(m)
            elif m <= 1000:
                candidates = range(0, m, max(1, m // 20))
            else:
                candidates = [0, m // 2, m // 3, 2 * m // 3, r_old]
            
            best_r = r_old
            best_score = -1
            
            for r in candidates:
                if r == r_old:
                    continue
                # Score: how many uncovered positions does this residue cover?
                s = np.sum(count_cover[r::m] == 0)
                if s > best_score:
                    best_score = s
                    best_r = r
            
            if best_r != r_old and best_score > 0:
                # Check if this actually improves the global uncovered count
                # by temporarily applying and measuring
                count_cover[r_old::m] -= 1
                count_cover[best_r::m] += 1
                new_uncovered = int(np.sum(count_cover == 0))
                
                if new_uncovered < best_uncovered:
                    residues[m] = best_r
                    best_uncovered = new_uncovered
                    best_residues = residues.copy()
                    improved = True
                else:
                    # Revert
                    count_cover[best_r::m] -= 1
                    count_cover[r_old::m] += 1
        
        if not improved:
            break
    
    classes = [[m, best_residues[m]] for m in moduli]
    return classes
