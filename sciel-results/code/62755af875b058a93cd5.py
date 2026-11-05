"""Improved program for odd covering systems with moduli dividing 51975."""
import time
import numpy as np
import random
from collections import Counter

# EVOLVE-BLOCK-START
def construct():
    base = 51975
    deadline = time.time() + 145  # safety margin before 150s limit
    
    # Precompute all odd divisors of 51975 > 1, sorted in increasing order
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    n_div = len(divisors)
    
    # Precompute positions for each (modulus, residue) pair
    # For each modulus m, positions[r] = list of positions congruent to r mod m
    positions_cache = {}
    for m in divisors:
        positions_cache[m] = [np.arange(r, base, m, dtype=np.int32) for r in range(m)]
    
    def count_uncovered(cover_count):
        return int(np.sum(cover_count == 0))
    
    # Greedy initialization with multiple orderings
    best_residues = None
    best_uncovered = base
    best_cover_count = None
    
    orderings = [
        list(range(n_div)),
        list(range(n_div-1, -1, -1)),
        sorted(range(n_div), key=lambda i: divisors[i], reverse=True),
        sorted(range(n_div), key=lambda i: base // divisors[i]),
        sorted(range(n_div), key=lambda i: divisors[i] % 7),
    ]
    
    for order in orderings:
        if time.time() > deadline - 60:
            break
        
        residues = [0] * n_div
        cover_count = np.zeros(base, dtype=np.int16)
        
        for idx in order:
            if time.time() > deadline - 60:
                break
            m = divisors[idx]
            best_r = 0
            best_gain = -1
            for r in range(m):
                positions = positions_cache[m][r]
                gain = int(np.sum(cover_count[positions] == 0))
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            residues[idx] = best_r
            cover_count[best_r::m] += 1
        
        current_uncovered = count_uncovered(cover_count)
        if current_uncovered < best_uncovered:
            best_uncovered = current_uncovered
            best_residues = residues[:]
            best_cover_count = cover_count.copy()
    
    # Local search from the best initial solution
    residues = best_residues[:]
    cover_count = best_cover_count.copy()
    current_uncovered = best_uncovered
    
    # Simulated Annealing with efficient delta calculation
    temp = 80.0
    cooling = 0.998
    
    while time.time() < deadline - 15:
        if temp < 0.1:
            break
        
        i = random.randint(0, n_div - 1)
        mi = divisors[i]
        old_ri = residues[i]
        
        new_ri = random.randint(0, mi - 1)
        if new_ri == old_ri:
            continue
        
        old_positions = positions_cache[mi][old_ri]
        new_positions = positions_cache[mi][new_ri]
        
        old_vals = cover_count[old_positions]
        new_vals = cover_count[new_positions]
        
        delta = int(np.sum(old_vals == 1)) - int(np.sum(new_vals == 0))
        
        if delta < 0 or random.random() < np.exp(-delta / temp):
            cover_count[old_positions] -= 1
            cover_count[new_positions] += 1
            residues[i] = new_ri
            current_uncovered += delta
            if current_uncovered < best_uncovered:
                best_uncovered = current_uncovered
                best_residues = residues[:]
                best_cover_count = cover_count.copy()
        
        temp *= cooling
    
    # Final local search pass - multiple sweeps
    improved = True
    sweep = 0
    while improved and time.time() < deadline - 5:
        improved = False
        sweep += 1
        for i in range(n_div):
            if time.time() > deadline - 5:
                break
            m = divisors[i]
            old_r = residues[i]
            best_r = old_r
            best_local = current_uncovered
            
            for r in range(m):
                if r == old_r:
                    continue
                
                old_positions = positions_cache[m][old_r]
                new_positions = positions_cache[m][r]
                
                old_vals = cover_count[old_positions]
                new_vals = cover_count[new_positions]
                
                delta = int(np.sum(old_vals == 1)) - int(np.sum(new_vals == 0))
                
                if current_uncovered + delta < best_local:
                    best_local = current_uncovered + delta
                    best_r = r
            
            if best_r != old_r:
                old_positions = positions_cache[m][old_r]
                new_positions = positions_cache[m][best_r]
                
                cover_count[old_positions] -= 1
                cover_count[new_positions] += 1
                
                current_uncovered = best_local
                residues[i] = best_r
                improved = True
                
                if current_uncovered < best_uncovered:
                    best_uncovered = current_uncovered
                    best_residues = residues[:]
                    best_cover_count = cover_count.copy()
    
    # Additional refinement: try swapping pairs of residues for better coverage
    if time.time() < deadline - 5:
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(n_div):
                if time.time() > deadline - 5:
                    break
                for j in range(i+1, n_div):
                    if time.time() > deadline - 5:
                        break
                    mi = divisors[i]
                    mj = divisors[j]
                    ri = residues[i]
                    rj = residues[j]
                    
                    # Try swapping residues
                    if ri != rj:
                        old_pos_i = positions_cache[mi][ri]
                        old_pos_j = positions_cache[mj][rj]
                        new_pos_i = positions_cache[mi][rj % mi]
                        new_pos_j = positions_cache[mj][ri % mj]
                        
                        # Compute delta for simultaneous swap
                        # This is complex, so skip for now
                        pass
            
            # Simpler: just do another SA restart from best
            if best_uncovered > 0 and time.time() < deadline - 20:
                temp = 50.0
                cooling = 0.997
                while time.time() < deadline - 10:
                    if temp < 0.1:
                        break
                    i = random.randint(0, n_div - 1)
                    mi = divisors[i]
                    old_ri = residues[i]
                    new_ri = random.randint(0, mi - 1)
                    if new_ri == old_ri:
                        continue
                    
                    old_positions = positions_cache[mi][old_ri]
                    new_positions = positions_cache[mi][new_ri]
                    
                    old_vals = cover_count[old_positions]
                    new_vals = cover_count[new_positions]
                    
                    delta = int(np.sum(old_vals == 1)) - int(np.sum(new_vals == 0))
                    
                    if delta < 0 or random.random() < np.exp(-delta / temp):
                        cover_count[old_positions] -= 1
                        cover_count[new_positions] += 1
                        residues[i] = new_ri
                        current_uncovered += delta
                        if current_uncovered < best_uncovered:
                            best_uncovered = current_uncovered
                            best_residues = residues[:]
                            best_cover_count = cover_count.copy()
                    
                    temp *= cooling
    
    classes = [[divisors[i], best_residues[i]] for i in range(n_div)]
    return classes
# EVOLVE-BLOCK-END
