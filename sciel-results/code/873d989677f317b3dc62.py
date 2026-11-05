import numpy as np
import time

def construct():
    base = 51975
    # Generate divisors
    divisors = []
    for i in range(3, base + 1, 2):
        if base % i == 0:
            divisors.append(i)
    
    # Initialize residues greedily
    # Start with 0s
    cover_count = np.zeros(base, dtype=np.int8)
    residues = {m: 0 for m in divisors}
    
    # Greedy initialization
    # To make it faster, we can just pick 0 for all, then run local search.
    # Or do a simple greedy pass.
    # Let's do a simple greedy pass to get a decent start.
    covered = np.zeros(base, dtype=bool)
    for m in sorted(divisors):
        best_r = 0
        best_gain = -1
        for r in range(m):
            # Count uncovered in r::m
            # covered is boolean array
            gain = np.sum(covered[r::m] == False)
            if gain > best_gain:
                best_gain = gain
                best_r = r
        residues[m] = best_r
        covered[best_r::m] = True
    
    # Now switch to cover_count for local search
    cover_count = np.zeros(base, dtype=np.int8)
    for m in divisors:
        r = residues[m]
        cover_count[r::m] += 1
        
    # Local Search
    deadline = time.time() + 140.0
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for m in divisors:
            r_old = residues[m]
            
            # Remove m's contribution
            cover_count[r_old::m] -= 1
            
            # Find best new residue
            # We want to maximize the number of elements in r::m that are currently 0 in cover_count
            # i.e., newly covered elements
            best_r = r_old
            best_gain = -1
            
            # Optimization: If m is large, iterating all r is slow.
            # But sum of m is small.
            # We can iterate r from 0 to m-1
            for r in range(m):
                # Check gain
                # gain = number of zeros in cover_count[r::m]
                # This is O(N/m)
                gain = np.sum(cover_count[r::m] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            
            # If best_r is different, apply it
            if best_r != r_old:
                # Add m's contribution with new residue
                cover_count[best_r::m] += 1
                residues[m] = best_r
                improved = True
            else:
                # Re-add old contribution (it was removed)
                # Actually, if best_r == r_old, we just need to add it back
                cover_count[r_old::m] += 1
                
    # Return list of [m, r]
    return [[m, residues[m]] for m in divisors]
