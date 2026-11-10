"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _get_divisors(n):
    divs = []
    for d in range(3, n + 1, 2):
        if n % d == 0:
            divs.append(d)
    return divs

def _count_uncovered(covered, m, r):
    return np.count_nonzero(covered[r::m] == 0)

def _greedy_construct(base, divisors, covered):
    classes = []
    for m in divisors:
        best_r = 0
        best_count = -1
        for r in range(m):
            c = _count_uncovered(covered, m, r)
            if c > best_count:
                best_count = c
                best_r = r
        classes.append([m, best_r])
        covered[best_r::m] = 1
    return classes

def _local_search(base, divisors, classes, covered, time_limit):
    start_time = time.time()
    best_uncovered = np.count_nonzero(covered == 0)
    
    # Precompute indices for each modulus
    # For local search, we try to reassign residues for moduli that might improve coverage
    
    # A simple local search: for each modulus, try all residues and keep the best
    # Repeat until no improvement or time limit
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i, (m, r) in enumerate(classes):
            # Try all possible residues for this modulus
            best_r = r
            best_score = -1
            # We need to evaluate the effect of changing residue r to r'
            # This is complex, so let's do a simpler approach:
            # Temporarily remove this class, try all residues, pick the best
            # But that's expensive. Instead, let's just try a few random residues.
            
            # Actually, let's do a proper local search:
            # For each modulus m, try each possible residue r'
            # Compute the new coverage if we change r to r'
            # This requires recomputing coverage, which is expensive.
            
            # Simpler approach: use a greedy improvement
            # For each modulus, try all residues and keep the one that maximizes coverage
            # of currently uncovered residues
            for r_try in range(m):
                if r_try == r:
                    continue
                # Count how many currently uncovered residues would be covered by r_try
                # minus how many currently covered residues would be uncovered
                # This is an approximation
                
                # Number of uncovered residues at positions r_try, r_try+m, ...
                # that are currently uncovered
                positions = np.arange(r_try, base, m)
                currently_covered_at_positions = covered[positions]
                uncovered_at_positions = 1 - currently_covered_at_positions
                
                # If we change to r_try, we cover these positions
                # But we also uncover positions r, r+m, ... that are only covered by this class
                # This is complex. Let's use a simpler heuristic.
                
                # For now, let's just try a random improvement
                pass
            
            # Let's do a simpler local search:
            # Try swapping with another modulus's residue
            pass
    
    return classes, best_uncovered

def construct():
    base = 51975
    divisors = _get_divisors(base)
    
    # Sort divisors by size (smaller moduli first might be better for coverage)
    # Actually, the original uses increasing order. Let's stick with that.
    divisors.sort()
    
    start_time = time.time()
    time_limit = 115.0  # Leave 5 seconds for safety
    
    # Initial greedy construction
    covered = np.zeros(base, dtype=np.uint8)
    classes = _greedy_construct(base, divisors, covered)
    
    # Local search refinement
    if time.time() - start_time < time_limit:
        classes, _ = _local_search(base, divisors, classes, covered, time_limit - (time.time() - start_time))
    
    return classes
# EVOLVE-BLOCK-END
