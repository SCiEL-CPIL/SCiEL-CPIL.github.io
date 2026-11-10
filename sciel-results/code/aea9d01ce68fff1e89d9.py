import time
import sys

# EVOLVE-BLOCK-START
import math

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000

def construct():
    start_time = time.time()
    deadline = 60.0  # seconds
    
    # We search for the minimum window width W
    # Start from a small W and increase
    # A lower bound on W: 
    # The values must be distinct.
    # For small W, the number of multiples for each d_i in [L, L+W] is ~ W/d_i.
    # We need at least 1 multiple for each.
    # The tightest constraint is usually the largest divisor.
    # Let's just start from W=1.
    
    best_solution = None
    best_W = None
    
    # We will iterate W. 
    # If W is small, the search over L is the bottleneck.
    # However, for a fixed W, if a solution exists, it likely exists for many L.
    # We just need one.
    
    # To speed up, we can jump L? 
    # The condition for a multiple of d to be in [L, L+W] depends on L mod d.
    # But checking all L is straightforward and the inner check is fast.
    
    # Let's iterate W from 1 upwards.
    # For each W, iterate L from 1 to MAX_VAL - W.
    
    # Optimization: 
    # If W is very small, say W < min(DIVISORS), it's impossible for some divisors 
    # unless L is aligned perfectly.
    
    for W in range(1, 1000):  # Search up to W=1000
        if time.time() - start_time > deadline - 5:
            break
            
        # For this W, check possible L
        # L can range from 1 to MAX_VAL - W
        
        # We can optimize the L loop by noting that the pattern of available multiples
        # for a given divisor d repeats every d.
        # But with 10 divisors, the period is LCM(6,10,...,39) which is large.
        # So we just iterate L.
        
        # To make it faster, we can sample L or just iterate.
        # 10^6 iterations * 1000 W values is 10^9, too slow.
        # We need a smarter approach.
        
        # Smarter approach:
        # For a fixed W, the set of valid L is periodic with period P = LCM(DIVISORS).
        # But P is ~ 10^7 or more.
        # Actually, let's look at the density.
        # If W is large enough, most L will work.
        # If W is small, few L will work.
        
        # Let's try a different search order.
        # Instead of iterating all L for a fixed W, 
        # we can iterate L and check if W is feasible? No, W is the variable we minimize.
        
        # Let's stick to iterating W, but optimize the L check.
        # If W is small, the number of multiples is small.
        # We can precompute for each divisor d, the residues r such that 
        # there is a multiple of d in [L, L+W].
        # A multiple of d is in [L, L+W] iff there exists k such that L <= kd <= L+W.
        # This is equivalent to: ceil(L/d) <= floor((L+W)/d).
        
        # For a fixed W, define valid_residues[d] = set of L mod d such that 
        # a multiple of d exists in [L, L+W].
        # Then L is a candidate iff for all d, L mod d is in valid_residues[d].
        
        # This allows us to skip many L values?
        # Not easily without CRT.
        
        # Let's just iterate L but break early.
        # For W=1, L=1..10^6. Check is fast.
        # If no solution for W=1, try W=2.
        
        # Actually, we can estimate the minimum W.
        # The "density" argument suggests W ~ 50-100.
        # Let's start W from 50 to save time on impossible small W?
        # No, start from 1, it's fast.
        
        for L in range(1, MAX_VAL - W + 1):
            if time.time() - start_time > deadline - 5:
                return best_solution if best_solution else DIVISORS[:]
                
            # Check if a valid assignment exists in [L, L+W]
            # 1. Gather available multiples for each divisor
            avail = []
            valid = True
            for d in DIVISORS:
                # First multiple >= L
                first_k = (L + d - 1) // d
                first_m = first_k * d
                if first_m > L + W:
                    valid = False
                    break
                # List of multiples
                ms = []
                m = first_m
                while m <= L + W:
                    ms.append(m)
                    m += d
                avail.append(ms)
            
            if not valid:
                continue
                
            # 2. Backtracking to assign distinct values
            # Sort divisors by number of available multiples (ascending)
            # We need to keep track of original indices to return in correct order
            indices = list(range(len(DIVISORS)))
            indices.sort(key=lambda i: len(avail[i]))
            
            used = set()
            assignment = [None] * len(DIVISORS)
            
            def backtrack(idx):
                if idx == len(indices):
                    return True
                i = indices[idx]
                for m in avail[i]:
                    if m not in used:
                        used.add(m)
                        assignment[i] = m
                        if backtrack(idx + 1):
                            return True
                        used.remove(m)
                        assignment[i] = None
                return False
            
            if backtrack(0):
                return assignment
            
    # Fallback if not found (should not happen)
    return DIVISORS[:]
# EVOLVE-BLOCK-END
