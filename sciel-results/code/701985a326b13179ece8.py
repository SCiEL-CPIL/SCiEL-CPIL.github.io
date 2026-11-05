"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s limit
    
    best_square_count = 100
    best_poly = [0] * 41
    # Initialize with a valid fallback: 12 ones at the start
    for i in range(12):
        best_poly[i] = 1
        
    def count_nonzero_square(poly):
        # poly is a list of length 41
        p = np.array(poly, dtype=int)
        sq = np.convolve(p, p)
        return int(np.count_nonzero(sq))
    
    # Initial evaluation
    best_square_count = count_nonzero_square(best_poly)
    
    def evaluate(poly):
        """
        Returns (score, valid).
        Score is the number of nonzero terms in the square.
        Valid if constraints are met.
        """
        if len(poly) != 41:
            return 1000, False
        if any(c < -64 or c > 64 for c in poly):
            return 1000, False
        
        nz_count = sum(1 for c in poly if c != 0)
        if nz_count != 12:
            return 1000, False
        
        sq_nz = count_nonzero_square(poly)
        return sq_nz, True

    # Generate candidate exponent sets
    # Focus on Arithmetic Progressions (APs) as they maximize sum collisions
    candidate_exponent_sets = []
    
    # APs with step 1 to 3 are most dense
    for step in range(1, 4):
        max_start = 40 - 11 * step
        for start in range(0, max_start + 1):
            exponents = [start + i * step for i in range(12)]
            candidate_exponent_sets.append(exponents)
            
    # Add some random sets to break symmetry
    for _ in range(50):
        exponents = sorted(random.sample(range(41), 12))
        candidate_exponent_sets.append(exponents)
        
    # Shuffle to mix APs and randoms, but prioritize APs slightly by putting them first?
    # Actually, let's just iterate through them.
    
    # Local Search Parameters
    max_inner_iters = 2000
    initial_temp = 5.0
    cooling_rate = 0.99
    
    def local_search(exponents, initial_coeffs):
        """
        Perform local search on coefficients for a fixed set of exponents.
        Returns the best (score, poly) found.
        """
        current_poly = [0] * 41
        for i, e in enumerate(exponents):
            current_poly[e] = initial_coeffs[i]
            
        current_score, valid = evaluate(current_poly)
        if not valid:
            return 1000, [0]*41
            
        best_local_score = current_score
        best_local_poly = current_poly[:]
        
        temp = initial_temp
        no_improve_count = 0
        
        for i in range(max_inner_iters):
            if time.time() > deadline:
                break
                
            new_poly = current_poly[:]
            # Pick a random nonzero index to modify
            idx = random.choice(exponents)
            
            # Move types:
            # 1. Change value to random small int
            # 2. Change value to neighbor +/- 1
            # 3. Swap with another coefficient
            
            move_type = random.random()
            
            if move_type < 0.6:
                # Change value
                # Sample from a small range to encourage cancellation patterns
                new_val = random.randint(-5, 5)
                if new_val == 0:
                    # We must keep exactly 12 nonzero terms.
                    # If we set to 0, we must set another zero index to nonzero.
                    # But for simplicity in this step, let's just avoid 0 or handle swap.
                    # Let's stick to non-zero changes for now, or handle 0 carefully.
                    # If we change a nonzero to 0, we violate "exactly 12 nonzero" unless we add one elsewhere.
                    # Let's just pick a new nonzero value.
                    new_val = random.choice([-3, -2, -1, 1, 2, 3, 4, 5, -4, -5])
                new_poly[idx] = new_val
            elif move_type < 0.8:
                # Swap coefficients of two terms
                other_idx = random.choice(exponents)
                if other_idx != idx:
                    new_poly[idx], new_poly[other_idx] = new_poly[other_idx], new_poly[idx]
            else:
                # Flip sign
                new_poly[idx] = -new_poly[idx]
                
            new_score, new_valid = evaluate(new_poly)
            
            if new_valid:
                if new_score < current_score:
                    current_poly = new_poly
                    current_score = new_score
                    no_improve_count = 0
                    
                    if current_score < best_local_score:
                        best_local_score = current_score
                        best_local_poly = current_poly[:]
                else:
                    # Simulated Annealing acceptance
                    delta = new_score - current_score
                    if random.random() < np.exp(-delta / temp):
                        current_poly = new_poly
                        current_score = new_score
                        no_improve_count += 1
                    else:
                        no_improve_count += 1
                        
                temp *= cooling_rate
                if temp < 0.01:
                    temp = 0.01 # Keep it small but non-zero to allow some exploration
            
        return best_local_score, best_local_poly

    # Iterate through candidate exponent sets
    # Sort by density (step size) to find good candidates early?
    # APs with step 1 are most likely to have cancellations if coefficients are right.
    
    for exponents in candidate_exponent_sets:
        if time.time() > deadline:
            break
            
        # Try a few different initial coefficient patterns
        init_patterns = [
            [1]*12,
            [-1]*12,
            [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1], # Alternating
            [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1], # Blocks
            [random.randint(-3, 3) for _ in range(12)]
        ]
        
        # Ensure no zeros in init patterns for the search to start valid
        # If a pattern has 0, replace with 1
        for pattern in init_patterns:
            if time.time() > deadline:
                break
            valid_pattern = [p if p != 0 else 1 for p in pattern]
            
            score, poly = local_search(exponents, valid_pattern)
            
            if score < best_square_count:
                best_square_count = score
                best_poly = poly
                # If we find a very good one, we can break early?
                # Theoretical minimum for 12 terms is 23 (no cancellation).
                # With cancellation, it can be lower.
                # Let's keep searching to be sure.
                
    # Final validation
    nz_count = sum(1 for c in best_poly if c != 0)
    if nz_count != 12 or any(c < -64 or c > 64 for c in best_poly):
        # Fallback
        best_poly = [0] * 41
        for i in range(12):
            best_poly[i] = 1
        best_square_count = count_nonzero_square(best_poly)
        
    # Trim trailing zeros
    result = best_poly[:]
    while len(result) > 1 and result[-1] == 0:
        result.pop()
        
    if not result:
        result = [1]
        
    return result
# EVOLVE-BLOCK-END
