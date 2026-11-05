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
    best_poly[0] = 1
    best_poly[1] = 1
    best_poly[2] = 1
    best_poly[3] = 1
    best_poly[4] = 1
    best_poly[5] = 1
    best_poly[6] = 1
    best_poly[7] = 1
    best_poly[8] = 1
    best_poly[9] = 1
    best_poly[10] = 1
    best_poly[11] = 1
    
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
        # Check constraints
        if len(poly) != 41:
            return 1000, False
        if any(c < -64 or c > 64 for c in poly):
            return 1000, False
        
        nz_count = sum(1 for c in poly if c != 0)
        if nz_count != 12:
            return 1000, False
        
        # Check nonzero final coefficient (degree <= 40, so last nonzero is <= 40)
        # The problem says "nonzero final coefficient". This usually means the coefficient
        # at the highest degree present is nonzero.
        # "length<=41" in the return value implies we strip trailing zeros.
        # But internally we use length 41.
        
        sq_nz = count_nonzero_square(poly)
        return sq_nz, True

    # Search Strategy:
    # 1. Try specific structured exponents (APs) with random coefficients.
    # 2. Local search from these best candidates.
    
    candidate_sets = []
    
    # Generate AP candidates
    for step in range(1, 6):
        for start in range(0, 10):
            exponents = [start + i * step for i in range(12)]
            if max(exponents) < 41:
                candidate_sets.append(exponents)
    
    # Generate random exponent sets
    for _ in range(50):
        exponents = sorted(random.sample(range(41), 12))
        candidate_sets.append(exponents)
    
    # Phase 1: Initial Population
    population = []
    for exponents in candidate_sets:
        if time.time() > deadline:
            break
        # Try a few coefficient patterns
        for _ in range(20):
            coeffs = [random.choice([-3, -2, -1, 1, 2, 3]) for _ in range(12)]
            poly = [0] * 41
            for i, e in enumerate(exponents):
                poly[e] = coeffs[i]
            
            score, valid = evaluate(poly)
            if valid:
                population.append((score, poly[:]))
    
    # Sort by score
    population.sort(key=lambda x: x[0])
    
    if population:
        best_score, best_poly = population[0]
        best_square_count = best_score
    
    # Phase 2: Local Search / Simulated Annealing
    # We will run multiple iterations, each starting from the current best or a random restart.
    
    max_iters = 20000
    current_temp = 10.0
    cooling = 0.995
    
    for restart in range(10):
        if time.time() > deadline:
            break
        
        # Start from best found so far, or a random valid candidate
        if restart == 0:
            current_poly = best_poly[:]
            current_score = best_square_count
        else:
            # Pick a random candidate from the initial population or generate new
            exponents = random.choice(candidate_sets)
            coeffs = [random.choice([-3, -2, -1, 1, 2, 3]) for _ in range(12)]
            current_poly = [0] * 41
            for i, e in enumerate(exponents):
                current_poly[e] = coeffs[i]
            current_score, valid = evaluate(current_poly)
            if not valid:
                continue
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
            
            # Create a neighbor
            new_poly = current_poly[:]
            nonzero_indices = [idx for idx, c in enumerate(new_poly) if c != 0]
            
            if not nonzero_indices:
                break
            
            move_type = random.random()
            
            if move_type < 0.3:
                # Change coefficient
                idx = random.choice(nonzero_indices)
                new_val = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
                if new_val != new_poly[idx]:
                    new_poly[idx] = new_val
            elif move_type < 0.6:
                # Move a term to a new exponent
                src = random.choice(nonzero_indices)
                zero_indices = [idx for idx, c in enumerate(new_poly) if c == 0]
                if zero_indices:
                    dst = random.choice(zero_indices)
                    new_poly[dst] = new_poly[src]
                    new_poly[src] = 0
            elif move_type < 0.8:
                # Swap exponents of two terms
                if len(nonzero_indices) >= 2:
                    i, j = random.sample(nonzero_indices, 2)
                    new_poly[i], new_poly[j] = new_poly[j], new_poly[i]
            else:
                # Flip sign
                idx = random.choice(nonzero_indices)
                new_poly[idx] = -new_poly[idx]
            
            # Evaluate
            new_score, valid = evaluate(new_poly)
            
            if valid:
                if new_score < current_score:
                    current_poly = new_poly
                    current_score = new_score
                    if current_score < best_square_count:
                        best_square_count = current_score
                        best_poly = current_poly[:]
                        if best_square_count <= 12:
                            return best_poly
                else:
                    # Simulated Annealing acceptance
                    delta = new_score - current_score
                    if random.random() < np.exp(-delta / current_temp):
                        current_poly = new_poly
                        current_score = new_score
            
            current_temp *= cooling
            if current_temp < 0.1:
                break
    
    # Final validation and formatting
    # Ensure exactly 12 nonzero terms
    nz_count = sum(1 for c in best_poly if c != 0)
    if nz_count != 12:
        # Fallback to a known valid simple polynomial
        best_poly = [0] * 41
        for i in range(12):
            best_poly[i] = 1
    
    # Trim trailing zeros
    result = best_poly[:]
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    
    if not result:
        result = [1]
        
    return result
# EVOLVE-BLOCK-END
