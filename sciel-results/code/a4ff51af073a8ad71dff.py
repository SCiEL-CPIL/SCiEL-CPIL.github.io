"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import math

def count_nonzero(arr):
    return sum(1 for x in arr if x != 0)

def compute_square_nonzero(poly):
    if not poly:
        return 0
    square = np.convolve(poly, poly)
    return count_nonzero(square)

def construct():
    start_time = time.time()
    time_limit = 55.0
    
    # We represent the polynomial by a list of (exponent, coefficient) pairs
    # to make moves easier, but evaluate using a dense array for speed.
    
    best_score = float('inf')
    best_poly = None
    
    # Multiple restarts with different random seeds/structures
    num_restarts = 20
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize exponents
        # Strategy: Mix of arithmetic progression and random
        if restart % 2 == 0:
            # Arithmetic progression with random start and step
            step = random.randint(1, 3)
            start = random.randint(0, 40 - 11*step)
            exponents = [start + i * step for i in range(12)]
            # Ensure uniqueness and bounds (AP ensures uniqueness if step>0)
            if max(exponents) > 40:
                exponents = random.sample(range(0, 41), 12)
                exponents.sort()
        else:
            exponents = random.sample(range(0, 41), 12)
            exponents.sort()
        
        # Initialize coefficients: +/- 1
        coeffs = [random.choice([-1, 1]) for _ in range(12)]
        
        # Build initial dense poly
        max_deg = max(exponents)
        current_poly = [0] * (max_deg + 1)
        for e, c in zip(exponents, coeffs):
            current_poly[e] = c
            
        current_score = compute_square_nonzero(current_poly)
        
        # Simulated Annealing Parameters
        temperature = 10.0
        cooling_rate = 0.95
        min_temperature = 0.1
        
        while temperature > min_temperature and time.time() - start_time < time_limit:
            # Generate a neighbor
            new_exponents = exponents[:]
            new_coeffs = coeffs[:]
            
            move_type = random.random()
            
            if move_type < 0.4:
                # Change a coefficient
                idx = random.randint(0, 11)
                old_c = new_coeffs[idx]
                # Pick a new non-zero coefficient
                # Bias towards small values
                new_c = random.choice([-1, 1, -2, 2, -3, 3])
                if new_c == old_c:
                    new_c = -old_c
                new_coeffs[idx] = new_c
                
            elif move_type < 0.7:
                # Swap two exponents
                i, j = random.sample(range(12), 2)
                new_exponents[i], new_exponents[j] = new_exponents[j], new_exponents[i]
                
            elif move_type < 0.9:
                # Move one exponent to a new random position
                idx = random.randint(0, 11)
                old_e = new_exponents[idx]
                # Find a new valid exponent not currently used
                used = set(new_exponents)
                available = [e for e in range(41) if e not in used]
                if available:
                    new_e = random.choice(available)
                    new_exponents[idx] = new_e
                else:
                    # No move possible
                    continue
                    
            else:
                # Flip sign of a coefficient
                idx = random.randint(0, 11)
                new_coeffs[idx] = -new_coeffs[idx]
                
            # Evaluate new state
            # Rebuild dense poly
            # Optimization: Only update if exponents changed? 
            # For simplicity and given small size (41), rebuilding is fast enough.
            new_max_deg = max(new_exponents)
            new_poly = [0] * (new_max_deg + 1)
            for e, c in zip(new_exponents, new_coeffs):
                new_poly[e] = c
                
            new_score = compute_square_nonzero(new_poly)
            
            # Acceptance criterion
            delta = new_score - current_score
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                # Accept move
                exponents = new_exponents
                coeffs = new_coeffs
                current_poly = new_poly
                current_score = new_score
                
                if current_score < best_score:
                    best_score = current_score
                    # Trim trailing zeros for storage
                    temp_best = current_poly[:]
                    while len(temp_best) > 1 and temp_best[-1] == 0:
                        temp_best.pop()
                    best_poly = temp_best
                    
            temperature *= cooling_rate
            
    # Fallback
    if best_poly is None:
        best_poly = [1] * 12 + [0] * 29
        best_poly = best_poly[:41]
        
    # Final cleanup
    if best_poly[-1] == 0:
        while len(best_poly) > 1 and best_poly[-1] == 0:
            best_poly.pop()
        if best_poly[-1] == 0:
             best_poly[-1] = 1
             
    return best_poly
# EVOLVE-BLOCK-END
