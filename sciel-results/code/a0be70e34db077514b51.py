import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Initialize exponents as an arithmetic progression 0..11
    # This maximizes the number of pairs (i,j) with i+j=k, enabling cancellations
    exponents = list(range(12))
    
    # Initial coefficients: all 1s
    # We need exactly 12 non-zero terms
    coeffs = [1] * 12
    
    def count_nonzero_square(c):
        # Compute square coefficients
        # Max degree is 22
        square_coeffs = [0] * 23
        for i in range(12):
            ci = c[i]
            if ci == 0:
                continue
            for j in range(12):
                cj = c[j]
                if cj == 0:
                    continue
                k = i + j
                square_coeffs[k] += ci * cj
        
        return sum(1 for x in square_coeffs if x != 0)

    current_score = count_nonzero_square(coeffs)
    best_score = current_score
    best_coeffs = coeffs.copy()
    
    # Candidate values for coefficients
    # Focus on small integers first, as they are more likely to cancel
    # e.g., 1 + (-1) = 0, 2 + (-2) = 0, 1*1 + (-1)*1 = 0
    candidates = list(range(-64, 65))
    # Prioritize small values in selection
    small_candidates = list(range(-10, 11))
    
    no_improve_count = 0
    
    while time.time() < deadline:
        # Pick a random index to modify
        idx = random.randint(0, 11)
        
        # Pick a new value
        # 80% chance pick from small range, 20% from full range
        if random.random() < 0.8:
            new_val = random.choice(small_candidates)
        else:
            new_val = random.choice(candidates)
            
        if new_val == 0:
            continue
            
        if new_val == coeffs[idx]:
            continue
            
        old_val = coeffs[idx]
        coeffs[idx] = new_val
        
        new_score = count_nonzero_square(coeffs)
        
        if new_score < current_score:
            current_score = new_score
            no_improve_count = 0
            if new_score < best_score:
                best_score = new_score
                best_coeffs = coeffs.copy()
        elif new_score == current_score:
            # Accept with some probability to escape local minima
            if random.random() < 0.5:
                no_improve_count += 1
            else:
                coeffs[idx] = old_val
        else:
            # Reject
            coeffs[idx] = old_val
            no_improve_count += 1
        
        # If stuck for a long time, restart with random coefficients
        if no_improve_count > 1000:
            coeffs = [random.choice([-1, 1, -2, 2]) for _ in range(12)]
            # Ensure non-zero
            for i in range(12):
                if coeffs[i] == 0:
                    coeffs[i] = 1
            current_score = count_nonzero_square(coeffs)
            no_improve_count = 0
            
    # Construct the final polynomial list
    # The task requires: "Return coefficient list in ascending degree, length<=41, nonzero final coefficient, entries integers in [-64,64], exactly 12 nonzero terms."
    # Our best_coeffs has 12 non-zero terms.
    # We need to map them to the correct degrees based on exponents.
    # Exponents are 0..11.
    # So the list length is 12.
    # Wait, "length<=41". If exponents are 0..11, length is 12.
    # Is it better to have a larger degree?
    # "Minimize nonzero terms in the EXACT integer square."
    # The square of a degree 11 poly has degree 22.
    # If we used exponents 0, 2, 4... 22, degree 22, square degree 44 > 40? No, square degree is 2*max_exp.
    # If max_exp = 20, square degree 40.
    # But AP 0..11 is denser.
    # Let's stick with the best found.
    
    # Ensure final coefficient is non-zero (it is, by construction)
    # Ensure entries in [-64, 64] (they are)
    # Ensure exactly 12 non-zero terms (they are)
    
    return best_coeffs

