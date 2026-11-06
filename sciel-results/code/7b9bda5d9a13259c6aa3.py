import numpy as np
import time
import random

def count_nonzero(arr):
    return np.count_nonzero(arr)

def square_nonzero_count(coeffs):
    # coeffs is a list of 12 integers
    # We assume the support is 0..11
    # If we allow gaps, we need to handle indices.
    # For now, stick to 0..11 support.
    sq = np.convolve(coeffs, coeffs)
    return count_nonzero(sq)

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # 50s margin
    
    best_count = 23
    best_coeffs = [1]*12
    
    # Try alternating signs
    alt = [1 if i%2==0 else -1 for i in range(12)]
    c = square_nonzero_count(alt)
    if c < best_count:
        best_count = c
        best_coeffs = alt.copy()
        
    # Local search
    # We will try to improve best_coeffs
    # Also try random restarts with different initializations
    
    # We can also try to vary the support?
    # The task says "exactly 12 nonzero terms".
    # If we use support 0..11, we have 12 terms.
    # If we use support {0, 2, 4, ...}, we have 12 terms but span is larger.
    # Larger span -> more distinct sums -> harder to cancel?
    # Actually, if terms are far apart, their squares don't overlap, so no cancellation.
    # So we want overlap. Consecutive is best.
    
    # We will do a random walk on the coefficients.
    # Coefficients in [-64, 64].
    
    for restart in range(1000): # Max restarts, but limited by time
        if time.time() > deadline:
            break
            
        # Initialize coeffs
        # Try small integers first
        init_strategy = restart % 4
        if init_strategy == 0:
            coeffs = [1]*12
        elif init_strategy == 1:
            coeffs = [1 if i%2==0 else -1 for i in range(12)]
        elif init_strategy == 2:
            coeffs = [random.randint(-2, 2) for _ in range(12)]
            # Ensure 12 non-zeros
            for i in range(12):
                if coeffs[i] == 0:
                    coeffs[i] = 1
        else:
            coeffs = [random.randint(-64, 64) for _ in range(12)]
            for i in range(12):
                if coeffs[i] == 0:
                    coeffs[i] = 1
                    
        current_count = square_nonzero_count(coeffs)
        if current_count < best_count:
            best_count = current_count
            best_coeffs = coeffs.copy()
            
        # Local improvement
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try changing one coefficient
            # Shuffle indices to avoid bias
            indices = list(range(12))
            random.shuffle(indices)
            for idx in indices:
                # Try nearby values
                # If current is 1, try 2, 0, -1, etc.
                # But must remain non-zero?
                # The task says "exactly 12 nonzero terms".
                # If we set a coeff to 0, we have 11 terms. Invalid.
                # So we must keep all 12 non-zero.
                
                # Try values in a small range around current
                # Or random small values
                # To be safe, try -2, -1, 1, 2, 3, -3
                # If current is 1, don't try 1.
                
                candidates = []
                for delta in range(-3, 4):
                    new_val = coeffs[idx] + delta
                    if new_val != 0 and -64 <= new_val <= 64 and new_val != coeffs[idx]:
                        candidates.append(new_val)
                        
                random.shuffle(candidates)
                for new_val in candidates:
                    old_val = coeffs[idx]
                    coeffs[idx] = new_val
                    new_count = square_nonzero_count(coeffs)
                    if new_count < current_count:
                        current_count = new_count
                        if new_count < best_count:
                            best_count = new_count
                            best_coeffs = coeffs.copy()
                        improved = True
                        break
                    else:
                        coeffs[idx] = old_val
                        
        # If we found a very good one, we can break early?
        # No, keep searching for better.
        if best_count == 12: # Theoretical min?
            break

    # Return the best coeffs found
    # Ensure it meets the criteria
    # Length 12, all non-zero, in [-64, 64]
    # The function returns the coefficient list.
    # The evaluator expects a list of length <= 41.
    # If our support is 0..11, the list is length 12.
    # This is valid.
    
    # One check: "nonzero final coefficient".
    # Our list ends at index 11. If coeffs[11] != 0, it's fine.
    # We ensured all are non-zero.
    
    return best_coeffs
