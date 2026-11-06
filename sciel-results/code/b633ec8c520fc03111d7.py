import time
import random
import numpy as np

def _count_nonzero(coeffs):
    return sum(1 for c in coeffs if c != 0)

def _square_nonzero_count(exponents, coefficients):
    """Compute number of nonzero terms in the square of a sparse polynomial."""
    n = len(exponents)
    # Build full coefficient array
    max_exp = max(exponents)
    coeffs = np.zeros(2 * max_exp + 1, dtype=np.int64)
    for e, c in zip(exponents, coefficients):
        coeffs[e] = c
    # Square via convolution
    sq = np.convolve(coeffs, coeffs)
    return int(np.count_nonzero(sq))

def _square_nonzero_count_fast(exponents, coefficients):
    """Faster: only compute the relevant sums."""
    n = len(exponents)
    # Dictionary approach: for each sum, accumulate
    sums = {}
    for i in range(n):
        for j in range(n):
            s = exponents[i] + exponents[j]
            sums[s] = sums.get(s, 0) + coefficients[i] * coefficients[j]
    return sum(1 for v in sums.values() if v != 0)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85 second safety margin
    
    random.seed(42)
    
    best_score = float('inf')
    best_exponents = list(range(12))
    best_coefficients = [1] * 12
    
    # Initial evaluation
    best_score = _square_nonzero_count_fast(best_exponents, best_coefficients)
    
    # Simulated annealing
    current_exponents = best_exponents[:]
    current_coefficients = best_coefficients[:]
    current_score = best_score
    
    temperature = 10.0
    cooling = 0.9995
    min_temp = 0.01
    
    coeff_choices = [-4, -3, -2, -1, 1, 2, 3, 4]
    
    while time.time() < deadline and temperature > min_temp:
        # Generate neighbor
        new_exponents = current_exponents[:]
        new_coefficients = current_coefficients[:]
        
        move_type = random.random()
        if move_type < 0.4:
            # Change one exponent
            idx = random.randint(0, 11)
            new_exponents[idx] = random.randint(0, 40)
        elif move_type < 0.7:
            # Change one coefficient
            idx = random.randint(0, 11)
            new_coefficients[idx] = random.choice(coeff_choices)
        elif move_type < 0.85:
            # Swap two exponents
            i, j = random.sample(range(12), 2)
            new_exponents[i], new_exponents[j] = new_exponents[j], new_exponents[i]
        else:
            # Change two coefficients
            i, j = random.sample(range(12), 2)
            new_coefficients[i] = random.choice(coeff_choices)
            new_coefficients[j] = random.choice(coeff_choices)
        
        # Validate: all exponents must be distinct, coefficients nonzero
        if len(set(new_exponents)) != 12:
            continue
        if any(c == 0 for c in new_coefficients):
            continue
        if any(c < -64 or c > 64 for c in new_coefficients):
            continue
        if any(e < 0 or e > 40 for e in new_exponents):
            continue
        
        new_score = _square_nonzero_count_fast(new_exponents, new_coefficients)
        
        # Acceptance
        delta = new_score - current_score
        if delta < 0 or random.random() < np.exp(-delta / temperature):
            current_exponents = new_exponents
            current_coefficients = new_coefficients
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_exponents = current_exponents[:]
                best_coefficients = current_coefficients[:]
        
        temperature *= cooling
    
    # Try some structured candidates
    structured = [
        # Consecutive with alternating signs
        (list(range(12)), [1 if i % 2 == 0 else -1 for i in range(12)]),
        # Consecutive with pattern 1,-1,1,-1...
        (list(range(12)), [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]),
        # Symmetric exponents
        ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        # Try exponents with gaps
        ([0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14], [1] * 12),
        # Try with more cancellation potential
        ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]),
    ]
    
    for exps, coefs in structured:
        if len(set(exps)) != 12 or any(c == 0 for c in coefs):
            continue
        score = _square_nonzero_count_fast(exps, coefs)
        if score < best_score:
            best_score = score
            best_exponents = exps[:]
            best_coefficients = coefs[:]
    
    # Build the result
    max_exp = max(best_exponents)
    result = [0] * (max_exp + 1)
    for e, c in zip(best_exponents, best_coefficients):
        result[e] = c
    
    # Trim trailing zeros
    while len(result) > 0 and result[-1] == 0:
        result.pop()
    
    # Final validation
    if len(result) > 41 or not result or result[-1] == 0:
        # Fallback
        result = [1] * 12
    if _count_nonzero(result) != 12:
        result = [1] * 12
    if any(c < -64 or c > 64 for c in result):
        result = [1] * 12
    
    return result
