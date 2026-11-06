import time
import random
import sys

def _square_nonzero_count(exponents, coeffs):
    """Count nonzero terms in the square of sum coeffs[i]*x^exponents[i]."""
    n = len(exponents)
    # Use a dict to accumulate
    sq = {}
    for i in range(n):
        ci = coeffs[i]
        ei = exponents[i]
        # self term
        pos = 2 * ei
        sq[pos] = sq.get(pos, 0) + ci * ci
        for j in range(i + 1, n):
            cj = coeffs[j]
            ej = exponents[j]
            pos = ei + ej
            sq[pos] = sq.get(pos, 0) + 2 * ci * cj
    return sum(1 for v in sq.values() if v != 0)

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    best_count = 144  # worst case: 12*12=144 terms
    best_exponents = list(range(12))
    best_coeffs = [1] * 12
    
    # Initial random setup
    def random_candidate():
        # Choose exponents: start with a base set and perturb
        exponents = sorted(random.sample(range(0, 41), 12))
        coeffs = [random.choice([-1, 1]) for _ in range(12)]
        return exponents, coeffs
    
    exponents, coeffs = random_candidate()
    cur_count = _square_nonzero_count(exponents, coeffs)
    if cur_count < best_count:
        best_count = cur_count
        best_exponents = exponents[:]
        best_coeffs = coeffs[:]
    
    # Simulated annealing
    temperature = 10.0
    cooling = 0.995
    min_temp = 0.1
    
    iteration = 0
    while time.time() < deadline and temperature > min_temp:
        # Check deadline more frequently
        if iteration % 100 == 0 and time.time() > deadline:
            break
            
        # Generate a move
        move_type = random.random()
        
        new_exponents = exponents[:]
        new_coeffs = coeffs[:]
        
        if move_type < 0.4:
            # Change one exponent
            idx = random.randint(0, 11)
            new_exponents[idx] = random.randint(0, 40)
            new_exponents = sorted(new_exponents)
        elif move_type < 0.6:
            # Swap two exponents
            i, j = random.sample(range(12), 2)
            new_exponents[i], new_exponents[j] = new_exponents[j], new_exponents[i]
            new_exponents = sorted(new_exponents)
        elif move_type < 0.8:
            # Change one coefficient sign
            idx = random.randint(0, 11)
            new_coeffs[idx] = -new_coeffs[idx]
        elif move_type < 0.9:
            # Change coefficient magnitude
            idx = random.randint(0, 11)
            new_coeffs[idx] = random.choice([-2, -1, 1, 2])
        else:
            # Shift all exponents
            shift = random.randint(-3, 3)
            new_exponents = [e + shift for e in exponents]
            if min(new_exponents) < 0:
                new_exponents = [e - min(new_exponents) for e in new_exponents]
            if max(new_exponents) >= 41:
                new_exponents = [e - (max(new_exponents) - 40) for e in new_exponents]
            new_exponents = sorted(new_exponents)
        
        new_count = _square_nonzero_count(new_exponents, new_coeffs)
        
        delta = new_count - cur_count
        if delta <= 0 or random.random() < (2.718281828 ** (-delta / temperature)):
            exponents = new_exponents
            coeffs = new_coeffs
            cur_count = new_count
            if cur_count < best_count:
                best_count = cur_count
                best_exponents = exponents[:]
                best_coeffs = coeffs[:]
        
        temperature *= cooling
        iteration += 1
    
    # Final local search for any remaining time
    remaining = deadline - time.time()
    if remaining > 2:
        for _ in range(int(remaining * 50)):
            if time.time() > deadline - 1:
                break
            idx = random.randint(0, 11)
            new_coeffs = best_coeffs[:]
            new_coeffs[idx] = -new_coeffs[idx]
            c = _square_nonzero_count(best_exponents, new_coeffs)
            if c < best_count:
                best_count = c
                best_coeffs = new_coeffs
    
    # Build result
    result = [0] * (max(best_exponents) + 1)
    for e, c in zip(best_exponents, best_coeffs):
        result[e] = c
    # Ensure length <= 41
    if len(result) > 41:
        result = result[:41]
    # Ensure nonzero final coefficient
    while result and result[-1] == 0:
        result.pop()
    if not result:
        result = [1]
    # Ensure exactly 12 nonzero terms
    nonzero = sum(1 for x in result if x != 0)
    if nonzero != 12:
        # Fallback: use a known valid 12-term polynomial
        result = [0] * 41
        result[0] = 1
        result[1] = 1
        result[2] = 1
        result[3] = 1
        result[4] = 1
        result[5] = 1
        result[6] = 1
        result[7] = 1
        result[8] = 1
        result[9] = 1
        result[10] = 1
        result[11] = 1
        while result and result[-1] == 0:
            result.pop()
    
    return result
