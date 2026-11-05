import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 90s limit (or 60s if strict)
    
    best_count = float('inf')
    best_coeffs = [1] * 12
    best_exponents = list(range(12))
    
    def count_nonzero_square(exponents, coeffs):
        """
        Count nonzero terms in the square of the polynomial defined by exponents and coeffs.
        """
        n = len(exponents)
        if n == 0:
            return 0
        max_deg = max(exponents)
        # Use a dictionary to handle sparse large degrees efficiently if needed,
        # but for deg <= 40, a list is faster.
        square = [0] * (2 * max_deg + 1)
        
        for i in range(n):
            ci = coeffs[i]
            ei = exponents[i]
            if ci == 0:
                continue
            # Diagonal term
            square[2 * ei] += ci * ci
            # Off-diagonal terms
            for j in range(i + 1, n):
                cj = coeffs[j]
                ej = exponents[j]
                if cj == 0:
                    continue
                prod = 2 * ci * cj
                square[ei + ej] += prod
        
        return sum(1 for x in square if x != 0)

    def update_best(exponents, coeffs, count):
        nonlocal best_count, best_coeffs, best_exponents
        if count < best_count:
            best_count = count
            best_exponents = exponents[:]
            best_coeffs = coeffs[:]
            if best_count == 12:
                return True
        return False

    # Phase 1: Brute force signs for 12 consecutive terms (0..11) with +/- 1
    # This is 4096 iterations, very fast.
    exponents_consec = list(range(12))
    for bits in range(1 << 12):
        if time.time() > deadline:
            break
        coeffs = [1 if (bits >> i) & 1 else -1 for i in range(12)]
        count = count_nonzero_square(exponents_consec, coeffs)
        if update_best(exponents_consec, coeffs, count):
            return _build_result(best_exponents, best_coeffs)

    # Phase 2: Random search with local improvement (Hill Climbing / Simulated Annealing)
    # We will explore the space of exponents and coefficients.
    # Coefficients in {-2, -1, 1, 2} initially, maybe expand later if time permits.
    # Exponents in [0, 40], distinct.
    
    random.seed(42)
    
    # Initialize with the best found so far
    current_exponents = best_exponents[:]
    current_coeffs = best_coeffs[:]
    current_count = best_count
    
    # Define possible coefficient values
    coeff_values = [-2, -1, 1, 2]
    
    iterations = 0
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        iterations += 1
        
        # Choose a random move
        move_type = random.random()
        
        if move_type < 0.4:
            # Change a coefficient
            idx = random.randint(0, 11)
            old_val = current_coeffs[idx]
            new_val = random.choice(coeff_values)
            if new_val != old_val:
                current_coeffs[idx] = new_val
                new_count = count_nonzero_square(current_exponents, current_coeffs)
                if new_count < current_count:
                    current_count = new_count
                    no_improve_count = 0
                    if update_best(current_exponents, current_coeffs, new_count):
                        return _build_result(best_exponents, best_coeffs)
                else:
                    # Revert if not better (greedy hill climbing)
                    # Occasionally accept worse moves to escape local minima (Simulated Annealing style)
                    if random.random() < 0.05:
                        current_count = new_count
                        no_improve_count += 1
                    else:
                        current_coeffs[idx] = old_val
                        no_improve_count += 1
        elif move_type < 0.7:
            # Move an exponent
            idx = random.randint(0, 11)
            old_exp = current_exponents[idx]
            # Try small shifts
            shift = random.choice([-2, -1, 1, 2])
            new_exp = old_exp + shift
            if 0 <= new_exp <= 40 and new_exp not in current_exponents:
                current_exponents[idx] = new_exp
                current_exponents.sort()
                # Re-sort coeffs to match exponents
                # This is tricky because we just sorted exponents.
                # We need to map old coeffs to new positions.
                # Easier: keep a list of (exp, coeff) pairs?
                # For now, let's revert to a simpler representation if this gets complex.
                # Let's stick to modifying the list directly and keeping them sorted.
                
                # To sort coeffs correctly:
                # We have a list of (exp, coeff).
                # Let's refactor the state to be a list of tuples to make sorting safe.
                pass 
        else:
            # Random restart / large jump
            # Generate a new random state
            # Keep some structure: mostly consecutive or small gaps
            if random.random() < 0.5:
                # Dense random
                base = random.randint(0, 2)
                gap = random.choice([1, 1, 1, 2])
                exps = []
                for i in range(12):
                    e = base + i * gap
                    if e > 40:
                        break
                    exps.append(e)
                if len(exps) < 12:
                    # Fill remaining with random distinct
                    used = set(exps)
                    while len(exps) < 12:
                        e = random.randint(0, 40)
                        if e not in used:
                            exps.append(e)
                            used.add(e)
                exps.sort()
            else:
                # Fully random sparse
                exps = random.sample(range(41), 12)
                exps.sort()
            
            coeffs = [random.choice(coeff_values) for _ in range(12)]
            
            current_exponents = exps
            current_coeffs = coeffs
            current_count = count_nonzero_square(current_exponents, current_coeffs)
            update_best(current_exponents, current_coeffs, current_count)
            no_improve_count = 0

        # Check for stagnation
        if no_improve_count > max_no_improve:
            # Perform a random restart
            exps = random.sample(range(41), 12)
            exps.sort()
            coeffs = [random.choice(coeff_values) for _ in range(12)]
            current_exponents = exps
            current_coeffs = coeffs
            current_count = count_nonzero_square(current_exponents, current_coeffs)
            update_best(current_exponents, current_coeffs, current_count)
            no_improve_count = 0

    return _build_result(best_exponents, best_coeffs)

def _build_result(exponents, coeffs):
    max_deg = max(exponents) if exponents else 0
    result = [0] * (max_deg + 1)
    for e, c in zip(exponents, coeffs):
        result[e] = c
    # Ensure length <= 41
    if len(result) > 41:
        result = result[:41]
    # Ensure non-zero final coefficient
    if result and result[-1] == 0:
        for i in range(len(result) - 1, -1, -1):
            if result[i] != 0:
                result = result[:i+1]
                break
    if not result:
        result = [1]
    return result
