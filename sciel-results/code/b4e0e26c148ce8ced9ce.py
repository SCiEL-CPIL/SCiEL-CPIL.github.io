import time
import random

def _count_nonzero_square(coeffs, exponents):
    """Count nonzero terms in the exact integer square of a sparse polynomial."""
    n = len(coeffs)
    max_deg = exponents[-1]
    square = [0] * (2 * max_deg + 1)
    for i in range(n):
        ci = coeffs[i]
        if ci == 0:
            continue
        ei = exponents[i]
        for j in range(i, n):
            cj = coeffs[j]
            if cj == 0:
                continue
            ej = exponents[j]
            s = ci * cj
            square[ei + ej] += s
            if i != j:
                square[ei + ej] += s
    return sum(1 for x in square if x != 0)

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n (as +1/-1)."""
    seq = [1]
    while len(seq) < n:
        seq += [-x for x in seq]
    return seq[:n]

def construct():
    start_time = time.time()
    timeout = 55.0
    
    best_count = float('inf')
    best_coeffs = [1]*12
    best_exponents = list(range(12))
    
    # Phase 1: Try structured sequences with consecutive exponents
    # Thue-Morse and its variants
    structured_patterns = []
    
    # Thue-Morse
    tm = _thue_morse(12)
    structured_patterns.append(tm)
    
    # Reversed Thue-Morse
    structured_patterns.append(tm[::-1])
    
    # All +1, all -1
    structured_patterns.append([1]*12)
    structured_patterns.append([-1]*12)
    
    # Alternating
    structured_patterns.append([1 if i%2==0 else -1 for i in range(12)])
    structured_patterns.append([-1 if i%2==0 else 1 for i in range(12)])
    
    # Brute force all sign patterns for 12 consecutive terms
    for bits in range(1 << 12):
        if time.time() - start_time > timeout:
            break
        coeffs = [1 if (bits >> i) & 1 else -1 for i in range(12)]
        count = _count_nonzero_square(coeffs, list(range(12)))
        if count < best_count:
            best_count = count
            best_coeffs = coeffs
            best_exponents = list(range(12))
    
    # Phase 2: Local search with small coefficient changes
    # Start from best found, try flipping signs or changing magnitudes
    random.seed(42)
    current_coeffs = best_coeffs[:]
    current_exponents = best_exponents[:]
    current_count = best_count
    
    allowed_coeffs = [-2, -1, 1, 2]
    
    iterations = 0
    while time.time() - start_time < timeout:
        iterations += 1
        
        # Random perturbation
        if random.random() < 0.7:
            # Flip a sign or change coefficient
            idx = random.randint(0, 11)
            new_val = random.choice(allowed_coeffs)
            if new_val != current_coeffs[idx]:
                old_val = current_coeffs[idx]
                current_coeffs[idx] = new_val
                new_count = _count_nonzero_square(current_coeffs, current_exponents)
                if new_count < current_count:
                    current_count = new_count
                    if new_count < best_count:
                        best_count = new_count
                        best_coeffs = current_coeffs[:]
                        best_exponents = current_exponents[:]
                        if best_count == 12:
                            break
                else:
                    # Revert if worse (greedy)
                    current_coeffs[idx] = old_val
        else:
            # Try moving an exponent
            idx = random.randint(0, 11)
            old_exp = current_exponents[idx]
            new_exp = random.randint(0, 40)
            # Ensure no duplicate exponents
            if new_exp not in current_exponents:
                current_exponents[idx] = new_exp
                current_exponents.sort()
                # Need to reorder coeffs to match
                old_coeffs = current_coeffs[:]
                new_coeffs = [0]*12
                for i, e in enumerate(current_exponents):
                    if e == old_exp:
                        new_coeffs[i] = old_coeffs[idx]
                    else:
                        # Find original coeff for this exponent
                        for j, oe in enumerate(sorted(current_exponents)):
                            pass
                # Simpler: just revert
                current_exponents[idx] = old_exp
                current_exponents.sort()
        
        if best_count == 12:
            break
    
    # Phase 3: Random restarts with local search
    remaining_time = timeout - (time.time() - start_time)
    if remaining_time > 5:
        for restart in range(50):
            if time.time() - start_time > timeout - 2:
                break
            
            # Random initial state
            exponents = random.sample(range(41), 12)
            exponents.sort()
            coeffs = [random.choice(allowed_coeffs) for _ in range(12)]
            
            count = _count_nonzero_square(coeffs, exponents)
            
            # Local search
            local_iters = 0
            while time.time() - start_time < timeout - 2 and local_iters < 200:
                local_iters += 1
                idx = random.randint(0, 11)
                if random.random() < 0.5:
                    new_val = random.choice(allowed_coeffs)
                    if new_val != coeffs[idx]:
                        old_val = coeffs[idx]
                        coeffs[idx] = new_val
                        new_count = _count_nonzero_square(coeffs, exponents)
                        if new_count < count:
                            count = new_count
                            if count < best_count:
                                best_count = count
                                best_coeffs = coeffs[:]
                                best_exponents = exponents[:]
                                if best_count == 12:
                                    return best_coeffs
                        else:
                            coeffs[idx] = old_val
                else:
                    old_exp = exponents[idx]
                    new_exp = random.randint(0, 40)
                    if new_exp not in exponents:
                        exponents[idx] = new_exp
                        exponents.sort()
                        # Rebuild coeffs in order
                        old_coeffs = coeffs[:]
                        coeffs = [0]*12
                        for i, e in enumerate(exponents):
                            if e == old_exp:
                                coeffs[i] = old_coeffs[idx]
                            else:
                                # Find which old exponent maps to this new one
                                # This is getting complex; just track differently
                                pass
                        # Simpler approach: revert
                        exponents[idx] = old_exp
                        exponents.sort()
    
    # Build final result
    if best_exponents is not None and len(best_exponents) > 0:
        max_deg = max(best_exponents)
        result = [0] * (max_deg + 1)
        for e, c in zip(best_exponents, best_coeffs):
            result[e] = c
        return result
    else:
        return best_coeffs
