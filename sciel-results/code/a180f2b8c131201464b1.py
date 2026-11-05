import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    def count_nonzero_square(exponents, coeffs):
        """
        Count nonzero terms in the square of the polynomial defined by exponents and coeffs.
        """
        n = len(exponents)
        if n == 0:
            return 0
        max_deg = max(exponents)
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

    def build_result(exponents, coeffs):
        """Build the final coefficient list in ascending degree."""
        max_deg = max(exponents)
        result = [0] * (max_deg + 1)
        for e, c in zip(exponents, coeffs):
            result[e] = c
        return result

    best_count = float('inf')
    best_exponents = list(range(12))
    best_coeffs = [1] * 12

    def update_best(exponents, coeffs, count):
        nonlocal best_count, best_exponents, best_coeffs
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
            if best_count == 12:
                return build_result(best_exponents, best_coeffs)

    # Phase 2: Structured Search using Composition P(x) = Q(x^d)
    # Strategy: If P(x) = Q(x^d), then P(x)^2 = Q(x^d)^2.
    # The exponents of P^2 are multiples of d.
    # Specifically, if Q has terms at degrees k_i, P has terms at d*k_i.
    # P^2 has terms at d*(k_i + k_j).
    # If we choose d large enough, all sums d*(k_i + k_j) might be distinct?
    # No, that's not right. The number of terms in Q^2 is fixed by Q.
    # However, if we pick Q to be sparse, Q^2 is sparse.
    # The key insight from the prompt: "search for divisors d >= 2 that force all cross-terms in P^2 to land on distinct degrees, thereby guaranteeing a square with exactly 12 non-zero terms (the squares of Q's coefficients) without relying on fragile cancellations."
    
    # Wait, if P(x) = Q(x^d), and Q has 12 terms, P has 12 terms.
    # P^2 = Q(x^d)^2. The coefficients of P^2 are the same as Q^2, just stretched by factor d.
    # So the number of nonzero terms in P^2 is exactly the number of nonzero terms in Q^2.
    # This doesn't reduce the count compared to Q. It just scales the exponents.
    
    # Let's re-read the strategy: "Model the 12-term polynomial P(x) as a composition Q(x^d) ... search for divisors d >= 2 that force all cross-terms in P^2 to land on distinct degrees, thereby guaranteeing a square with exactly 12 non-zero terms (the squares of Q's coefficients)"
    
    # This implies that if we can find a Q such that Q^2 has exactly 12 non-zero terms, then P(x)=Q(x^d) also has a square with exactly 12 non-zero terms.
    # But Q itself has 12 terms. If Q^2 has 12 terms, that's the minimum possible (since there are 12 diagonal terms c_i^2 which are always non-zero).
    # So the goal is to find a 12-term polynomial Q such that Q^2 has exactly 12 non-zero terms.
    # This happens if and only if all cross-terms cancel out.
    # i.e., for all i != j, the coefficient of x^(k_i + k_j) in Q^2 is 0.
    # Coefficient is 2*c_i*c_j. Since c_i, c_j are integers, this is non-zero unless c_i or c_j is 0.
    # But Q has exactly 12 non-zero terms. So c_i != 0 for all i.
    # Thus, cross-terms cannot cancel if they land on the same degree.
    # Therefore, for Q^2 to have only 12 terms, all cross-terms must land on degrees that are *not* in the set of diagonal degrees? No, that doesn't help, they would still be non-zero terms in the square.
    # Wait, "guaranteeing a square with exactly 12 non-zero terms (the squares of Q's coefficients)".
    # This means the cross-terms must cancel *each other* or cancel with diagonals?
    # Diagonals are at 2*k_i. Cross terms are at k_i + k_j.
    # If k_i + k_j = 2*k_m, then cross terms can cancel with diagonals? No, diagonals are positive squares. Cross terms can be negative.
    # But if all cross-terms cancel out to zero, then Q^2 has only the 12 diagonal terms.
    # This requires that for every pair (i,j) with i!=j, the sum of coefficients of x^(k_i+k_j) is 0.
    # Since each pair (i,j) and (j,i) contributes 2*c_i*c_j, and if there are multiple pairs summing to the same degree, they can cancel.
    
    # Let's try to find such a Q.
    # This is a hard combinatorial problem.
    # However, the prompt suggests this is a promising direction.
    
    # Let's implement a search for Q (dense, degree 11) or sparse Q (degree up to 40)
    # such that Q^2 has minimal terms.
    # We already did brute force for consecutive exponents.
    # Let's try random exponents and coefficients in {-1, 1, 2, -2} to see if we can get lucky with cancellations.
    
    # Phase 3: Random search with local improvement (Hill Climbing / Simulated Annealing)
    # We will explore the space of exponents and coefficients.
    # Coefficients in {-2, -1, 1, 2} initially.
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
    max_no_improve = 1000
    
    while time.time() < deadline:
        iterations += 1
        
        # Choose a random move
        move_type = random.random()
        
        if move_type < 0.5:
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
                        if best_count == 12:
                            return build_result(best_exponents, best_coeffs)
                else:
                    # Revert if not better (greedy hill climbing)
                    # Occasionally accept worse moves to escape local minima (Simulated Annealing style)
                    if random.random() < 0.1:
                        current_count = new_count
                        no_improve_count += 1
                    else:
                        current_coeffs[idx] = old_val
                        no_improve_count += 1
        elif move_type < 0.8:
            # Move an exponent
            idx = random.randint(0, 11)
            old_exp = current_exponents[idx]
            # Try small shifts
            shift = random.choice([-3, -2, -1, 1, 2, 3])
            new_exp = old_exp + shift
            if 0 <= new_exp <= 40 and new_exp not in current_exponents:
                current_exponents[idx] = new_exp
                # Sort exponents and reorder coefficients accordingly
                pairs = list(zip(current_exponents, current_coeffs))
                pairs.sort(key=lambda x: x[0])
                current_exponents, current_coeffs = zip(*pairs)
                current_exponents = list(current_exponents)
                current_coeffs = list(current_coeffs)
                
                new_count = count_nonzero_square(current_exponents, current_coeffs)
                if new_count < current_count:
                    current_count = new_count
                    no_improve_count = 0
                    if update_best(current_exponents, current_coeffs, new_count):
                        if best_count == 12:
                            return build_result(best_exponents, best_coeffs)
                else:
                    if random.random() < 0.1:
                        current_count = new_count
                        no_improve_count += 1
                    else:
                        # Revert
                        current_exponents[idx] = old_exp
                        pairs = list(zip(current_exponents, current_coeffs))
                        pairs.sort(key=lambda x: x[0])
                        current_exponents, current_coeffs = zip(*pairs)
                        current_exponents = list(current_exponents)
                        current_coeffs = list(current_coeffs)
                        no_improve_count += 1
        else:
            # Swap two exponents
            i, j = random.sample(range(12), 2)
            old_ei, old_ej = current_exponents[i], current_exponents[j]
            current_exponents[i], current_exponents[j] = old_ej, old_ei
            pairs = list(zip(current_exponents, current_coeffs))
            pairs.sort(key=lambda x: x[0])
            current_exponents, current_coeffs = zip(*pairs)
            current_exponents = list(current_exponents)
            current_coeffs = list(current_coeffs)
            
            new_count = count_nonzero_square(current_exponents, current_coeffs)
            if new_count < current_count:
                current_count = new_count
                no_improve_count = 0
                if update_best(current_exponents, current_coeffs, new_count):
                    if best_count == 12:
                        return build_result(best_exponents, best_coeffs)
            else:
                if random.random() < 0.1:
                    current_count = new_count
                    no_improve_count += 1
                else:
                    # Revert
                    current_exponents[i], current_exponents[j] = old_ei, old_ej
                    pairs = list(zip(current_exponents, current_coeffs))
                    pairs.sort(key=lambda x: x[0])
                    current_exponents, current_coeffs = zip(*pairs)
                    current_exponents = list(current_exponents)
                    current_coeffs = list(current_coeffs)
                    no_improve_count += 1
        
        if no_improve_count > max_no_improve:
            # Restart with random state
            no_improve_count = 0
            current_exponents = sorted(random.sample(range(41), 12))
            current_coeffs = [random.choice(coeff_values) for _ in range(12)]
            current_count = count_nonzero_square(current_exponents, current_coeffs)
            update_best(current_exponents, current_coeffs, current_count)

    return build_result(best_exponents, best_coeffs)
