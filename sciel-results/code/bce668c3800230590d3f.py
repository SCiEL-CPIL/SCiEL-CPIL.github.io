"""
Strategy: Structured Product Construction + Targeted Local Search

Instead of starting from random 12-term polynomials, we exploit the algebraic
structure P = Q * R where Q and R are 6-term polynomials. The square P^2 = Q^2 * R^2
has a natural block structure. By carefully choosing exponents and coefficients
of Q and R, we can force many cross-term cancellations in P^2.

Key insight: If Q and R have "separated" exponent supports, the square P^2 will
have a predictable structure where cancellations can be engineered.
"""
import time
import random
import numpy as np
from itertools import product

def _square_count(coeffs_list):
    """Calculate the number of non-zero terms in the square."""
    if not coeffs_list or len(coeffs_list) == 0:
        return 0
    sq = np.convolve(coeffs_list, coeffs_list)
    return int(np.count_nonzero(sq))

def _poly_mul(p1, p2):
    """Multiply two polynomials represented as lists."""
    if not p1 or not p2:
        return [0]
    result = [0] * (len(p1) + len(p2) - 1)
    for i, c1 in enumerate(p1):
        if c1 == 0:
            continue
        for j, c2 in enumerate(p2):
            if c2 == 0:
                continue
            result[i + j] += c1 * c2
    # Trim trailing zeros but keep at least one element
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result

def _pad_to_length(poly, max_len=41):
    """Pad polynomial to max_len if needed, or truncate if too long."""
    if len(poly) > max_len:
        return poly[:max_len]
    if len(poly) < max_len:
        return poly + [0] * (max_len - len(poly))
    return poly

def _generate_6term_polynomial(start_exp, step, coeff_range=(-2, 2)):
    """Generate a random 6-term polynomial with given start and step."""
    signs = [random.choice([-1, 1]) for _ in range(6)]
    coeffs = [random.choice([1, 2]) for _ in range(6)]
    poly = [0] * 41
    for i in range(6):
        exp = start_exp + step * i
        if 0 <= exp < 41:
            poly[exp] = signs[i] * coeffs[i]
    return poly

def _generate_product_candidate():
    """
    Generate a candidate P = Q * R where Q and R are 6-term polynomials.
    Use separated exponent supports to create structure for cancellations.
    """
    # Choose structure for Q
    q_start = random.randint(0, 5)
    q_step = random.choice([1, 2, 3])
    Q = _generate_6term_polynomial(q_start, q_step)
    
    # Choose structure for R, offset from Q to create separation
    r_start = q_start + random.randint(10, 20)  # Large gap
    r_step = random.choice([1, 2, 3])
    R = _generate_6term_polynomial(r_start, r_step)
    
    # Multiply
    P = _poly_mul(Q, R)
    
    # Check constraints
    if len(P) > 41:
        return None
    P = _pad_to_length(P, 41)
    
    # Check exactly 12 nonzero terms
    nz_count = sum(1 for x in P if x != 0)
    if nz_count != 12:
        return None
    
    # Check nonzero final coefficient
    if P[-1] == 0:
        # Find last nonzero
        for i in range(len(P) - 1, -1, -1):
            if P[i] != 0:
                P = P[:i+1]
                break
        else:
            return None
    
    if len(P) > 41 or P[-1] == 0:
        return None
    
    # Check coefficient bounds
    if any(abs(x) > 64 for x in P):
        return None
    
    return P

def _generate_alternating_product():
    """
    Generate P = Q * R where Q and R have alternating signs to encourage cancellations.
    """
    # Q with alternating signs
    q_start = random.randint(0, 3)
    q_step = random.choice([1, 2])
    Q = [0] * 41
    for i in range(6):
        exp = q_start + q_step * i
        if exp < 41:
            Q[exp] = 1 if i % 2 == 0 else -1
    
    # R with alternating signs, offset
    r_start = q_start + random.randint(8, 15)
    r_step = random.choice([1, 2])
    R = [0] * 41
    for i in range(6):
        exp = r_start + r_step * i
        if exp < 41:
            R[exp] = 1 if i % 2 == 0 else -1
    
    P = _poly_mul(Q, R)
    P = _pad_to_length(P, 41)
    
    nz_count = sum(1 for x in P if x != 0)
    if nz_count != 12:
        return None
    
    if len(P) > 41 or P[-1] == 0:
        return None
    
    if any(abs(x) > 64 for x in P):
        return None
    
    return P

def _local_search_targeted(poly, max_iters=3000, deadline=None):
    """
    Targeted local search focusing on:
    1. Coefficient magnitude changes (1 <-> 2)
    2. Sign flips
    3. Small exponent shifts
    
    Uses a more aggressive acceptance criterion to escape local minima.
    """
    current_poly = poly[:]
    current_score = _square_count(current_poly)
    best_poly = current_poly[:]
    best_score = current_score
    
    nz_indices = [i for i, v in enumerate(current_poly) if v != 0]
    
    # Simulated annealing parameters
    temperature = 2.0
    cooling_rate = 0.9995
    
    for i in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        move_type = random.random()
        
        if move_type < 0.4:
            # Change coefficient value
            if not nz_indices:
                break
            idx = random.choice(nz_indices)
            old_val = current_poly[idx]
            # Try values that might help with cancellation
            new_val = random.choice([-2, -1, 1, 2])
            if new_val == old_val:
                continue
            
            current_poly[idx] = new_val
            new_score = _square_count(current_poly)
            
            delta = new_score - current_score
            if delta < 0 or (temperature > 0.01 and random.random() < np.exp(-delta / temperature)):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_poly = current_poly[:]
            else:
                current_poly[idx] = old_val
                
        elif move_type < 0.7:
            # Flip sign
            if not nz_indices:
                break
            idx = random.choice(nz_indices)
            old_val = current_poly[idx]
            current_poly[idx] = -old_val
            new_score = _square_count(current_poly)
            
            delta = new_score - current_score
            if delta < 0 or (temperature > 0.01 and random.random() < np.exp(-delta / temperature)):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_poly = current_poly[:]
            else:
                current_poly[idx] = old_val
                
        else:
            # Shift exponent by 1
            if not nz_indices:
                break
            idx = random.choice(nz_indices)
            val = current_poly[idx]
            shift = random.choice([-1, 1])
            new_idx = idx + shift
            
            if new_idx < 0 or new_idx >= len(current_poly):
                continue
            if current_poly[new_idx] != 0:
                continue
            
            current_poly[idx] = 0
            current_poly[new_idx] = val
            new_score = _square_count(current_poly)
            
            delta = new_score - current_score
            if delta < 0 or (temperature > 0.01 and random.random() < np.exp(-delta / temperature)):
                current_score = new_score
                nz_indices.remove(idx)
                nz_indices.append(new_idx)
                if new_score < best_score:
                    best_score = new_score
                    best_poly = current_poly[:]
            else:
                current_poly[new_idx] = 0
                current_poly[idx] = val
                
        temperature *= cooling_rate
    
    return best_poly, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    best_poly = None
    best_score = 1000
    
    # Phase 1: Generate product-based candidates
    candidate_count = 0
    for _ in range(200):
        if time.time() > deadline - 10.0:
            break
        
        # Alternate between different generation strategies
        if candidate_count % 3 == 0:
            cand = _generate_product_candidate()
        elif candidate_count % 3 == 1:
            cand = _generate_alternating_product()
        else:
            # Random 12-term polynomial
            exps = sorted(random.sample(range(0, 41), 12))
            signs = [random.choice([-1, 1]) for _ in range(12)]
            coeffs = [random.choice([1, 2]) for _ in range(12)]
            cand = [0] * 41
            for i in range(12):
                cand[exps[i]] = signs[i] * coeffs[i]
            if cand[-1] == 0:
                # Ensure last is nonzero
                for i in range(len(cand) - 1, -1, -1):
                    if cand[i] != 0:
                        cand = cand[:i+1]
                        break
                else:
                    candidate_count += 1
                    continue
        
        if cand is None:
            candidate_count += 1
            continue
        
        # Validate
        if len(cand) > 41 or cand[-1] == 0:
            candidate_count += 1
            continue
        if sum(1 for x in cand if x != 0) != 12:
            candidate_count += 1
            continue
        if any(abs(x) > 64 for x in cand):
            candidate_count += 1
            continue
        
        score = _square_count(cand)
        if score < best_score:
            best_score = score
            best_poly = cand[:]
        
        # Local search if promising
        if time.time() < deadline - 15.0 and score <= 25:
            cand_deadline = time.time() + 1.5
            refined_poly, refined_score = _local_search_targeted(cand, max_iters=2000, deadline=cand_deadline)
            
            if refined_score < best_score:
                best_score = refined_score
                best_poly = refined_poly[:]
        
        candidate_count += 1
        
        # Early termination if very good
        if best_score <= 14:
            break
    
    # Phase 2: If we have a good candidate, do more intensive local search
    if best_poly is not None and best_score <= 20 and time.time() < deadline - 5.0:
        final_deadline = time.time() + 5.0
        refined_poly, refined_score = _local_search_targeted(
            best_poly, max_iters=5000, deadline=final_deadline
        )
        if refined_score < best_score:
            best_score = refined_score
            best_poly = refined_poly[:]
    
    if best_poly is None:
        # Fallback
        return [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]
    
    return best_poly
