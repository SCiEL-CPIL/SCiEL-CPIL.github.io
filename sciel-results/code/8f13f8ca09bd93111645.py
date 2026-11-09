import numpy as np
import time
from itertools import combinations
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin
    
    best_count = 144  # worst case: all 144 terms nonzero
    best_poly = None
    
    # Helper: compute square and count nonzeros
    def square_nonzero_count(coeffs):
        """coeffs is a list of (exponent, coefficient) pairs"""
        if not coeffs:
            return 0, []
        max_deg = max(e for e, c in coeffs)
        arr = np.zeros(2 * max_deg + 1, dtype=np.int64)
        for e, c in coeffs:
            arr[e] = c
        sq = np.convolve(arr, arr)
        return np.count_nonzero(sq), sq
    
    # Helper: convert list of (exp, coeff) to coefficient list
    def to_coeff_list(coeffs):
        if not coeffs:
            return [0]
        max_deg = max(e for e, c in coeffs)
        result = [0] * (max_deg + 1)
        for e, c in coeffs:
            result[e] = c
        # Trim trailing zeros but keep nonzero final
        while len(result) > 1 and result[-1] == 0:
            result.pop()
        return result
    
    # Strategy 1: Try structured exponent sets
    # Geometric-like, arithmetic-like, and random sets
    
    coeff_choices = [-2, -1, 1, 2]
    
    # Try various structured exponent sets
    exponent_sets = []
    
    # Arithmetic progressions
    for a in range(0, 5):
        for d in range(1, 8):
            exps = [a + i * d for i in range(12)]
            if max(exps) <= 40:
                exponent_sets.append(tuple(exps))
    
    # Geometric-like (powers of 2, 3, etc.)
    for base in [2, 3]:
        exps = [base ** i for i in range(12)]
        if max(exps) <= 40:
            exponent_sets.append(tuple(exps))
    
    # Mixed: some structure + random
    random.seed(42)
    for _ in range(200):
        # Start with a base structure and perturb
        base_exp = random.randint(0, 5)
        d = random.randint(1, 6)
        exps = [base_exp + i * d + random.randint(-2, 2) for i in range(12)]
        exps = sorted(set(exps))
        if len(exps) < 12:
            # Fill gaps
            for e in range(41):
                if e not in exps and len(exps) < 12:
                    exps.append(e)
            exps = sorted(exps[:12])
        if max(exps) <= 40 and len(exps) == 12:
            exponent_sets.append(tuple(exps))
    
    # Random sets
    for _ in range(300):
        exps = sorted(random.sample(range(41), 12))
        exponent_sets.append(tuple(exps))
    
    # Deduplicate
    exponent_sets = list(set(exponent_sets))
    
    # For each exponent set, try coefficient combinations
    # Full 4^12 is too large, so use random sampling + local search
    
    for exp_set in exponent_sets:
        if time.time() > deadline:
            break
        
        # Try several random coefficient assignments
        for trial in range(50):
            if time.time() > deadline:
                break
            
            coeffs = []
            for e in exp_set:
                c = random.choice(coeff_choices)
                coeffs.append((e, c))
            
            count, _ = square_nonzero_count(coeffs)
            if count < best_count:
                best_count = count
                best_poly = coeffs
                if best_count <= 12:
                    # Very good, try to polish
                    break
        
        # Local search on best found for this exp_set
        if best_poly and tuple(e for e, c in best_poly) == exp_set:
            # Try flipping individual coefficients
            improved = True
            while improved:
                if time.time() > deadline:
                    break
                improved = False
                for i in range(len(best_poly)):
                    e, c = best_poly[i]
                    for new_c in coeff_choices:
                        if new_c == c:
                            continue
                        trial_poly = list(best_poly)
                        trial_poly[i] = (e, new_c)
                        count, _ = square_nonzero_count(trial_poly)
                        if count < best_count:
                            best_count = count
                            best_poly = trial_poly
                            improved = True
                            break
    
    # If we found something good, try to optimize further
    if best_poly is None:
        # Fallback: use a known sparse structure
        # (1+x+x^2+...+x^11)^2 has 23 nonzero terms
        best_poly = [(i, 1) for i in range(12)]
        best_count, _ = square_nonzero_count(best_poly)
    
    # Convert to coefficient list
    result = to_coeff_list(best_poly)
    
    # Ensure length <= 41
    if len(result) > 41:
        result = result[:41]
    
    # Ensure nonzero final coefficient
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    
    # Ensure entries in [-64, 64]
    result = [max(-64, min(64, c)) for c in result]
    
    # Ensure exactly 12 nonzero terms in the polynomial itself
    nonzero = [(i, c) for i, c in enumerate(result) if c != 0]
    if len(nonzero) != 12:
        # This shouldn't happen, but handle it
        pass
    
    return result
