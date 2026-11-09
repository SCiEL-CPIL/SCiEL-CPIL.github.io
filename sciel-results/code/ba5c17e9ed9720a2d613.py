import numpy as np
import time
import random
from itertools import combinations

def _sq_nonzero_count(coeffs):
    """Compute the number of nonzero terms in the square of a polynomial given by coefficient list."""
    # coeffs is a list where coeffs[i] is the coefficient of x^i
    # Square via convolution
    sq = np.convolve(coeffs, coeffs)
    # Count nonzero entries
    return int(np.count_nonzero(sq))

def _valid(coeffs):
    """Check if coeffs is a valid candidate: length<=41, nonzero final coeff, entries in [-64,64], exactly 12 nonzero terms."""
    if len(coeffs) > 41:
        return False
    if len(coeffs) == 0:
        return False
    if coeffs[-1] == 0:
        return False
    if any(abs(c) > 64 for c in coeffs):
        return False
    nz = sum(1 for c in coeffs if c != 0)
    if nz != 12:
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    best_score = None
    best_coeffs = None
    
    # Strategy: search over exponent patterns and coefficient assignments
    # Key: choose 12 exponents, assign small integer coefficients, check square sparsity
    
    # Precompute: try palindromic structures which encourage cancellation
    # If P is palindromic of degree d, P(x) = x^d * P(1/x), then P^2 is palindromic too
    
    # Phase 1: Try structured patterns
    # Pattern A: exponents symmetric around center
    # Pattern B: arithmetic progression
    
    np.random.seed(42)
    random.seed(42)
    
    # Try various exponent patterns
    def try_pattern(exponents, coeff_range=(-3, 3)):
        nonlocal best_score, best_coeffs
        exponents = sorted(set(exponents))
        if len(exponents) < 12:
            return
        exponents = exponents[:12]  # take first 12 if too many
        if len(exponents) != 12:
            return
        
        max_deg = exponents[-1]
        if max_deg > 40:
            return
        
        # Try random coefficient assignments
        for _ in range(200):
            coeffs_arr = np.zeros(max_deg + 1, dtype=int)
            for i, e in enumerate(exponents):
                c = random.randint(coeff_range[0], coeff_range[1])
                if c == 0:
                    c = 1  # ensure nonzero
                coeffs_arr[e] = c
            
            # Ensure last coefficient is nonzero
            if coeffs_arr[-1] == 0:
                coeffs_arr[-1] = 1
            
            # Check exactly 12 nonzero
            nz = np.count_nonzero(coeffs_arr)
            if nz != 12:
                continue
            
            # Compute square
            sq = np.convolve(coeffs_arr, coeffs_arr)
            nz_sq = np.count_nonzero(sq)
            
            if best_score is None or nz_sq < best_score:
                best_score = nz_sq
                best_coeffs = coeffs_arr.tolist()
            
            if best_score <= 2:  # Can't do much better than 2 (theoretical min for 12 terms)
                return
            if time.time() > deadline:
                return
    
    # Phase 1a: Palindromic patterns
    # For palindromic polynomial of degree d: coeffs[i] = coeffs[d-i]
    # Choose 6 pairs (or 6 terms if center exists)
    
    for d in range(12, 41, 2):  # even degree for palindromic
        if time.time() > deadline:
            break
        # Choose 6 exponent pairs that sum to d
        # exponents: a, d-a, b, d-b, ... with a < b < ... < d/2
        half = d // 2
        candidates = list(range(0, half + 1))
        
        for _ in range(50):
            if time.time() > deadline:
                break
            if len(candidates) < 6:
                continue
            chosen = random.sample(candidates, 6)
            exponents = []
            for c in chosen:
                if c == d - c:  # center
                    exponents.append(c)
                else:
                    exponents.append(c)
                    exponents.append(d - c)
            exponents = sorted(set(exponents))
            if len(exponents) != 12:
                continue
            try_pattern(exponents, (-4, 4))
    
    # Phase 1b: Arithmetic progression patterns
    for _ in range(100):
        if time.time() > deadline:
            break
        a = random.randint(0, 20)
        step = random.randint(1, 4)
        exponents = [a + i * step for i in range(12)]
        if exponents[-1] > 40:
            continue
        try_pattern(exponents, (-4, 4))
    
    # Phase 1c: Random exponent patterns
    for _ in range(200):
        if time.time() > deadline:
            break
        exponents = sorted(random.sample(range(0, 41), 12))
        try_pattern(exponents, (-4, 4))
    
    # Phase 2: Local search - perturb coefficients of best solution
    if best_coeffs is not None and best_score > 2:
        if time.time() > deadline:
            pass
        else:
            cur = np.array(best_coeffs, dtype=int)
            cur_score = best_score
            max_iter = 5000
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                # Random perturbation: change one coefficient
                idx = random.randint(0, len(cur) - 1)
                old_val = cur[idx]
                new_val = random.randint(-64, 64)
                if new_val == old_val:
                    continue
                cur[idx] = new_val
                
                # Check validity
                if cur[-1] == 0:
                    cur[idx] = old_val
                    continue
                if any(abs(c) > 64 for c in cur):
                    cur[idx] = old_val
                    continue
                nz = np.count_nonzero(cur)
                if nz != 12:
                    cur[idx] = old_val
                    continue
                
                sq = np.convolve(cur, cur)
                nz_sq = np.count_nonzero(sq)
                
                if nz_sq < cur_score:
                    cur_score = nz_sq
                    best_score = nz_sq
                    best_coeffs = cur.tolist()
                    if cur_score <= 2:
                        break
                else:
                    cur[idx] = old_val
    
    # Phase 3: Try to find even sparser squares with targeted search
    # Look for P where P^2 has very few terms
    # Key insight: if P(x) = x^a * Q(x) where Q^2 is sparse, same result
    # Try: P with coefficients that create exact cancellations
    
    # Try specific known-good structures
    # If P(x) = x^k * (x^n + x^m + ...) where the exponents create a "difference set"
    
    if best_coeffs is None:
        # Fallback: simple 12-term polynomial
        best_coeffs = [0] * 41
        for i in range(12):
            best_coeffs[i] = 1
        best_score = _sq_nonzero_count(best_coeffs)
    
    # Ensure validity
    if not _valid(best_coeffs):
        # Fix: ensure exactly 12 nonzero, last nonzero, in range
        best_coeffs = [0] * 41
        for i in range(12):
            best_coeffs[i] = 1
        best_score = _sq_nonzero_count(best_coeffs)
    
    return best_coeffs
