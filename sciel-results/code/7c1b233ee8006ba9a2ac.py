"""Strategy: Local search (hill-climbing) with NumPy-accelerated square evaluation.
Initializes with structured blocks to ensure validity, then iteratively perturbs
coefficients to minimize the number of nonzero terms in the square.
"""
import numpy as np
import time
import sys

# EVOLVE-BLOCK-START

_DEADLINE = time.time() + 85.0

def _is_deadline():
    return time.time() >= _DEADLINE

def _square_count_fast(coeffs):
    """
    Compute the number of nonzero terms in the square of the polynomial
    defined by coeffs (list of ints, ascending degree).
    Uses numpy convolution for speed.
    """
    if len(coeffs) == 0:
        return 0
    # Ensure integer type to avoid float issues, though coeffs are ints
    arr = np.array(coeffs, dtype=np.int64)
    # Convolution of arr with itself gives the coefficients of the square
    # np.convolve does full convolution
    sq = np.convolve(arr, arr)
    # Count nonzeros
    return np.count_nonzero(sq)

def _normalize(poly):
    """
    Ensure the polynomial meets all constraints:
    - Length <= 41
    - Nonzero final coefficient
    - Exactly 12 nonzero terms
    - Entries in [-64, 64]
    Returns the normalized list or None.
    """
    if not poly:
        return None
    
    # Trim trailing zeros
    while len(poly) > 1 and poly[-1] == 0:
        poly = poly[:-1]
    
    if len(poly) > 41:
        return None
    if poly[-1] == 0:
        return None
    if len(poly) < 1:
        return None
        
    nz = sum(1 for c in poly if c != 0)
    if nz != 12:
        return None
        
    for c in poly:
        if not (-64 <= c <= 64):
            return None
            
    return poly

def _generate_initial_candidate():
    """
    Generate a valid 12-term polynomial.
    Strategy: Use 3 blocks of 4 terms each, separated by gaps to minimize initial overlap,
    or just random distinct degrees.
    Let's try a structured approach: 
    Degrees: 0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23
    Coeffs: All 1s initially.
    """
    poly = [0] * 41
    degrees = [0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23]
    for d in degrees:
        poly[d] = 1
    return poly

def _search():
    # Start with a valid candidate
    current_poly = _generate_initial_candidate()
    norm = _normalize(current_poly)
    if norm is None:
        # Fallback if generation fails (shouldn't happen)
        current_poly = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1]
        # This fallback is messy, let's just use a simple valid one
        current_poly = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # Wait, that's 12 terms? 4+4+4 = 12. Yes.
        # Length is 24. Nonzero final coeff is 1.
        # Let's just use the generator result, it's valid.
        pass
    
    current_poly = _normalize(current_poly)
    if current_poly is None:
        # Extreme fallback
        current_poly = [1]*12
        current_poly = _normalize(current_poly)
        
    current_score = _square_count_fast(current_poly)
    best_poly = list(current_poly)
    best_score = current_score
    
    # Local search
    # We will iterate, trying to change one coefficient at a time.
    # Moves:
    # 1. Change value of an existing nonzero term to another value in [-64, 64] \ {0}
    # 2. Move a nonzero term to a new degree (zeroing old, setting new)
    # 3. Change a zero term to a nonzero value, and zero out another term (maintain 12 nonzero count)
    
    # To keep it simple and fast, we'll focus on value changes and moves.
    
    # Precompute indices of nonzero terms
    nz_indices = [i for i, c in enumerate(current_poly) if c != 0]
    
    # Value options for coefficients
    # We don't want to try all 128 values every time. 
    # Common "good" values for cancellation are small integers or factors.
    # Let's try a subset: [-4, -3, -2, -1, 1, 2, 3, 4, -8, 8, -16, 16, -32, 32, -64, 64]
    # And maybe the current value +/- 1.
    value_pool = [-64, -32, -16, -8, -4, -3, -2, -1, 1, 2, 3, 4, 8, 16, 32, 64]
    
    no_improve_count = 0
    max_no_improve = 5000
    
    while not _is_deadline():
        if no_improve_count > max_no_improve:
            # Try a random restart or larger perturbation?
            # For now, break and return best
            break
            
        # Pick a random nonzero index to modify
        idx = nz_indices[np.random.randint(len(nz_indices))]
        current_val = current_poly[idx]
        
        # Option A: Change value
        # Pick a new value from pool, different from current
        new_val = value_pool[np.random.randint(len(value_pool))]
        if new_val == current_val:
            new_val = -new_val if new_val != 0 else 1
            if new_val == current_val:
                new_val = 1
                
        old_val = current_poly[idx]
        current_poly[idx] = new_val
        
        # Check validity (value range is already ok, nonzero count unchanged)
        # Just check score
        new_score = _square_count_fast(current_poly)
        
        if new_score < current_score:
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_poly = list(current_poly)
            no_improve_count = 0
        else:
            current_poly[idx] = old_val
            no_improve_count += 1
            
        # Option B: Move term (with some probability)
        if np.random.random() < 0.2:
            # Pick a new degree
            # Ensure new degree is within [0, 40]
            new_deg = np.random.randint(0, 41)
            if new_deg == idx:
                continue
                
            # Check if new_deg is already occupied
            if current_poly[new_deg] != 0:
                continue
                
            # Move
            val = current_poly[idx]
            current_poly[idx] = 0
            current_poly[new_deg] = val
            
            new_score = _square_count_fast(current_poly)
            
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_poly = list(current_poly)
                no_improve_count = 0
                # Update nz_indices
                nz_indices.remove(idx)
                nz_indices.append(new_deg)
            else:
                current_poly[new_deg] = 0
                current_poly[idx] = val
                no_improve_count += 1
                
    return best_poly

# EVOLVE-BLOCK-END

def construct():
    """
    Return coefficient list in ascending degree, length<=41, nonzero final
    coefficient, entries integers in [-64,64], exactly 12 nonzero terms.
    Minimize nonzero terms in the EXACT integer square.
    """
    result = _search()
    if result is None:
        result = [1] * 12
    return result

