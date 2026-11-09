"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import itertools
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _square_coeffs(c):
    """Return coefficient list of c*c (convolution), trimmed to remove trailing zeros."""
    if not c:
        return [0]
    # Use numpy for speed
    c_arr = np.array(c, dtype=np.int64)
    result = np.convolve(c_arr, c_arr)
    # Trim trailing zeros
    nz = np.nonzero(result)[0]
    if len(nz) == 0:
        return [0]
    last = nz[-1]
    return result[:last+1].tolist()

def _nnz(c):
    """Count non-zero terms in a coefficient list."""
    if not c:
        return 0
    return np.count_nonzero(np.array(c, dtype=np.int64))

def _evaluate_poly(coeffs, exponents):
    """
    Build the full coefficient list from sparse representation (coeffs, exponents)
    and return (nnz_of_square, full_coeff_list).
    """
    max_deg = max(exponents) if exponents else 0
    full = [0] * (max_deg + 1)
    for c, e in zip(coeffs, exponents):
        if 0 <= e <= max_deg:
            full[e] = c
    
    # Check validity: exactly 12 non-zero terms in P
    if _nnz(full) != 12:
        return float('inf'), full
    
    sq = _square_coeffs(full)
    return _nnz(sq), full

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    best_nnz = float('inf')
    best_poly = [1] * 12  # Fallback
    
    # Helper to update best
    def try_update(nnz_val, poly_list):
        nonlocal best_nnz, best_poly
        if nnz_val < best_nnz:
            best_nnz = nnz_val
            best_poly = poly_list[:]
    
    # 1. Generate diverse initial candidates
    candidates = []
    
    # Candidate type A: Palindromic degree 11 (12 coeffs)
    # This was the seed's focus, keep it as a strong baseline
    for _ in range(100):
        free = [random.choice([-2, -1, 1, 2, 3, -3]) for _ in range(6)]
        # Ensure non-zero
        free = [f if f != 0 else 1 for f in free]
        poly = free + [free[5], free[4], free[3], free[2], free[1], free[0]]
        candidates.append(poly)
        
    # Candidate type B: Sparse random exponents
    # 12 non-zero terms, random exponents in [0, 40]
    for _ in range(200):
        exponents = random.sample(range(41), 12)
        exponents.sort()
        coeffs = [random.choice([-2, -1, 1, 2, 3, 4, -3, -4]) for _ in range(12)]
        full = [0] * 41
        for c, e in zip(coeffs, exponents):
            full[e] = c
        # Trim trailing zeros
        while len(full) > 1 and full[-1] == 0:
            full.pop()
        if len(full) > 0 and full[-1] != 0 and _nnz(full) == 12:
            candidates.append(full)
            
    # Candidate type C: Structured "Clustered" exponents
    # e.g., exponents in two clusters to promote cancellation
    for _ in range(100):
        # Two clusters of 6 terms
        center1 = random.randint(0, 10)
        center2 = random.randint(20, 40)
        exps1 = random.sample(range(max(0, center1-3), min(41, center1+4)), 6)
        exps2 = random.sample(range(max(0, center2-3), min(41, center2+4)), 6)
        exps = sorted(exps1 + exps2)
        if len(set(exps)) != 12: continue
        coeffs = [random.choice([-1, 1, -2, 2]) for _ in range(12)]
        full = [0] * 41
        for c, e in zip(coeffs, exps):
            full[e] = c
        while len(full) > 1 and full[-1] == 0:
            full.pop()
        if _nnz(full) == 12:
            candidates.append(full)

    # Evaluate initial candidates
    for c in candidates:
        if time.time() > deadline:
            break
        if len(c) > 41: continue
        if _nnz(c) != 12: continue
        if any(abs(x) > 64 for x in c): continue
        if not c or c[-1] == 0: continue
        
        nnz_sq = _nnz(_square_coeffs(c))
        try_update(nnz_sq, c)

    # 2. Local Search (Hill Climbing with Annealing)
    # We will run local search on the best candidate found so far, and also on a few other top candidates.
    
    # Get top 5 candidates by square sparsity
    scored_candidates = []
    seen = set()
    for c in candidates:
        key = tuple(c)
        if key in seen: continue
        seen.add(key)
        if _nnz(c) != 12: continue
        nnz_sq = _nnz(_square_coeffs(c))
        scored_candidates.append((nnz_sq, c))
        
    scored_candidates.sort(key=lambda x: x[0])
    
    # Run local search on the top 5 distinct candidates
    for i in range(min(5, len(scored_candidates))):
        if time.time() > deadline:
            break
        _, current_poly = scored_candidates[i]
        current_nnz = _nnz(_square_coeffs(current_poly))
        
        # Represent as list of (coeff, exponent) for easier manipulation
        # But for speed, we operate on the full list and swap values.
        
        steps = 0
        max_steps = 2000
        temperature = 1.0
        
        while steps < max_steps and time.time() < deadline:
            steps += 1
            if steps % 50 == 0:
                temperature *= 0.95 # Anneal
                
            # Choose a move
            # 1. Change a random non-zero coefficient
            # 2. Change a random zero coefficient to non-zero and set another non-zero to zero (keep count 12)
            
            move_type = random.random()
            new_poly = current_poly[:]
            
            if move_type < 0.7:
                # Change coefficient value
                non_zero_indices = [i for i, v in enumerate(new_poly) if v != 0]
                if not non_zero_indices: continue
                idx = random.choice(non_zero_indices)
                old_val = new_poly[idx]
                new_val = random.choice([-64, -32, -16, -8, -4, -2, -1, 1, 2, 4, 8, 16, 32, 64])
                if new_val == old_val: continue
                new_poly[idx] = new_val
            else:
                # Swap exponents: pick one non-zero and one zero, swap their "status"
                # i.e., move a term from one exponent to another
                non_zero_indices = [i for i, v in enumerate(new_poly) if v != 0]
                zero_indices = [i for i, v in enumerate(new_poly) if v == 0]
                if not non_zero_indices or not zero_indices: continue
                
                src_idx = random.choice(non_zero_indices)
                dst_idx = random.choice(zero_indices)
                
                # Move coeff from src to dst
                val = new_poly[src_idx]
                new_poly[src_idx] = 0
                new_poly[dst_idx] = val
                
                # Ensure we still have 12 non-zeros (we just moved one, so count is preserved)
                # But we might have created a trailing zero if dst was the last element and src was before it?
                # No, we just moved the value. The list length is fixed.
                # However, if the new highest degree term is zero, we should trim?
                # The problem says "length <= 41". It doesn't require minimal length, just valid.
                # But "nonzero final coefficient" is a constraint.
                # So if new_poly[-1] becomes 0, we must ensure some higher term is non-zero or trim.
                # Actually, if we move the last non-zero term to a lower index, the final coefficient becomes 0.
                # We need to trim trailing zeros to satisfy "nonzero final coefficient" if we interpret it strictly.
                # The contract says: "Return coefficient list ... nonzero final coefficient".
                # So we must trim.
                
                # Trim trailing zeros
                while len(new_poly) > 1 and new_poly[-1] == 0:
                    new_poly.pop()
                
                if not new_poly or new_poly[-1] == 0:
                    # Invalid state, revert
                    new_poly = current_poly[:]
                    continue
                    
                if _nnz(new_poly) != 12:
                    # Should not happen if we just moved a non-zero to a zero slot
                    # unless the zero slot was out of bounds? No, indices are within list.
                    continue

            # Evaluate new state
            if any(abs(x) > 64 for x in new_poly):
                continue
                
            new_nnz = _nnz(_square_coeffs(new_poly))
            
            # Acceptance criterion
            if new_nnz < current_nnz:
                current_poly = new_poly
                current_nnz = new_nnz
                try_update(current_nnz, current_poly)
            elif new_nnz == current_nnz:
                # Accept equal with some probability to explore
                if random.random() < 0.2:
                    current_poly = new_poly
            else:
                # Accept worse with probability based on temperature
                # delta = new_nnz - current_nnz (positive)
                # prob = exp(-delta / temp)
                # Since nnz is small integer, delta is usually 1, 2, etc.
                if random.random() < (0.1 * temperature):
                    current_poly = new_poly
                    current_nnz = new_nnz

    # Final validation
    # Ensure best_poly is valid
    if len(best_poly) > 41:
        best_poly = best_poly[:41]
    while len(best_poly) > 1 and best_poly[-1] == 0:
        best_poly.pop()
    
    if not best_poly or best_poly[-1] == 0 or _nnz(best_poly) != 12:
        # Fallback to a known valid 12-term poly
        best_poly = [1]*12
        
    if any(abs(x) > 64 for x in best_poly):
        best_poly = [1]*12

    return best_poly

# EVOLVE-BLOCK-END
