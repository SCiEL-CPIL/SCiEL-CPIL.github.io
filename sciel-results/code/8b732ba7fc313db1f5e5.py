"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import product

def _square_nonzero_count_from_pairs(pairs):
    """Compute number of nonzero terms in square of polynomial given as list of (exponent, coeff)."""
    n = len(pairs)
    sq = {}
    for i in range(n):
        ei, ci = pairs[i]
        for j in range(i, n):
            ej, cj = pairs[j]
            pos = ei + ej
            val = ci * cj
            if i != j:
                val *= 2
            sq[pos] = sq.get(pos, 0) + val
    return sum(1 for v in sq.values() if v != 0)

def _square_nonzero_count_dense(coeffs):
    """Compute number of nonzero terms in square of dense coefficient list."""
    n = len(coeffs)
    sq = [0] * (2 * n - 1)
    for i in range(n):
        ci = coeffs[i]
        if ci == 0:
            continue
        for j in range(i, n):
            cj = coeffs[j]
            if cj == 0:
                continue
            val = ci * cj
            if i != j:
                val *= 2
            sq[i + j] += val
    return sum(1 for v in sq if v != 0)

def _validate_and_build(exps, coeffs):
    """Validate exponent/coeff pair and build dense list. Returns (list, sq_count) or (None, 999)."""
    if len(exps) != 12 or len(coeffs) != 12:
        return None, 999
    # Check exactly 12 nonzero
    nz = sum(1 for c in coeffs if c != 0)
    if nz != 12:
        return None, 999
    # Check bounds
    if any(c < -64 or c > 64 for c in coeffs):
        return None, 999
    # Check exponents
    if any(e < 0 for e in exps):
        return None, 999
    max_deg = max(exps)
    if max_deg >= 41:
        return None, 999
    # Build dense list
    dense = [0] * (max_deg + 1)
    for e, c in zip(exps, coeffs):
        dense[e] = c
    # Trim trailing zeros
    while len(dense) > 1 and dense[-1] == 0:
        dense.pop()
    # Final check: last coefficient nonzero
    if dense[-1] == 0:
        return None, 999
    # Count square terms
    pairs = [(exps[i], coeffs[i]) for i in range(12) if coeffs[i] != 0]
    sq_count = _square_nonzero_count_from_pairs(pairs)
    return dense, sq_count

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin before 90s kill
    
    best_count = 23
    best_poly = [1] * 12
    
    # Phase 1: Brute-force sign patterns for consecutive exponents (0..11)
    # This is fast: 2^12 = 4096 patterns
    exps_consec = list(range(12))
    for pattern in product([-1, 1], repeat=12):
        if time.time() - start_time > deadline:
            break
        cnt = _square_nonzero_count_from_pairs([(exps_consec[i], pattern[i]) for i in range(12)])
        if cnt < best_count:
            best_count = cnt
            best_poly = list(pattern)
    
    # Also try small integer coefficients beyond ±1 for consecutive exponents
    # Sample some structured patterns
    small_coeffs_pool = [-3, -2, -1, 1, 2, 3]
    
    # Try patterns with symmetry
    for trial in range(200):
        if time.time() - start_time > deadline:
            break
        # Random small coefficients
        coeffs = [random.choice(small_coeffs_pool) for _ in range(12)]
        # Ensure last is nonzero (it is by construction)
        cnt = _square_nonzero_count_dense(coeffs)
        if cnt < best_count:
            best_count = cnt
            best_poly = coeffs[:]
    
    # Phase 2: Local search from best found
    current_poly = best_poly[:]
    current_count = best_count
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() - start_time < deadline and no_improve < max_no_improve:
        # Pick random index
        idx = random.randint(0, 11)
        old_val = current_poly[idx]
        
        # Try a new value
        new_val = random.randint(-10, 10)
        if new_val == 0 or new_val == old_val:
            continue
        
        current_poly[idx] = new_val
        new_count = _square_nonzero_count_dense(current_poly)
        
        if new_count <= current_count:
            current_count = new_count
            no_improve = 0
            if new_count < best_count:
                best_count = new_count
                best_poly = current_poly[:]
        else:
            current_poly[idx] = old_val
            no_improve += 1
            
        # Periodically reset to best to escape local minima
        if no_improve % 2000 == 0:
            current_poly = best_poly[:]
            current_count = best_count
    
    # Phase 3: Try non-consecutive exponent sets with brute-force signs
    if time.time() - start_time < deadline - 5.0:
        # Try some structured exponent sets
        exp_sets = []
        # Gapped sets
        exp_sets.append(list(range(0, 24, 2)))  # 0,2,4,...,22
        exp_sets.append(list(range(0, 24, 2)))  # same
        # Two clusters
        exp_sets.append([0,1,2,3,4,5, 15,16,17,18,19,20])
        exp_sets.append([0,1,2,3,4,5, 10,11,12,13,14,15])
        
        for exps in exp_sets:
            if time.time() - start_time > deadline - 2.0:
                break
            if max(exps) >= 41:
                continue
            # Brute force signs for this exponent set
            for pattern in product([-1, 1], repeat=12):
                if time.time() - start_time > deadline - 1.0:
                    break
                pairs = [(exps[i], pattern[i]) for i in range(12)]
                cnt = _square_nonzero_count_from_pairs(pairs)
                if cnt < best_count:
                    best_count = cnt
                    # Build dense list
                    max_deg = max(exps)
                    dense = [0] * (max_deg + 1)
                    for e, c in zip(exps, pattern):
                        dense[e] = c
                    while len(dense) > 1 and dense[-1] == 0:
                        dense.pop()
                    best_poly = dense
    
    # Final validation
    if best_poly[-1] == 0:
        best_poly[-1] = 1
    
    return best_poly
# EVOLVE-BLOCK-END
