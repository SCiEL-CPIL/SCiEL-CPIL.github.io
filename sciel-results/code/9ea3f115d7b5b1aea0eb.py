"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product, combinations

def construct():
    deadline = time.time() + 82.0  # safety margin before 90s kill
    
    best_poly = [1] * 12
    best_count = 23  # seed: (1+x+...+x^11)^2 has 23 nonzero terms
    
    def square_nonzero_count_fast(coeff_list):
        """Fast square computation using FFT convolution."""
        n = len(coeff_list)
        if n == 0:
            return 0
        # Pad to power of 2 for FFT efficiency
        size = 1
        while size < 2 * n - 1:
            size *= 2
        a = np.zeros(size, dtype=np.float64)
        for i, c in enumerate(coeff_list):
            if c != 0:
                a[i] = c
        # FFT convolution
        fa = np.fft.fft(a)
        sq = np.fft.ifft(fa * fa).real
        sq = np.round(sq).astype(np.int64)
        sq = sq[:2 * n - 1]
        return int(np.count_nonzero(sq))
    
    def square_nonzero_count_exact(coeff_list):
        """Exact square computation for verification."""
        n = len(coeff_list)
        sq = [0] * (2 * n - 1)
        for i in range(n):
            ci = coeff_list[i]
            if ci == 0:
                continue
            for j in range(n):
                cj = coeff_list[j]
                if cj == 0:
                    continue
                sq[i + j] += ci * cj
        return sum(1 for v in sq if v != 0)
    
    def trim(poly):
        """Trim trailing zeros, ensure length >= 1."""
        while len(poly) > 1 and poly[-1] == 0:
            poly.pop()
        return poly
    
    def is_valid(poly):
        """Check if polynomial meets all constraints."""
        if len(poly) > 41:
            return False
        if len(poly) < 1 or poly[-1] == 0:
            return False
        nonzero = sum(1 for c in poly if c != 0)
        if nonzero != 12:
            return False
        if any(c < -64 or c > 64 for c in poly):
            return False
        return True
    
    def try_poly(coeffs):
        """Try a coefficient list, update best if improved."""
        nonlocal best_poly, best_count
        # Quick validity check
        nonzero = sum(1 for c in coeffs if c != 0)
        if nonzero != 12:
            return
        if any(abs(c) > 64 for c in coeffs):
            return
        # Find max degree
        max_deg = 0
        for i in range(len(coeffs) - 1, -1, -1):
            if coeffs[i] != 0:
                max_deg = i
                break
        if max_deg >= len(coeffs):
            return
        trimmed = trim(coeffs[:])
        if trimmed[-1] == 0:
            return
        cnt = square_nonzero_count_fast(trimmed)
        if cnt < best_count:
            best_count = cnt
            best_poly = trimmed
    
    # Phase 1: Exhaustive sign search on structured exponent families
    # These families are chosen to maximize pairwise sum collisions
    
    exponent_families = [
        # Consecutive
        list(range(12)),
        # Even spacing
        list(range(0, 24, 2)),
        # Triple spacing
        list(range(0, 36, 3)),
        # Mixed: two blocks of 6
        [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12],
        # Three blocks of 4
        [0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14, 15],
        # Four blocks of 3
        [0, 1, 2, 5, 6, 7, 10, 11, 12, 15, 16, 17],
        # Six blocks of 2
        [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16],
        # Alternating gaps
        [0, 1, 3, 4, 6, 7, 8, 10, 11, 13, 14, 15],
        # Clustered with wide gaps
        [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15],
        # Fibonacci-like
        [0, 1, 2, 3, 5, 8, 13, 21, 34, 35, 36, 37],
        # Symmetric around center
        [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15],
        # Powers of 2 subset
        [0, 1, 2, 4, 5, 8, 9, 10, 16, 17, 18, 19],
    ]
    
    for exp_set in exponent_families:
        if time.time() > deadline - 30:
            break
        # Fix first coefficient sign = +1, enumerate remaining 11 signs
        for signs in product([1, -1], repeat=11):
            if time.time() > deadline - 30:
                break
            coeffs = [0] * 41
            coeffs[exp_set[0]] = 1
            for i, s in enumerate(signs):
                coeffs[exp_set[i+1]] = s
            try_poly(coeffs)
    
    # Phase 2: Local search on best solution found
    if time.time() < deadline - 15:
        work_poly = best_poly[:]
        work_count = best_count
        max_degree = len(work_poly)
        
        # Single-position local search
        for iteration in range(5000):
            if time.time() > deadline - 15:
                break
            
            pos = np.random.randint(0, max_degree)
            old_val = work_poly[pos]
            
            # Try different values
            if old_val == 0:
                new_vals = [1, -1, 2, -2, 3, -3]
            elif old_val == 1:
                new_vals = [-1, 2, -2, 0, 3, -3]
            elif old_val == -1:
                new_vals = [1, -2, 2, 0, -3, 3]
            elif old_val == 2:
                new_vals = [1, -1, 0, -2, 3, -3]
            elif old_val == -2:
                new_vals = [-1, 1, 0, 2, -3, 3]
            else:
                new_vals = [1, -1, 2, -2, 0, 3, -3]
            
            for new_val in new_vals:
                if new_val == old_val or abs(new_val) > 64:
                    continue
                
                work_poly[pos] = new_val
                nonzero = sum(1 for c in work_poly if c != 0)
                if nonzero != 12 or work_poly[-1] == 0:
                    work_poly[pos] = old_val
                    continue
                
                cnt = square_nonzero_count_fast(work_poly)
                if cnt < work_count:
                    work_count = cnt
                    if cnt < best_count:
                        best_count = cnt
                        best_poly = trim(work_poly[:])
                else:
                    work_poly[pos] = old_val
    
    # Phase 3: Paired-position local search
    if time.time() < deadline - 5:
        work_poly = best_poly[:]
        work_count = best_count
        max_degree = len(work_poly)
        
        for iteration in range(5000):
            if time.time() > deadline - 5:
                break
            
            # Pick 2 positions to modify
            pos1, pos2 = np.random.choice(max_degree, size=2, replace=False)
            old1, old2 = work_poly[pos1], work_poly[pos2]
            
            val_choices = [1, -1, 2, -2, 3, -3, 0]
            v1 = np.random.choice(val_choices)
            v2 = np.random.choice(val_choices)
            
            if abs(v1) > 64 or abs(v2) > 64:
                continue
            
            work_poly[pos1] = v1
            work_poly[pos2] = v2
            
            nonzero = sum(1 for c in work_poly if c != 0)
            if nonzero != 12 or work_poly[-1] == 0:
                work_poly[pos1] = old1
                work_poly[pos2] = old2
                continue
            
            cnt = square_nonzero_count_fast(work_poly)
            if cnt < work_count:
                work_count = cnt
                if cnt < best_count:
                    best_count = cnt
                    best_poly = trim(work_poly[:])
            else:
                work_poly[pos1] = old1
                work_poly[pos2] = old2
    
    # Phase 4: Structured coefficient patterns with ±2
    if time.time() < deadline:
        # Try patterns where some coefficients are ±2 to create more cancellations
        for exp_set in exponent_families[:6]:
            if time.time() > deadline:
                break
            
            n_terms = len(exp_set)
            # Choose 1-3 positions to have coefficient ±2
            for n_twos in range(1, 4):
                if time.time() > deadline:
                    break
                for two_positions in combinations(range(n_terms), n_twos):
                    if time.time() > deadline:
                        break
                    
                    # For each choice of ±2 positions, try sign patterns
                    # Fix position 0 sign = +1
                    other_positions = [p for p in range(n_terms) if p != 0]
                    # Limit sign patterns to keep tractable
                    max_patterns = 500
                    patterns = list(product([1, -1], repeat=len(other_positions)))
                    if len(patterns) > max_patterns:
                        patterns = patterns[:max_patterns]
                    
                    for signs in patterns:
                        if time.time() > deadline:
                            break
                        coeffs = [0] * 41
                        coeffs[exp_set[0]] = 1
                        sign_idx = 0
                        for pos in other_positions:
                            s = signs[sign_idx]
                            sign_idx += 1
                            if pos in two_positions:
                                coeffs[exp_set[pos]] = 2 * s
                            else:
                                coeffs[exp_set[pos]] = s
                        try_poly(coeffs)
    
    # Final verification and cleanup
    best_poly = trim(best_poly)
    
    if len(best_poly) > 41:
        best_poly = best_poly[:41]
    
    # Verify with exact computation
    if is_valid(best_poly):
        exact_count = square_nonzero_count_exact(best_poly)
        if exact_count < best_count:
            best_count = exact_count
    
    if not is_valid(best_poly):
        return [1] * 12
    
    return best_poly
# EVOLVE-BLOCK-END
