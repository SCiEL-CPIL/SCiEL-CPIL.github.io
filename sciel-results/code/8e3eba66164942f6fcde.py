"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product, combinations

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    best_poly = [1] * 12
    best_count = 23  # seed: (1+x+...+x^11)^2 has 23 nonzero terms
    
    def square_nonzero_count_fast(coeff_list):
        """Fast square computation using FFT convolution."""
        n = len(coeff_list)
        if n == 0:
            return 0
        size = 1
        while size < 2 * n - 1:
            size *= 2
        a = np.zeros(size, dtype=np.float64)
        for i, c in enumerate(coeff_list):
            if c != 0:
                a[i] = c
        fa = np.fft.fft(a)
        sq = np.fft.ifft(fa * fa).real
        sq = np.round(sq).astype(np.int64)
        sq = sq[:2 * n - 1]
        return int(np.count_nonzero(sq))
    
    def square_nonzero_count_exact(coeff_list):
        """Exact square computation for small coefficient lists."""
        n = len(coeff_list)
        if n == 0:
            return 0
        sq = [0] * (2 * n - 1)
        nz = [(i, c) for i, c in enumerate(coeff_list) if c != 0]
        for i, ci in nz:
            for j, cj in nz:
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
        if any(c < -64 or c > 64 or c != int(c) for c in poly):
            return False
        return True
    
    # Phase 1: Systematic search over structured exponent sets with ±1 coefficients
    # Key insight: exponent sets with many "sum collisions" (i+j = k+l for distinct pairs)
    # lead to more cancellations in the square
    
    exponent_families = [
        # Consecutive
        list(range(12)),
        # Even spacing
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        # Triple spacing
        [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33],
        # Quadruple spacing
        [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44],  # too large, max degree 40
        # Pairs of consecutive
        [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16],
        # Triplets
        [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14],
        # Alternating gaps
        [0, 1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 15],
        # Fibonacci-like
        [0, 1, 2, 3, 5, 8, 13, 21, 28, 35, 38, 40],
        # Symmetric around center
        [0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23],
        # Power of 2 exponents
        [0, 1, 2, 4, 8, 16, 24, 28, 32, 36, 38, 40],
        # Dense clusters
        [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15],
        # Sparse with structure
        [0, 2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 30],
    ]
    
    # Filter out families that exceed degree 40
    valid_families = [f for f in exponent_families if max(f) < 41]
    
    for exp_set in valid_families:
        if time.time() > deadline:
            break
        # Try all 2^11 sign patterns (fix first coefficient sign = +1)
        for signs in product([1, -1], repeat=11):
            if time.time() > deadline:
                break
            coeffs = [0] * 41
            coeffs[exp_set[0]] = 1
            for i, s in enumerate(signs):
                coeffs[exp_set[i+1]] = s
            cnt = square_nonzero_count_fast(coeffs)
            if cnt < best_count:
                best_count = cnt
                best_poly = trim(coeffs[:])
    
    # Phase 2: Local search - perturb coefficients of best solution
    if time.time() < deadline:
        work_poly = best_poly[:]
        work_count = best_count
        max_degree = len(work_poly)
        
        # Ensure we have a reasonable working length
        while len(work_poly) < 41:
            work_poly.append(0)
        max_degree = len(work_poly)
        
        np.random.seed(42)
        for iteration in range(5000):
            if time.time() > deadline:
                break
            
            # Randomly pick a position to modify
            pos = np.random.randint(0, max_degree)
            old_val = work_poly[pos]
            
            # Try different values
            if old_val == 0:
                new_vals = [1, -1, 2, -2, 3, -3, 4, -4]
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
                if new_val == old_val:
                    continue
                if abs(new_val) > 64:
                    continue
                
                work_poly[pos] = new_val
                
                # Check if still exactly 12 nonzero
                nonzero = sum(1 for c in work_poly if c != 0)
                if nonzero != 12:
                    work_poly[pos] = old_val
                    continue
                
                # Ensure final coefficient is nonzero
                if work_poly[-1] == 0:
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
    
    # Phase 3: Try coefficient values of ±2 in structured ways
    if time.time() < deadline:
        for exp_set in valid_families[:6]:
            if time.time() > deadline:
                break
            
            # Choose 1-3 positions to have coefficient ±2, rest ±1
            n_terms = len(exp_set)
            for n_twos in range(1, 4):
                if time.time() > deadline:
                    break
                for two_positions in combinations(range(1, n_terms), n_twos):  # skip position 0
                    if time.time() > deadline:
                        break
                    
                    # Determine which positions need sign assignment
                    # Position 0 is fixed at +1
                    # Positions in two_positions get ±2
                    # Other positions get ±1
                    sign_positions = []
                    for p in range(1, n_terms):
                        sign_positions.append(p)
                    
                    # Try a subset of sign patterns
                    max_patterns = 500
                    all_patterns = list(product([1, -1], repeat=len(sign_positions)))
                    if len(all_patterns) > max_patterns:
                        all_patterns = all_patterns[:max_patterns]
                    
                    for signs in all_patterns:
                        if time.time() > deadline:
                            break
                        coeffs = [0] * 41
                        coeffs[exp_set[0]] = 1
                        sign_idx = 0
                        for i in range(1, n_terms):
                            s = signs[sign_idx]
                            sign_idx += 1
                            if i in two_positions:
                                coeffs[exp_set[i]] = 2 * s
                            else:
                                coeffs[exp_set[i]] = s
                        
                        cnt = square_nonzero_count_fast(coeffs)
                        if cnt < best_count:
                            best_count = cnt
                            best_poly = trim(coeffs[:])
    
    # Phase 4: Try mixed coefficient patterns with values in {-3,-2,-1,1,2,3}
    if time.time() < deadline:
        # Focus on the best exponent family found so far
        # Try random coefficient assignments
        np.random.seed(123)
        for exp_set in valid_families[:4]:
            if time.time() > deadline:
                break
            n_terms = len(exp_set)
            
            for trial in range(2000):
                if time.time() > deadline:
                    break
                
                # Random coefficient assignment
                coeffs = [0] * 41
                coeffs[exp_set[0]] = 1  # Fix first
                for i in range(1, n_terms):
                    coeffs[exp_set[i]] = np.random.choice([-3, -2, -1, 1, 2, 3])
                
                cnt = square_nonzero_count_fast(coeffs)
                if cnt < best_count:
                    best_count = cnt
                    best_poly = trim(coeffs[:])
    
    # Phase 5: Final local refinement
    if time.time() < deadline:
        work_poly = best_poly[:]
        while len(work_poly) < 41:
            work_poly.append(0)
        work_count = best_count
        max_degree = len(work_poly)
        
        # More aggressive local search with multiple perturbations
        np.random.seed(456)
        for iteration in range(3000):
            if time.time() > deadline:
                break
            
            # Perturb 1-2 positions
            n_perturb = np.random.choice([1, 1, 1, 2])
            positions = np.random.choice(max_degree, size=n_perturb, replace=False)
            old_vals = [work_poly[p] for p in positions]
            
            for new_vals in product([-3, -2, -1, 0, 1, 2, 3], repeat=n_perturb):
                if time.time() > deadline:
                    break
                
                for p, v in zip(positions, new_vals):
                    work_poly[p] = v
                
                # Check constraints
                nonzero = sum(1 for c in work_poly if c != 0)
                if nonzero != 12:
                    for p, v in zip(positions, old_vals):
                        work_poly[p] = v
                    continue
                
                if work_poly[-1] == 0:
                    for p, v in zip(positions, old_vals):
                        work_poly[p] = v
                    continue
                
                if any(abs(c) > 64 for c in work_poly):
                    for p, v in zip(positions, old_vals):
                        work_poly[p] = v
                    continue
                
                cnt = square_nonzero_count_fast(work_poly)
                if cnt < work_count:
                    work_count = cnt
                    if cnt < best_count:
                        best_count = cnt
                        best_poly = trim(work_poly[:])
                else:
                    for p, v in zip(positions, old_vals):
                        work_poly[p] = v
    
    # Final cleanup: ensure valid output
    best_poly = trim(best_poly)
    
    # Ensure length <= 41
    if len(best_poly) > 41:
        best_poly = best_poly[:41]
    
    # Ensure exactly 12 nonzero terms
    nonzero_count = sum(1 for c in best_poly if c != 0)
    if nonzero_count != 12:
        # Fallback to seed
        return [1] * 12
    
    # Ensure entries in [-64, 64]
    for i, c in enumerate(best_poly):
        if c < -64:
            best_poly[i] = -64
        elif c > 64:
            best_poly[i] = 64
        best_poly[i] = int(best_poly[i])
    
    # Final validation
    if not is_valid(best_poly):
        return [1] * 12
    
    return best_poly
# EVOLVE-BLOCK-END
