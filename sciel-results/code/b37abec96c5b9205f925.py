"""
Strategy: Dynamic search over exponent sets and coefficients using FFT acceleration.
Instead of fixing a "Golay" set and only flipping signs, we:
1. Use FFT to evaluate the square's sparsity in O(N log N) instead of O(k^2).
2. Search over exponent sets (combinations from a pool) and coefficient values (±1, ±2, ±3).
3. Use local search (hill climbing) to minimize the number of non-zero terms in the square.
"""

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    deadline = time.time() + 85.0
    best_score = 23
    best_poly = [1] * 12
    
    # Precompute FFT size for max degree 40 (length 41) -> square length 81
    # Next power of 2 >= 81 is 128
    FFT_SIZE = 128
    fft_poly = np.zeros(FFT_SIZE, dtype=np.complex128)

    def count_nonzero_square(poly):
        """
        Fast count of non-zero coefficients in the square of poly using FFT.
        poly: list of ints, length <= 41.
        """
        n = len(poly)
        if n == 0:
            return 0
        
        # Fill FFT array
        fft_poly[:] = 0
        fft_poly[:n] = poly
        
        # FFT, square, IFFT
        f = np.fft.fft(fft_poly)
        sq_f = f * f
        sq = np.fft.ifft(sq_f).real
        
        # Round to nearest integer to handle floating point errors
        sq_int = np.rint(sq).astype(np.int64)
        
        # The square of a polynomial of degree d has degree 2d.
        # Length is 2*d + 1.
        # We only care about indices 0..2*(n-1)
        max_deg = 2 * (n - 1)
        return np.count_nonzero(sq_int[:max_deg + 1])

    def try_local_search(poly, steps=5000):
        """
        Local search on the polynomial coefficients and potentially exponents (if we allow moving terms).
        Here we stick to optimizing coefficients for a fixed support (exponents) first,
        then we will explore changing the support in the main loop.
        """
        curr_score = count_nonzero_square(poly)
        n = len(poly)
        nz_indices = [i for i in range(n) if poly[i] != 0]
        
        for _ in range(steps):
            if time.time() > deadline:
                break
                
            # Randomly pick a non-zero term to modify
            if not nz_indices:
                break
            idx = random.choice(nz_indices)
            old_val = poly[idx]
            
            # Try changing the coefficient to a small integer
            # Possible values: -3, -2, -1, 1, 2, 3 (excluding 0 to maintain 12 terms, 
            # though we could allow 0 and re-add elsewhere, but let's keep support fixed for now)
            candidates = [v for v in [-3, -2, -1, 1, 2, 3] if v != old_val]
            random.shuffle(candidates)
            
            improved = False
            for new_val in candidates:
                poly[idx] = new_val
                new_score = count_nonzero_square(poly)
                if new_score < curr_score:
                    curr_score = new_score
                    improved = True
                    break
                elif new_score == curr_score and random.random() < 0.05:
                    # Accept neutral moves occasionally to escape local minima
                    curr_score = new_score
                    improved = True
                    break
                else:
                    # Revert
                    poly[idx] = old_val
                    break
            
            if not improved:
                poly[idx] = old_val

        return curr_score, poly

    # Strategy: Generate candidate exponent sets and optimize coefficients
    
    # 1. Start with the "Golay-like" set from the seed
    golay_exponents = [0, 1, 2, 4, 7, 8, 11, 12, 14, 16, 19, 20]
    if len(golay_exponents) == 12:
        poly = [0] * (max(golay_exponents) + 1)
        for e in golay_exponents:
            poly[e] = 1
        score, poly = try_local_search(poly, steps=2000)
        if score < best_score:
            best_score = score
            best_poly = poly[:]

    # 2. Randomized search for new exponent sets
    # We want sets with high "collision" in sums to allow cancellation.
    # Small exponents (0..20) create more collisions than large sparse ones.
    
    max_exp_pool = 25
    pool_size = max_exp_pool + 1
    
    # We will try several random configurations
    attempts = 0
    while time.time() < deadline - 5.0:
        attempts += 1
        
        # Generate a random set of 12 exponents from 0..max_exp_pool
        # Ensure 0 is included to keep the constant term non-zero (optional, but good for structure)
        # Actually, the task says "exactly 12 nonzero terms". It doesn't force x^0.
        # But typically sparse squares work best with symmetric or structured sets.
        
        exponents = random.sample(range(pool_size), 12)
        exponents.sort()
        
        # Initialize coefficients with random small integers
        poly = [0] * (max(exponents) + 1)
        for e in exponents:
            poly[e] = random.choice([1, -1, 2, -2])
            
        # Run local search
        score, poly = try_local_search(poly, steps=1000)
        
        if score < best_score:
            best_score = score
            best_poly = poly[:]
            # If we found something very good (<= 14), we might want to refine it more
            if best_score <= 14:
                # Do a deeper refinement
                score2, poly2 = try_local_search(best_poly, steps=5000)
                if score2 < best_score:
                    best_score = score2
                    best_poly = poly2

        # Also try perturbing the best solution found so far
        if attempts % 5 == 0 and best_poly is not None:
            perturbed = best_poly[:]
            n = len(perturbed)
            # Change one coefficient
            nz = [i for i in range(n) if perturbed[i] != 0]
            if nz:
                idx = random.choice(nz)
                old = perturbed[idx]
                perturbed[idx] = random.choice([v for v in [-3, -2, -1, 1, 2, 3] if v != old])
                score_p, poly_p = try_local_search(perturbed, steps=500)
                if score_p < best_score:
                    best_score = score_p
                    best_poly = poly_p

    # Final validation and cleanup
    if best_poly is None or len(best_poly) == 0:
        return [1] * 12
        
    # Trim trailing zeros
    while best_poly and best_poly[-1] == 0:
        best_poly.pop()
        
    if not best_poly:
        return [1] * 12

    # Check constraints
    if len(best_poly) > 41:
        # If it's too long, we failed the constraint. Return seed fallback.
        return [1] * 12
        
    nz_count = sum(1 for c in best_poly if c != 0)
    if nz_count != 12:
        return [1] * 12
        
    # Check coefficient bounds
    for c in best_poly:
        if c < -64 or c > 64:
            return [1] * 12

    return best_poly
# EVOLVE-BLOCK-END
