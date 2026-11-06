"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill

    def square_nonzero_count(positions, coeffs):
        """Count nonzero terms in exact integer square given positions and coeffs."""
        n = len(positions)
        if n == 0:
            return 0
        # Build square using dictionary for sparse representation
        sq = {}
        for i in range(n):
            for j in range(i, n):
                degree = positions[i] + positions[j]
                prod = coeffs[i] * coeffs[j]
                if i == j:
                    sq[degree] = sq.get(degree, 0) + prod
                else:
                    sq[degree] = sq.get(degree, 0) + 2 * prod
        return sum(1 for v in sq.values() if v != 0)

    def evaluate_poly(positions, coeffs):
        """Evaluate a polynomial given by positions and coeffs, return square nonzero count."""
        return square_nonzero_count(positions, coeffs)

    best_score = float('inf')
    best_positions = None
    best_coeffs = None

    # Generate candidate position sets
    position_sets = []
    
    # Consecutive positions
    for start in range(0, 30):
        positions = list(range(start, start + 12))
        if max(positions) <= 40:
            position_sets.append(positions)
    
    # Arithmetic progressions
    for start in range(0, 30):
        for step in range(1, 8):
            positions = [start + k * step for k in range(12)]
            if max(positions) <= 40:
                position_sets.append(positions)
    
    # Symmetric patterns around center
    for center in range(10, 35):
        for spread in range(1, 15):
            positions = sorted(set([center + d for d in range(-spread, spread + 1)]))
            if len(positions) >= 12:
                positions = positions[:12]
                if max(positions) <= 40:
                    position_sets.append(positions)
    
    # Random clustered positions
    for _ in range(200):
        base = random.randint(0, 25)
        positions = sorted(set(base + random.randint(0, 15) for _ in range(20)))
        if len(positions) >= 12:
            positions = positions[:12]
            if max(positions) <= 40:
                position_sets.append(positions)
    
    # Deduplicate
    position_sets = list(set(tuple(p) for p in position_sets))
    
    # Magnitude patterns to try
    magnitude_patterns = []
    # All ones
    magnitude_patterns.append([1] * 12)
    # Small integers
    for _ in range(50):
        magnitude_patterns.append([random.randint(1, 4) for _ in range(12)])
    # Pairs of equal magnitudes
    for _ in range(50):
        mags = []
        for _ in range(6):
            v = random.randint(1, 8)
            mags.append(v)
            mags.append(v)
        random.shuffle(mags)
        magnitude_patterns.append(mags)
    
    # For each position set, try sign brute force with selected magnitudes
    for pos_tuple in position_sets:
        if time.time() > deadline - 5:
            break
        positions = list(pos_tuple)
        
        for mags in magnitude_patterns:
            if time.time() > deadline - 5:
                break
            
            # Brute force all 2^12 sign assignments
            # Optimize: only need to consider signs, magnitudes are fixed
            # For speed, use the fact that flipping all signs doesn't change the square
            # So we can fix the first sign to +1, giving 2^11 = 2048 cases
            best_local = float('inf')
            best_signs = None
            
            # Fix first coefficient to positive to reduce search space
            for sign_mask in range(1, 1 << 11):  # 2^11 cases, first sign fixed to +
                signs = [1]  # First is always positive
                for i in range(11):
                    signs.append(1 if (sign_mask >> i) & 1 else -1)
                
                coeffs = [mags[i] * signs[i] for i in range(12)]
                score = square_nonzero_count(positions, coeffs)
                
                if score < best_local:
                    best_local = score
                    best_signs = signs[:]
                
                if best_local <= 3:  # Very good, can stop early
                    break
            
            if best_local < best_score:
                best_score = best_local
                best_positions = positions[:]
                best_coeffs = [mags[i] * best_signs[i] for i in range(12)]
                
                if best_score <= 4:
                    # Found something very good, do more refinement
                    break
        
        if best_score <= 4:
            break
    
    # If we haven't found something great, try more random search
    if best_score > 6 and time.time() < deadline - 10:
        for _ in range(500):
            if time.time() > deadline - 5:
                break
            positions = sorted(random.sample(range(41), 12))
            mags = [random.randint(1, 16) for _ in range(12)]
            
            # Try a few random sign patterns
            for _ in range(20):
                signs = [1 if random.random() < 0.5 else -1 for _ in range(12)]
                coeffs = [mags[i] * signs[i] for i in range(12)]
                score = square_nonzero_count(positions, coeffs)
                if score < best_score:
                    best_score = score
                    best_positions = positions[:]
                    best_coeffs = coeffs[:]
    
    # Build the final polynomial
    if best_positions is None:
        # Fallback
        poly = [0] * 41
        poly[0] = 1
        poly[1] = 1
        poly[2] = 1
        poly[3] = 1
        poly[4] = 1
        poly[5] = 1
        poly[6] = 1
        poly[7] = 1
        poly[8] = 1
        poly[9] = 1
        poly[10] = 1
        poly[11] = 1
        return poly[:12]
    
    poly = [0] * 41
    for pos, coeff in zip(best_positions, best_coeffs):
        poly[pos] = coeff
    
    # Trim trailing zeros
    last_nz = -1
    for i in range(len(poly) - 1, -1, -1):
        if poly[i] != 0:
            last_nz = i
            break
    
    if last_nz == -1:
        return [1] + [0] * 11
    
    result = poly[:last_nz + 1]
    
    # Verify constraints
    nz_count = sum(1 for c in result if c != 0)
    if nz_count != 12:
        # Fallback
        poly = [0] * 41
        for i in range(12):
            poly[i] = 1
        return poly[:12]
    
    if result[-1] == 0:
        # Shouldn't happen after trimming, but just in case
        return [1] + [0] * 11
    
    return result
# EVOLVE-BLOCK-END
