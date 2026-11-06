"""Sparse squares of twelve-term integer polynomials."""
import time
import itertools
import random
from collections import defaultdict

_DEADLINE = 85.0  # safety margin before 90s kill

def _square_nonzero_count(coeffs):
    """Given coefficient list (ascending degree), return number of nonzero terms in the square."""
    n = len(coeffs)
    if n == 0:
        return 0
    # Square: result degree up to 2*(n-1)
    sq = [0] * (2 * n - 1)
    for i in range(n):
        ai = coeffs[i]
        if ai == 0:
            continue
        for j in range(n):
            aj = coeffs[j]
            if aj == 0:
                continue
            sq[i + j] += ai * aj
    return sum(1 for c in sq if c != 0)

def _validate(coeffs):
    """Check all constraints: len<=41, nonzero final, entries in [-64,64], exactly 12 nonzero."""
    if len(coeffs) > 41:
        return False
    if len(coeffs) == 0 or coeffs[-1] == 0:
        return False
    if any(c < -64 or c > 64 for c in coeffs):
        return False
    if sum(1 for c in coeffs if c != 0) != 12:
        return False
    return True

def _search_exponents(exponents, deadline, rng, best_count, best_coeffs):
    """Search over sign patterns and coefficient magnitudes for a given exponent set."""
    n = len(exponents)
    if n != 12:
        return best_count, best_coeffs
    
    # Precompute degree pairs: for each degree d, which (i,j) pairs contribute
    # degree d gets contributions from pairs (i,j) where exponents[i]+exponents[j]==d
    # We need to find sign/magnitude assignments to minimize nonzero count in square
    
    # First, determine which degrees are "diagonal" (i==j) vs "cross"
    # Diagonal degrees: 2*exponents[i] for each i
    # Cross degrees: exponents[i]+exponents[j] for i!=j
    
    # For each degree d in the square, the coefficient is:
    #   sum over all (i,j) with e_i+e_j=d of a_i*a_j
    # This includes diagonal terms 2*a_i^2 (when i==j) and cross terms a_i*a_j (when i!=j)
    
    # Strategy: try all 2^12 sign patterns, and for magnitudes try a few values
    # For each sign pattern, try to find magnitudes that create cancellations
    
    # Key observation: if we fix signs, the square coefficients are quadratic forms in magnitudes
    # We can try small integer magnitudes: 1, 2, 3, 4
    
    # For efficiency, precompute the structure
    # For each degree d, list the (i,j) pairs
    deg_to_pairs = defaultdict(list)
    for i in range(n):
        for j in range(n):
            d = exponents[i] + exponents[j]
            deg_to_pairs[d].append((i, j))
    
    # Sort degrees
    all_degrees = sorted(deg_to_pairs.keys())
    
    # For each sign pattern (2^12 = 4096), try a few magnitude patterns
    # Magnitude patterns: all 1s, or a few variations
    sign_count = 1 << n
    
    # Precompute: for each sign pattern s and magnitude pattern m, compute square nonzero count
    # To speed up, use the fact that we only need to check a few magnitude patterns
    
    # Try: all magnitudes = 1 (sign pattern determines everything)
    # Then: some magnitudes = 2, others = 1
    # Then: mix of 1, 2, 3
    
    best_local = best_count
    best_local_coeffs = best_coeffs
    
    # Phase 1: all magnitudes = 1, vary signs
    for s in range(sign_count):
        if time.time() > deadline:
            break
        signs = [1 if (s >> i) & 1 else -1 for i in range(n)]
        coeffs = [0] * (max(exponents) + 1)
        for i, e in enumerate(exponents):
            coeffs[e] = signs[i]
        cnt = _square_nonzero_count(coeffs)
        if cnt < best_local:
            best_local = cnt
            best_local_coeffs = coeffs[:]
    
    # Phase 2: try some magnitude variations with best sign patterns
    # Get top sign patterns from phase 1
    # Actually, let's try a few structured magnitude patterns with all sign patterns
    
    # Magnitude pattern: first k terms have magnitude 2, rest have magnitude 1
    # Or: terms at even positions have magnitude 2
    for mag_pattern_name in ["all2", "half2", "alt2"]:
        if time.time() > deadline:
            break
        if mag_pattern_name == "all2":
            mags = [2] * n
        elif mag_pattern_name == "half2":
            mags = [2] * 6 + [1] * 6
        elif mag_pattern_name == "alt2":
            mags = [2 if i % 2 == 0 else 1 for i in range(n)]
        
        for s in range(sign_count):
            if time.time() > deadline:
                break
            signs = [1 if (s >> i) & 1 else -1 for i in range(n)]
            coeffs = [0] * (max(exponents) + 1)
            for i, e in enumerate(exponents):
                coeffs[e] = signs[i] * mags[i]
            if any(c < -64 or c > 64 for c in coeffs):
                continue
            cnt = _square_nonzero_count(coeffs)
            if cnt < best_local:
                best_local = cnt
                best_local_coeffs = coeffs[:]
    
    # Phase 3: try magnitude patterns where specific positions get larger coefficients
    # to create targeted cancellations
    for s in range(min(sign_count, 512)):
        if time.time() > deadline:
            break
        signs = [1 if (s >> i) & 1 else -1 for i in range(n)]
        # Try: make coefficients at positions that share cross-degree sums have matching magnitudes
        # Simple heuristic: try magnitudes [1,1,1,1,1,1,2,2,2,2,2,2] and permutations
        for perm_seed in range(3):
            mags = [1] * n
            # Set some to 2
            for i in range(6):
                mags[(i + perm_seed * 2) % n] = 2
            coeffs = [0] * (max(exponents) + 1)
            for i, e in enumerate(exponents):
                coeffs[e] = signs[i] * mags[i]
            if any(c < -64 or c > 64 for c in coeffs):
                continue
            cnt = _square_nonzero_count(coeffs)
            if cnt < best_local:
                best_local = cnt
                best_local_coeffs = coeffs[:]
    
    return best_local, best_local_coeffs

def _generate_exponent_sets(rng, count=200):
    """Generate diverse exponent sets of 12 elements in [0, 40]."""
    sets = []
    
    # Arithmetic progressions
    for start in range(0, 5):
        for step in range(1, 4):
            exponents = [start + i * step for i in range(12)]
            if max(exponents) <= 40:
                sets.append(exponents)
    
    # Two arithmetic progressions (interleaved)
    for s1 in range(0, 3):
        for st1 in range(1, 3):
            for s2 in range(3, 6):
                for st2 in range(1, 3):
                    e1 = [s1 + i * st1 for i in range(6)]
                    e2 = [s2 + i * st2 for i in range(6)]
                    exponents = sorted(e1 + e2)
                    if len(set(exponents)) == 12 and max(exponents) <= 40:
                        sets.append(exponents)
    
    # Geometric-like
    for base in [2, 3]:
        exponents = [base ** i for i in range(12)]
        if max(exponents) <= 40:
            sets.append(exponents)
    
    # Random sets
    for _ in range(count):
        exponents = sorted(rng.sample(range(0, 41), 12))
        sets.append(exponents)
    
    # Specific structured sets known to create many cross-sum collisions
    # e.g., exponents where many pairs sum to the same value
    # 0,1,2,3,4,5, 10,11,12,13,14,15 -> cross sums overlap with diagonal sums
    sets.append([0,1,2,3,4,5,10,11,12,13,14,15])
    sets.append([0,1,2,3,4,5,6,7,8,9,10,11])
    sets.append([0,2,4,6,8,10,12,14,16,18,20,22])
    sets.append([0,1,2,3,4,5, 20,21,22,23,24,25])
    sets.append([0,1,3,4,6,7, 12,13,15,16,18,19])
    sets.append([0,2,4,6,8,10, 20,22,24,26,28,30])
    sets.append([0,1,2,4,5,6, 12,13,14,16,17,18])
    
    return sets

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    rng = random.Random(42)
    
    best_count = 999
    best_coeffs = None
    
    # Generate exponent sets
    exp_sets = _generate_exponent_sets(rng, count=300)
    
    # Shuffle for diversity
    rng.shuffle(exp_sets)
    
    # Search over exponent sets
    for exponents in exp_sets:
        if time.time() > deadline:
            break
        if max(exponents) > 40:
            continue
        if len(set(exponents)) != 12:
            continue
        
        # Quick feasibility: count distinct sums to estimate minimum nonzero terms
        # The square has nonzero terms at degrees where the sum doesn't fully cancel
        # Minimum possible = number of distinct diagonal degrees (12) if all cross terms cancel
        # But cross terms at diagonal degrees add to diagonal, so those can't cancel the diagonal
        # Minimum is at least 12 (the 12 diagonal contributions)
        
        best_count, best_coeffs = _search_exponents(
            exponents, deadline, rng, best_count, best_coeffs
        )
        
        # If we've found the theoretical minimum (12), we can stop
        if best_count <= 12:
            break
    
    # If we haven't found anything good, use a default
    if best_coeffs is None or not _validate(best_coeffs):
        # Fallback: simple 12-term polynomial
        best_coeffs = [1] * 12
        best_count = _square_nonzero_count(best_coeffs)
    
    # Ensure validity
    if not _validate(best_coeffs):
        best_coeffs = [1] * 12
    
    return best_coeffs
