import time
import itertools
from collections import defaultdict

def count_nnz(poly):
    """Count nonzero coefficients in a dict or list representation."""
    if isinstance(poly, dict):
        return sum(1 for v in poly.values() if v != 0)
    return sum(1 for v in poly if v != 0)

def square_poly(coeffs):
    """Compute square of polynomial given as dict {exponent: coeff}."""
    sq = defaultdict(int)
    items = list(coeffs.items())
    for i, (ei, ci) in enumerate(items):
        for j in range(i, len(items)):
            ej, cj = items[j]
            if i == j:
                sq[ei + ej] += ci * cj
            else:
                sq[ei + ej] += 2 * ci * cj
    return sq

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    best_nnz = 21  # seed value
    best_coeffs = [1] * 12
    
    def update_best(coeffs):
        nonlocal best_nnz, best_coeffs
        if len(coeffs) != 12:
            return
        # Check constraints
        if any(c == 0 for c in coeffs):
            return
        if any(abs(c) > 64 for c in coeffs):
            return
        # Compute square
        poly = {}
        for i, c in enumerate(coeffs):
            if c != 0:
                poly[i] = c
        sq = square_poly(poly)
        nnz = sum(1 for v in sq.values() if v != 0)
        if nnz < best_nnz:
            best_nnz = nnz
            best_coeffs = list(coeffs)
    
    # Strategy 1: P = A + x^k * A where A has 6 terms
    # Try various A patterns and k values
    
    # First, try simple structured patterns
    # Pattern: 6-term base A, then shift by k
    
    # Try A as AP with various steps
    for step in range(1, 8):
        for start in range(0, 4):
            exponents_A = [start + i * step for i in range(6)]
            if max(exponents_A) >= 40:
                continue
            for k in range(1, 15):
                exponents_P = exponents_A + [e + k for e in exponents_A]
                if len(set(exponents_P)) != 12:
                    continue
                if max(exponents_P) > 40:
                    continue
                exponents_P.sort()
                
                # Try sign patterns for A (12 signs total, but P = A + x^k A means
                # the second half has same signs as first)
                # Actually P = A + x^k A means coeffs at exponents_A are a_i and
                # coeffs at exponents_A + k are also a_i
                # So we only need 6 signs/magnitudes
                
                # Try all sign combinations for the 6 base terms
                for signs in itertools.product([1, -1], repeat=6):
                    if time.time() > deadline:
                        return best_coeffs
                    coeffs = [0] * 41
                    for i, e in enumerate(exponents_A):
                        coeffs[e] = signs[i]
                        coeffs[e + k] = signs[i]
                    # Trim to actual degree
                    last_nz = max(i for i in range(41) if coeffs[i] != 0)
                    coeffs_trimmed = coeffs[:last_nz + 1]
                    if len(coeffs_trimmed) > 41:
                        continue
                    # Count nonzero
                    nnz = sum(1 for c in coeffs_trimmed if c != 0)
                    if nnz != 12:
                        continue
                    update_best(coeffs_trimmed)
    
    # Strategy 2: Try more general 12-term patterns with clustered exponents
    # Use small exponent sets with high collision density
    
    # Try exponents as two clusters of 6
    for gap in range(1, 20):
        for offset in range(0, 5):
            cluster1 = [offset + i for i in range(6)]
            cluster2 = [offset + gap + i for i in range(6)]
            exponents = sorted(cluster1 + cluster2)
            if len(set(exponents)) != 12 or max(exponents) > 40:
                continue
            
            # Try signs
            for signs in itertools.product([1, -1], repeat=12):
                if time.time() > deadline:
                    return best_coeffs
                coeffs = [0] * 41
                for i, e in enumerate(exponents):
                    coeffs[e] = signs[i]
                last_nz = max(i for i in range(41) if coeffs[i] != 0)
                coeffs_trimmed = coeffs[:last_nz + 1]
                nnz = sum(1 for c in coeffs_trimmed if c != 0)
                if nnz != 12:
                    continue
                update_best(coeffs_trimmed)
    
    # Strategy 3: Try P = A + x^k A with magnitudes other than 1
    for step in range(1, 5):
        for start in range(0, 3):
            exponents_A = [start + i * step for i in range(6)]
            if max(exponents_A) >= 40:
                continue
            for k in range(1, 10):
                exponents_P = sorted(exponents_A + [e + k for e in exponents_A])
                if len(set(exponents_P)) != 12 or max(exponents_P) > 40:
                    continue
                
                # Try a few magnitude patterns
                for mags in [(1,1,1,1,1,1), (1,1,1,1,-1,1), (1,-1,1,-1,1,-1)]:
                    for signs in itertools.product([1, -1], repeat=6):
                        if time.time() > deadline:
                            return best_coeffs
                        coeffs = [0] * 41
                        for i, e in enumerate(exponents_A):
                            val = signs[i] * mags[i]
                            coeffs[e] = val
                            coeffs[e + k] = val
                        last_nz = max(i for i in range(41) if coeffs[i] != 0)
                        coeffs_trimmed = coeffs[:last_nz + 1]
                        nnz = sum(1 for c in coeffs_trimmed if c != 0)
                        if nnz != 12:
                            continue
                        update_best(coeffs_trimmed)
    
    return best_coeffs
