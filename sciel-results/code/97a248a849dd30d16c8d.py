"""Sparse squares of twelve-term integer polynomials."""
import time
from itertools import combinations, product

# EVOLVE-BLOCK-START
def construct():
    """
    Return coefficient list in ascending degree, length<=41, nonzero final coefficient,
    entries integers in [-64,64], exactly 12 nonzero terms.
    Minimize nonzero terms in the EXACT integer square.
    """
    start_time = time.time()
    DEADLINE = 80  # safety margin before 90s limit
    
    best_poly = None
    best_sq_nz = float('inf')
    
    def count_nz_square(coeffs):
        """Count nonzero coefficients in the square of the polynomial."""
        n = len(coeffs)
        sq_len = 2 * n - 1
        sq = [0] * sq_len
        for i in range(n):
            if coeffs[i] == 0:
                continue
            for j in range(n):
                if coeffs[j] == 0:
                    continue
                sq[i + j] += coeffs[i] * coeffs[j]
        return sum(1 for c in sq if c != 0)
    
    def validate(coeffs):
        """Check all constraints."""
        if len(coeffs) > 41:
            return False
        if len(coeffs) == 0 or coeffs[-1] == 0:
            return False
        nz_count = sum(1 for c in coeffs if c != 0)
        if nz_count != 12:
            return False
        for c in coeffs:
            if not (-64 <= c <= 64):
                return False
            if not isinstance(c, int):
                return False
        return True
    
    def trim_trailing_zeros(coeffs):
        """Remove trailing zeros but keep at least one element."""
        while len(coeffs) > 1 and coeffs[-1] == 0:
            coeffs.pop()
        return coeffs
    
    # Strategy: Search for polynomials with structure that causes cancellations in the square.
    # Key insight: If P(x) = Q(x)^2 * R(x) where R has special structure, or if P has 
    # symmetric coefficient patterns, the square may have fewer nonzero terms.
    
    # Approach 1: Try polynomials of form a*x^k + b*x^m + c*x^n + ... where exponents
    # are chosen so that pairwise sums collide (i+j = k+l for different pairs).
    
    # Approach 2: Use the fact that if P(x) = (sum of terms)^2, then P^2 = (sum)^4,
    # but we want P to have 12 terms and P^2 to have few terms.
    
    # Better: Directly search over sparse exponent sets with small coefficient values.
    
    # For a polynomial with 12 nonzero terms at exponents e_1 < e_2 < ... < e_12,
    # the square has terms at exponents e_i + e_j for all i <= j (with cross terms doubled).
    # To minimize nonzero terms in the square, we want many collisions: e_i + e_j = e_k + e_l.
    
    # This is related to additive combinatorics: we want a set with many additive relations.
    
    # Let's try geometric progressions or arithmetic progressions of exponents.
    
    # Arithmetic progression: exponents = a, a+d, a+2d, ..., a+11d
    # Square exponents: 2a, 2a+d, 2a+2d, ..., 2a+22d
    # All distinct -> 23 nonzero terms in square. Not great.
    
    # Try: exponents in two clusters to create collisions.
    # E.g., {0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15}
    # Sums from first cluster: 0..10 (11 values)
    # Sums from second cluster: 20..30 (11 values)
    # Cross sums: 10..19 (10 values)
    # Total: need to check for overlaps.
    
    # Let me try a different structure: exponents that form a "Sidon-like" set but with deliberate collisions.
    
    # Actually, let's try: P(x) = (1 + x + x^2 + x^3 + x^4 + x^5)^2
    # This has 11 terms in the square (coefficients 1,2,3,4,5,6,5,4,3,2,1) - all nonzero!
    # But P itself has 6 terms, not 12.
    
    # What if P(x) = (1 + x + x^2 + x^3 + x^4 + x^5) * (1 + x^6)?
    # P has 12 terms. P^2 = (1+x+...+x^5)^2 * (1+x^6)^2
    # = (1+2x+3x^2+4x^3+5x^4+6x^5+5x^6+4x^7+3x^8+2x^9+x^10) * (1+2x^6+x^12)
    # Let's compute this...
    
    # Let me try a more systematic search.
    
    # Try polynomials of the form: sum of two "blocks"
    # Block 1: terms at exponents 0..k-1 with coefficients
    # Block 2: terms at exponents m..m+k-1 with coefficients
    
    # For simplicity, start with all coefficients = 1.
    
    # Try: exponents = [0,1,2,3,4,5, d, d+1, d+2, d+3, d+4, d+5] for various d
    # The square will have:
    # - From block 1: exponents 0..10
    # - From block 2: exponents 2d..2d+10
    # - Cross: exponents d..d+10 and d+1..d+11 (overlapping)
    
    # If d=10: block1: 0..10, block2: 20..30, cross: 10..20
    # Overlap at 10 and 20.
    # Total distinct exponents: 0..30 = 31 terms. Still a lot.
    
    # Try d=6: block1: 0..10, block2: 12..22, cross: 6..16
    # Union: 0..22 = 23 terms.
    
    # This approach gives too many terms. Let me think differently.
    
    # What if we use coefficients that cause CANCELLATION in the square?
    # If P(x) = a*x^i + b*x^j + c*x^k, then P^2 has term at i+j with coefficient 2ab.
    # If we choose a, b, c such that some coefficients cancel...
    
    # Example: P(x) = 1 - x + x^2. P^2 = 1 - 2x + 3x^2 - 2x^3 + x^4. All nonzero.
    # P(x) = 1 - x. P^2 = 1 - 2x + x^2. All nonzero.
    
    # For cancellation, we need multiple pairs (i,j) and (k,l) with i+j = k+l 
    # and a_i*a_j + a_k*a_l = 0 (for cross terms) or similar.
    
    # Let me try a small systematic search over coefficient patterns.
    
    # Try: P(x) = sum_{i=0}^{11} a_i * x^{e_i} where e_i are chosen and a_i in {-1, 0, 1}
    # or {-1, 1} with exactly 12 nonzero.
    
    # Due to time constraints, let me try a few structured candidates first.
    
    candidates = []
    
    # Candidate 1: All ones at exponents 0-11
    c1 = [1] * 12
    candidates.append(c1)
    
    # Candidate 2: Alternating signs
    c2 = [1 if i % 2 == 0 else -1 for i in range(12)]
    candidates.append(c2)
    
    # Candidate 3: Exponents with collisions
    # Try exponents: 0,1,2,3,4,5,10,11,12,13,14,15
    c3 = [0] * 16
    for e in [0,1,2,3,4,5,10,11,12,13,14,15]:
        c3[e] = 1
    trim_trailing_zeros(c3)
    candidates.append(c3)
    
    # Candidate 4: Exponents in arithmetic progression with ratio 2
    c4 = [0] * 23
    for i in range(12):
        c4[2*i] = 1
    trim_trailing_zeros(c4)
    candidates.append(c4)
    
    # Candidate 5: Try to create more collisions
    # Exponents: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 but with some coefficients -1
    # to cause cancellations
    
    # Let's do a small random search
    import random
    random.seed(42)
    
    for trial in range(50000):
        if time.time() - start_time > DEADLINE:
            break
        
        # Random sparse polynomial with 12 nonzero terms
        max_deg = 40
        # Choose 12 distinct exponents
        exponents = sorted(random.sample(range(max_deg + 1), 12))
        
        # Choose coefficients from {-1, 1, 2, -2}
        coeffs_list = []
        for e in exponents:
            c = random.choice([-1, 1, 2, -2])
            coeffs_list.append((e, c))
        
        # Build coefficient array
        poly = [0] * (max_deg + 1)
        for e, c in coeffs_list:
            poly[e] = c
        trim_trailing_zeros(poly)
        
        if not validate(poly):
            continue
        
        sq_nz = count_nz_square(poly)
        if sq_nz < best_sq_nz:
            best_sq_nz = sq_nz
            best_poly = poly[:]
    
    # Also try the structured candidates
    for cand in candidates:
        if not validate(cand):
            continue
        sq_nz = count_nz_square(cand)
        if sq_nz < best_sq_nz:
            best_sq_nz = sq_nz
            best_poly = cand[:]
    
    # Try a more targeted search: exponents in two clusters with specific spacing
    if time.time() - start_time < DEADLINE:
        for d in range(3, 15):
            if time.time() - start_time > DEADLINE:
                break
            # Two clusters of 6 terms each
            exponents = list(range(6)) + list(range(d, d + 6))
            if max(exponents) > 40:
                continue
            poly = [0] * (max(exponents) + 1)
            for e in exponents:
                poly[e] = 1
            trim_trailing_zeros(poly)
            if validate(poly):
                sq_nz = count_nz_square(poly)
                if sq_nz < best_sq_nz:
                    best_sq_nz = sq_nz
                    best_poly = poly[:]
    
    # Try with some -1 coefficients to induce cancellation
    if time.time() - start_time < DEADLINE:
        for d in range(3, 15):
            if time.time() - start_time > DEADLINE:
                break
            exponents = list(range(6)) + list(range(d, d + 6))
            if max(exponents) > 40:
                continue
            # Try different sign patterns
            for signs in [
                [1]*6 + [1]*6,
                [1]*6 + [-1]*6,
                [1,1,1,-1,-1,-1] + [1,1,1,-1,-1,-1],
                [1,-1,1,-1,1,-1] + [1,-1,1,-1,1,-1],
                [1]*3 + [-1]*3 + [1]*3 + [-1]*3,
            ]:
                poly = [0] * (max(exponents) + 1)
                for i, e in enumerate(exponents):
                    poly[e] = signs[i]
                trim_trailing_zeros(poly)
                if validate(poly):
                    sq_nz = count_nz_square(poly)
                    if sq_nz < best_sq_nz:
                        best_sq_nz = sq_nz
                        best_poly = poly[:]
    
    # If we found a good candidate, return it; otherwise return the seed
    if best_poly is not None and best_sq_nz < 12:  # 12 is the seed's square nonzero count? Let me check
        return best_poly
    
    # The seed [1]*12 has square with coefficients: 1,2,3,...,12,11,...,2,1 -> all 23 nonzero
    # So any improvement over 23 is good.
    
    if best_poly is not None:
        return best_poly
    
    # Fallback: return the seed
    return [1] * 12

# EVOLVE-BLOCK-END
