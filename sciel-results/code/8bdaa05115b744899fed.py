"""Sparse squares of twelve-term integer polynomials."""
import time
import itertools
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

def _square_nonzero_count(coeffs):
    """Compute the number of nonzero terms in the exact square of a polynomial given by coeff list."""
    n = len(coeffs)
    sq = [0] * (2 * n - 1)
    for i in range(n):
        if coeffs[i] == 0:
            continue
        ci = coeffs[i]
        for j in range(i, n):
            if coeffs[j] == 0:
                continue
                pass
            # Actually need to be careful: we need all pairs (i,j) with i<=j
            # Let me redo this properly
    # Redo properly
    sq = [0] * (2 * n - 1)
    for i in range(n):
        ci = coeffs[i]
        if ci == 0:
            continue
        for j in range(i, n):
            cj = coeffs[j]
            if cj == 0:
                continue
            prod = ci * cj
            if i == j:
                sq[i + j] += prod
            else:
                sq[i + j] += 2 * prod
    return sum(1 for c in sq if c != 0)


def _build_poly(exponents, magnitudes):
    """Build coefficient list from exponents and magnitudes (same length)."""
    max_exp = max(exponents) if exponents else 0
    coeffs = [0] * (max_exp + 1)
    for e, m in zip(exponents, magnitudes):
        coeffs[e] = m
    return coeffs


def _sign_flip_search(exponents, magnitudes, deadline):
    """Try all sign patterns (2^12 = 4096) to minimize nonzero count in square."""
    n_terms = len(exponents)
    best_count = float('inf')
    best_mags = list(magnitudes)
    
    # 2^12 = 4096 patterns, very fast
    for mask in range(1 << n_terms):
        mags = [0] * n_terms
        for i in range(n_terms):
            sign = -1 if (mask >> i) & 1 else 1
            mags[i] = sign * magnitudes[i]
        coeffs = _build_poly(exponents, mags)
        cnt = _square_nonzero_count(coeffs)
        if cnt < best_count:
            best_count = cnt
            best_mags = mags
        if time.time() > deadline:
            break
    
    return best_count, best_mags


def _search_structured(deadline):
    """Search over structured exponent patterns with sign optimization."""
    best_count = float('inf')
    best_coeffs = None
    best_exponents = None
    best_mags = None
    
    # Strategy: try various exponent sets
    # Key insight: if exponents form a "sum-free" structure where 2a+b = 2c+d only when {a,b}={c,d},
    # then the square has exactly 2*n-1 terms (no cancellation possible except by signs).
    # But we want MINIMIZATION, so we WANT collisions that cancel.
    
    # Approach: try many random exponent sets, optimize signs
    rng = random.Random(42)
    
    # Try structured patterns first
    # Pattern 1: arithmetic progression
    for start in range(0, 5):
        for step in range(1, 5):
            exps = [start + i * step for i in range(12)]
            if max(exps) > 40:
                continue
            for mags in [[1]*12, [1]*11+[2], [2]*12]:
                bc, bm = _sign_flip_search(exps, mags, deadline)
                if bc < best_count:
                    best_count = bc
                    best_mags = bm
                    best_exponents = exps
                    coeffs = _build_poly(exps, bm)
                    best_coeffs = coeffs
                if time.time() > deadline:
                    return best_count, best_coeffs
    
    # Pattern 2: geometric-like (powers of 2) - these have few collisions
    # Actually we want MORE collisions for cancellation
    # Pattern 3: clusters - group terms close together
    for cluster_gap in [1, 2, 3, 4]:
        for n_clusters in [2, 3, 4, 6]:
            per_cluster = 12 // n_clusters
            remainder = 12 % n_clusters
            exps = []
            pos = 0
            for c in range(n_clusters):
                cnt = per_cluster + (1 if c < remainder else 0)
                for i in range(cnt):
                    exps.append(pos + i)
                pos += cluster_gap
            if max(exps) > 40:
                continue
            for mags in [[1]*12, [2]*12]:
                bc, bm = _sign_flip_search(exps, mags, deadline)
                if bc < best_count:
                    best_count = bc
                    best_mags = bm
                    best_exponents = exps
                    best_coeffs = _build_poly(exps, bm)
                if time.time() > deadline:
                    return best_count, best_coeffs
    
    # Pattern 4: symmetric exponents (palindrome-like)
    for offset in range(0, 3):
        exps = []
        # 6 pairs symmetric around center
        center = 15 + offset
        for i in range(6):
            exps.append(center - i)
            exps.append(center + i + 1)
        exps.sort()
        if max(exps) > 40:
            continue
        for mags in [[1]*12, [2]*12]:
            bc, bm = _sign_flip_search(exps, mags, deadline)
            if bc < best_count:
                best_count = bc
                best_mags = bm
                best_exponents = exps
                best_coeffs = _build_poly(exps, bm)
            if time.time() > deadline:
                return best_count, best_coeffs
    
    # Pattern 5: random exponent sets
    for _ in range(2000):
        if time.time() > deadline:
            break
        # Generate 12 distinct exponents in [0, 40]
        exps = sorted(rng.sample(range(0, 41), 12))
        # Use all 1s or all 2s
        mags = [1] * 12
        bc, bm = _sign_flip_search(exps, mags, deadline)
        if bc < best_count:
            best_count = bc
            best_mags = bm
            best_exponents = exps
            best_coeffs = _build_poly(exps, bm)
    
    return best_count, best_coeffs


def _search_three_factor(deadline):
    """
    Model polynomial as product of three 4-term factors with disjoint exponent supports.
    P = A * B * C where A, B, C each have 4 terms.
    P^2 = A^2 * B^2 * C^2
    Each factor's square has at most 7 terms (4 terms -> 7 positions).
    We want the total product to have few nonzero terms.
    
    Actually this is complex. Let me try a simpler variant:
    P = A * B where A has 6 terms and B has 2 terms.
    P^2 = A^2 * B^2
    B^2 has 3 terms. A^2 has up to 11 terms.
    The product could have up to 33 terms but with cancellation potentially fewer.
    
    Let me try: B = x^a + x^b (two terms, both coeff 1)
    B^2 = x^(2a) + 2x^(a+b) + x^(2b)
    P = A * B, P^2 = A^2 * B^2
    For cancellation in A^2 * B^2, we need A^2 to have structure.
    
    Actually, let me try a direct approach:
    P = x^0 + x^1 + x^2 + x^3 + x^4 + x^5 + x^6 + x^7 + x^8 + x^9 + x^10 + x^11
    This is (x^12 - 1)/(x - 1)
    P^2 = sum of terms... let me compute.
    """
    best_count = float('inf')
    best_coeffs = None
    
    # Try P = (1+x+x^2+x^3) * (1+x^4+x^8+x^12)
    # This has 12 terms if all cross terms are distinct
    # Exponents: {0,1,2,3} + {0,4,8,12} = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
    # That's 16 terms, not 12. Need to be more careful.
    
    # Try: P with exponents that create a "perfect square" structure
    # If P = Q^2 where Q has 6 terms, then P^2 = Q^4
    # Q^4 of a 6-term polynomial... 
    # Q = 1+x+x^2+x^3+x^4+x^5, Q^2 has 11 terms, Q^4 has up to 21 terms
    # But with coefficients, maybe fewer nonzero.
    
    # Let me just try specific known good candidates
    candidates = []
    
    # Candidate: geometric series
    for deg in range(12, 20):
        exps = list(range(deg))
        candidates.append((exps, [1]*12))
    
    # Candidate: sparse with gaps
    for gap in [1, 2, 3, 4, 5]:
        exps = [i * gap for i in range(12)]
        if max(exps) <= 40:
            candidates.append((exps, [1]*12))
            candidates.append((exps, [2]*12))
    
    # Candidate: mixed
    exps = [0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14, 15]
    candidates.append((exps, [1]*12))
    exps = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15]
    candidates.append((exps, [1]*12))
    
    for exps, mags in candidates:
        if time.time() > deadline:
            break
        bc, bm = _sign_flip_search(exps, mags, deadline)
        if bc < best_count:
            best_count = bc
            best_coeffs = _build_poly(exps, bm)
    
    return best_count, best_coeffs


def _verify(coeffs):
    """Verify the coefficient list meets all constraints."""
    if len(coeffs) > 41:
        return False
    if len(coeffs) == 0 or coeffs[-1] == 0:
        return False
    if any(c < -64 or c > 64 for c in coeffs):
        return False
    nonzero = sum(1 for c in coeffs if c != 0)
    if nonzero != 12:
        return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, 10s safety margin
    
    best_count = float('inf')
    best_coeffs = None
    
    # First, get a quick baseline
    # The seed returns [1]*12 which has 12 nonzero terms
    # Its square: (1+x+...+x^11)^2
    # Coefficient of x^k in the square is min(k+1, 23-k, 12) for k in 0..22
    # All are positive, so 23 nonzero terms in the square.
    
    baseline = [1] * 12
    baseline_count = _square_nonzero_count(baseline)
    best_count = baseline_count
    best_coeffs = baseline
    
    # Search for better solutions
    # 1. Structured search
    c1, coeff1 = _search_structured(deadline)
    if c1 < best_count:
        best_count = c1
        best_coeffs = coeff1
    
    # 2. Three-factor / specific candidates
    c2, coeff2 = _search_three_factor(deadline)
    if c2 < best_count:
        best_count = c2
        best_coeffs = coeff2
    
    # 3. Additional random search with time remaining
    if time.time() < deadline - 5:
        rng = random.Random(123)
        for _ in range(5000):
            if time.time() > deadline - 5:
                break
            # Random 12 exponents
            exps = sorted(rng.sample(range(0, 41), 12))
            # Random magnitudes from {1, 2, 3, 4}
            mags = [rng.choice([1, 2, 3, 4]) for _ in range(12)]
            # Only try sign search if the magnitude pattern looks promising
            # Quick check: compute collision count
            # For speed, just do sign search (4096 iterations is fast)
            bc, bm = _sign_flip_search(exps, mags, deadline)
            if bc < best_count:
                best_count = bc
                best_coeffs = _build_poly(exps, bm)
    
    # Final verification
    if best_coeffs is not None and _verify(best_coeffs):
        return best_coeffs
    else:
        # Fallback to valid seed
        return [1] * 12

# EVOLVE-BLOCK-END
