"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START
import sys

def _square_nonzero_count(coeffs):
    """Compute the number of nonzero terms in the exact square of a polynomial.
    coeffs is a list of integers in ascending degree order."""
    n = len(coeffs)
    # Convolution: square of polynomial
    # Use numpy for speed
    arr = np.array(coeffs, dtype=np.int64)
    sq = np.convolve(arr, arr)
    return int(np.count_nonzero(sq))

def _validate(coeffs):
    """Check all constraints."""
    if len(coeffs) > 41:
        return False
    if len(coeffs) == 0:
        return False
    if coeffs[-1] == 0:
        return False
    nz = sum(1 for c in coeffs if c != 0)
    if nz != 12:
        return False
    for c in coeffs:
        if c < -64 or c > 64:
            return False
    return True

def _generate_structural_candidate(rng, max_degree=40):
    """Generate a candidate using structural algebraic forms known to have sparse squares.
    
    Key insight: (a*x^i + b*x^j)^2 = a^2*x^(2i) + 2ab*x^(i+j) + b^2*x^(2j)
    has only 3 nonzero terms. Products of such binomials can create more terms
    but with controlled sparsity.
    
    We build P as a sum of shifted binomials: P = sum_k (a_k * x^(s_k) + b_k * x^(t_k))
    where the shifts are chosen to minimize overlap in P^2.
    """
    max_deg = max_degree
    # Start with a base structure
    # Strategy: use products of (1 + c*x^d) type factors, expanded
    # Or: sum of few shifted monomials with specific coefficient relations
    
    # Method 1: Sum of shifted binomials with disjoint support in square
    n_terms = 12
    # Pick 12 exponents and coefficients
    # Try to make P^2 sparse by ensuring cross terms cancel or coincide
    
    # Approach: P = sum_{i=1}^{6} a_i * (x^{u_i} + x^{v_i})
    # This gives 12 terms. P^2 has terms from a_i^2*(x^{2u_i} + 2x^{u_i+v_i} + x^{2v_i})
    # plus cross terms 2*a_i*a_j*(x^{u_i+u_j} + x^{u_i+v_j} + x^{v_i+u_j} + x^{v_i+v_j})
    
    # To minimize nonzero terms in P^2, we want many of these exponents to coincide
    # and coefficients to cancel.
    
    # Simpler: try P = (x^a + x^b + x^c)^2 pattern but that's 9 terms...
    # Let's try: P = sum of 12 monomials where exponents are chosen so that
    # the 210 pairwise sums (with replacement, 12*13/2 = 78 unique pairs) 
    # map to few distinct values.
    
    # Use a geometric progression for exponents: e_i = i * gap
    # Then e_i + e_j = (i+j)*gap, so many collisions.
    # But we need exactly 12 nonzero terms in P.
    
    # Let's try exponents: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 (consecutive)
    # P^2 has exponents 0..22, all potentially nonzero -> 23 terms. Not great.
    
    # Better: use sparse exponents. E.g., powers of 2: 0,1,2,4,8,16,32,...
    # But max degree 40.
    
    # Let's try: exponents = [0, 1, 3, 7, 15, 31] (Mersenne-like)
    # P has 6 terms, need 12. Duplicate with different coefficients?
    # No, must have exactly 12 nonzero terms.
    
    # New approach: P = A(x) * B(x) where A and B are sparse,
    # and P^2 = A^2 * B^2. If A^2 and B^2 are sparse, P^2 is sparse.
    # A with 4 terms, B with 3 terms -> P has up to 12 terms.
    # A^2 has up to 10 terms, B^2 has up to 7 terms, P^2 has up to 70 terms.
    # Not necessarily better.
    
    # Best structural idea: P = (1 + x + x^2 + ... ) type with specific coefficients
    # that cause cancellation in P^2.
    
    # Let's try: P(x) = sum_{i=0}^{11} c_i * x^{e_i}
    # where e_i are chosen from a small set to maximize collisions in e_i + e_j.
    
    # Use exponents from {0, 1, 2} with 4 terms each:
    # e = [0,0,0,0, 1,1,1,1, 2,2,2,2] but that's not a valid polynomial (multiple same exponent)
    # We need distinct exponents for a standard polynomial representation.
    
    # OK, let's just do a structured random search with bias toward:
    # 1. Small coefficient values (1, -1, 2, -2)
    # 2. Exponents that are multiples of a common gap
    # 3. Symmetric patterns
    
    gap = rng.choice([1, 2, 3, 4, 5])
    # Generate 12 distinct exponents
    base = rng.randint(0, 5)
    exponents = []
    used = set()
    for _ in range(12):
        e = base + gap * rng.randint(0, max_deg // gap)
        while e in used or e > max_deg:
            e = base + gap * rng.randint(0, max_deg // gap)
            if e > max_deg:
                e = rng.randint(0, max_deg)
        if e not in used:
            used.add(e)
            exponents.append(e)
    
    exponents.sort()
    coeffs = [0] * (exponents[-1] + 1)
    for e in exponents:
        coeffs[e] = rng.choice([1, -1, 1, -1, 2, -2, 1, -1])
    
    return coeffs

def _local_search(coeffs, rng, time_limit, max_iter=5000):
    """Local search: flip signs or move exponents to reduce square sparsity."""
    current = list(coeffs)
    current_sq = _square_nonzero_count(current)
    best = list(current)
    best_sq = current_sq
    
    n = len(current)
    max_degree = n - 1
    
    start_time = time.time()
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Random move
        move_type = rng.random()
        
        if move_type < 0.4:
            # Flip sign of a random nonzero coefficient
            nz_indices = [i for i in range(n) if current[i] != 0]
            if not nz_indices:
                break
            idx = rng.choice(nz_indices)
            old_val = current[idx]
            current[idx] = -old_val
            new_sq = _square_nonzero_count(current)
            if new_sq < best_sq:
                best_sq = new_sq
                best = list(current)
            elif new_sq >= current_sq and rng.random() > 0.1:
                current[idx] = old_val  # revert
        elif move_type < 0.7:
            # Change coefficient value
            nz_indices = [i for i in range(n) if current[i] != 0]
            if not nz_indices:
                break
            idx = rng.choice(nz_indices)
            old_val = current[idx]
            new_val = rng.choice([1, -1, 2, -2, 3, -3, 4, -4])
            current[idx] = new_val
            new_sq = _square_nonzero_count(current)
            if new_sq < best_sq:
                best_sq = new_sq
                best = list(current)
            elif new_sq >= current_sq and rng.random() > 0.1:
                current[idx] = old_val
        else:
            # Move a term to a new exponent
            nz_indices = [i for i in range(n) if current[i] != 0]
            if len(nz_indices) < 2:
                break
            idx = rng.choice(nz_indices)
            val = current[idx]
            # Choose new exponent
            new_e = rng.randint(0, max_degree)
            # Ensure no collision
            if new_e != idx and current[new_e] == 0:
                current[idx] = 0
                current[new_e] = val
                new_sq = _square_nonzero_count(current)
                if new_sq < best_sq:
                    best_sq = new_sq
                    best = list(current)
                elif new_sq >= current_sq and rng.random() > 0.15:
                    current[new_e] = 0
                    current[idx] = val
            else:
                # Try swapping
                other_nz = [i for i in nz_indices if i != idx]
                if other_nz:
                    other = rng.choice(other_nz)
                    current[idx], current[other] = current[other], current[idx]
                    new_sq = _square_nonzero_count(current)
                    if new_sq < best_sq:
                        best_sq = new_sq
                        best = list(current)
                    elif rng.random() > 0.2:
                        current[idx], current[other] = current[other], current[idx]
    
    return best, best_sq

def _try_algebraic_construction(rng):
    """Try specific algebraic constructions known to have sparse squares.
    
    Key: P(x) = (x^a + x^b + x^c + x^d)(x^e + x^f + x^g)
    If chosen well, P^2 = (x^a+x^b+x^c+x^d)^2 * (x^e+x^f+x^g)^2
    Each factor squared has few terms, and their product may be sparse.
    
    Better: P(x) = x^k * (1 + x + x^2 + x^3 + x^4 + x^5 + x^6 + x^7 + x^8 + x^9 + x^10 + x^11)
    P^2 has 23 terms. Not sparse enough.
    
    Try: P = (1+x)(1+x^2)(1+x^4)(1+x^8)(1+x^16) * c
    This has 32 terms, too many.
    
    Try: P = (1+x+x^2)(1+x^3+x^6)(1+x^9)
    = (1+x+x^2+x^3+x^4+x^5+x^6+x^7+x^8+x^9+x^10+x^11)
    12 terms! P^2 has 23 terms.
    
    Try: P = (1+x+x^2+x^3)(1+x^4+x^8)
    = 1+x+x^2+x^3+x^4+x^5+x^6+x^7+x^8+x^9+x^10+x^11
    Same thing.
    
    The issue is that consecutive exponents give 23 terms in the square.
    
    What if we use: P = (1+x^2+x^4+x^6)(1+x^8+x^16)
    = 1+x^2+x^4+x^6+x^8+x^10+x^12+x^14+x^16+x^18+x^20+x^22
    12 terms. P^2: exponents are even, 0 to 44, step 2 -> 23 terms.
    
    P = (1+x^3+x^6+x^9)(1+x^12+x^24)
    = 1+x^3+x^6+x^9+x^12+x^15+x^18+x^21+x^24+x^27+x^30+x^33
    12 terms. P^2: exponents multiples of 3, 0 to 66 -> 23 terms.
    
    It seems hard to beat 23 with this structure.
    
    What about: P = sum of 12 terms where exponents come in pairs that sum to the same value?
    E.g., exponents: 0, 10, 1, 9, 2, 8, 3, 7, 4, 6, 5, 11
    Then e_i + e_j for i+j pairing... not straightforward.
    
    Let's try: P = a0 + a1*x + a2*x^2 + ... + a11*x^11
    with a_i = a_{11-i} (palindromic). Then P is self-reciprocal.
    P^2 is also palindromic. Doesn't directly help sparsity.
    
    Try specific coefficient patterns:
    P = 1 - x + x^2 - x^3 + x^4 - x^5 + x^6 - x^7 + x^8 - x^9 + x^10 - x^11
    = (1 - x^12)/(1+x)
    P^2 = (1-x^12)^2/(1+x)^2 = (1 - 2x^12 + x^24)/(1+2x+x^2)
    This is a rational function, not a polynomial. Let me compute directly.
    P^2 = sum... this will have many terms.
    
    Actually, let's just compute: (1-x+x^2-...+x^10-x^11)^2
    The square of an alternating sum... let me think.
    (sum_{k=0}^{11} (-1)^k x^k)^2 = sum_{i,j} (-1)^{i+j} x^{i+j}
    Coefficient of x^m = sum_{i+j=m} (-1)^{i+j} = sum_{i+j=m} 1 = number of pairs (i,j) with i+j=m, 0<=i,j<=11
    For m=0: 1, m=1: 2, ..., m=11: 12, m=12: 11, ..., m=22: 1
    All positive! So P^2 = 1+2x+3x^2+...+12x^11+11x^12+...+x^22
    23 nonzero terms.
    
    What if we use: P = 1 + x + x^2 + ... + x^11 (all +1)?
    Same result, 23 terms.
    
    To get fewer, we need cancellations. 
    P = 1 + x - x^2 - x^3 + x^4 + x^5 - x^6 - x^7 + x^8 + x^9 - x^10 - x^11
    Grouped: (1+x)(1-x^2)(1+x^4)(1-x^6)... no.
    P = (1+x)(1-x^2)(1+x^4)(1-x^6)? 
    (1+x)(1-x^2) = 1+x-x^2-x^3
    (1+x^4)(1-x^6) = 1+x^4-x^6-x^10
    Product: (1+x-x^2-x^3)(1+x^4-x^6-x^10)
    = 1+x-x^2-x^3 + x^4+x^5-x^6-x^7 - x^6-x^7+x^8+x^9 - x^10-x^11+x^12+x^13
    = 1+x-x^2-x^3+x^4+x^5-2x^6-2x^7+x^8+x^9-x^10-x^11+x^12+x^13
    14 terms, too many.
    
    Let me just go with the search approach.
    """
    return None

def construct():
    """Construct a 12-term polynomial with sparse square."""
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline, 5s safety margin
    
    rng = random.Random(42)
    
    best_coeffs = None
    best_sq_count = float('inf')
    
    # Phase 1: Try structural candidates
    for attempt in range(200):
        if time.time() > deadline - 30:
            break
        coeffs = _generate_structural_candidate(rng)
        if not _validate(coeffs):
            continue
        sq_count = _square_nonzero_count(coeffs)
        if sq_count < best_sq_count:
            best_sq_count = sq_count
            best_coeffs = list(coeffs)
    
    # Phase 2: Local search from best
    if best_coeffs is not None:
        remaining = deadline - time.time()
        if remaining > 2:
            best_coeffs, best_sq_count = _local_search(
                best_coeffs, rng, time_limit=max(1, remaining - 5), max_iter=20000
            )
    
    # Phase 3: More random restarts with local search
    while time.time() < deadline - 10:
        coeffs = _generate_structural_candidate(rng)
        if not _validate(coeffs):
            continue
        remaining = deadline - time.time() - 5
        if remaining < 1:
            break
        coeffs, sq_count = _local_search(coeffs, rng, time_limit=remaining, max_iter=5000)
        if _validate(coeffs) and sq_count < best_sq_count:
            best_sq_count = sq_count
            best_coeffs = list(coeffs)
    
    if best_coeffs is None:
        # Fallback: simple 12-term polynomial
        best_coeffs = [0] * 12
        for i in range(12):
            best_coeffs[i] = 1
    
    return best_coeffs
# EVOLVE-BLOCK-END
