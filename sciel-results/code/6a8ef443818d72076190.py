"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Return coefficient list in ascending degree, length<=41, nonzero final coefficient,
    entries integers in [-64,64], exactly 12 nonzero terms.
    Minimize nonzero terms in the EXACT integer square.
    """
    # We want a 12-term polynomial P(x) = sum_{i=0}^{11} c_i x^{e_i}
    # such that P(x)^2 has as few nonzero terms as possible.
    
    # Strategy: Use a polynomial where the square causes massive cancellation.
    # The seed program returns [1]*12 which is 1+x+x^2+...+x^11.
    # Its square has 23 nonzero terms (coefficients 1,2,3,...,12,11,...,1).
    
    # Let's try to construct P such that P^2 is very sparse.
    # One approach: P = A + B where A and B are 6-term polynomials with disjoint supports
    # and carefully chosen signs/magnitudes so that cross terms A*B cancel.
    
    # Actually, let's think about this differently. If P = sum c_i x^{e_i}, then
    # P^2 = sum c_i^2 x^{2e_i} + 2 sum_{i<j} c_i c_j x^{e_i + e_j}
    
    # For P^2 to be sparse, we need many of the cross terms to cancel or coincide.
    
    # Let's try a specific construction. Consider P(x) = x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 - x^{-1} ... 
    # No, we need non-negative exponents and length <= 41.
    
    # Let's try: P(x) = (x^6 - 1)(x^2 - 1)(x - 1) / (x-1) ... no.
    
    # Simpler idea: Let P be a product of cyclotomic-like polynomials.
    # Or: P(x) = 1 - x + x^2 - x^3 + x^4 - x^5 + x^6 - x^7 + x^8 - x^9 + x^10 - x^11
    # This is (1 - x^12)/(1 - x) with alternating signs? No, it's (1+x^12)/(1+x) with alternating...
    # Actually 1 - x + x^2 - ... + x^10 - x^11 = (1 - x^12)/(1+x) ... let's check:
    # (1+x)(1 - x + x^2 - ... + x^10 - x^11) = 1 - x^12. Yes!
    # So P = (1 - x^12)/(1+x). Then P^2 = (1-x^12)^2/(1+x)^2.
    # This doesn't immediately give a sparse polynomial.
    
    # Let's try a different approach. What if we use coefficients that are not just +/-1?
    # For example, if P = a*x^0 + b*x^1 + c*x^2 + ... we can tune a,b,c,...
    
    # Actually, let's try the simplest possible improvement: 
    # The seed has 23 nonzero terms in the square. Can we do better?
    
    # Consider P(x) = x^10 + x^9 + x^8 + x^7 + x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 + x^20
    # No, that's not helpful.
    
    # Let's try: P(x) = (x^4 + 1)(x^2 + 1)(x + 1) = x^7 + x^6 + x^5 + x^4 + x^3 + x^2 + x + 1. That's 8 terms.
    
    # We need exactly 12 terms. Let's try P = (x^6 - 1)(x^2 + x + 1) + something?
    # (x^6 - 1)(x^2 + x + 1) = x^8 + x^7 + x^6 - x^2 - x - 1. That's 6 terms.
    
    # Let's try P = (x^4 + x^3 + x^2 + x + 1)(x^2 + 1) = x^6 + x^5 + 2x^4 + 2x^3 + 2x^2 + 2x + 1. 
    # That's 7 terms with some coefficients of 2.
    
    # Hmm, let me just try a few candidates and count nonzero terms in the square.
    
    def count_nonzero_terms(poly_coeffs):
        """Count nonzero terms in a polynomial given as coefficient list."""
        return sum(1 for c in poly_coeffs if c != 0)
    
    def square_coeffs(p):
        """Compute P^2 coefficients from P's coefficient list."""
        n = len(p)
        result = [0] * (2*n - 1)
        for i in range(n):
            for j in range(n):
                result[i+j] += p[i] * p[j]
        return result
    
    # Candidate 1: The seed
    p1 = [1]*12
    sq1 = square_coeffs(p1)
    nz1 = count_nonzero_terms(sq1)
    
    # Candidate 2: Alternating signs
    p2 = [1 if i % 2 == 0 else -1 for i in range(12)]
    sq2 = square_coeffs(p2)
    nz2 = count_nonzero_terms(sq2)
    
    # Candidate 3: Try to make a "difference of squares" structure
    # P = A - B where A and B have overlapping structure
    # Let's try P = x^11 + x^10 + x^9 + x^8 + x^7 + x^6 - x^5 - x^4 - x^3 - x^2 - x - 1
    p3 = [1 if i >= 6 else -1 for i in range(12)]
    sq3 = square_coeffs(p3)
    nz3 = count_nonzero_terms(sq3)
    
    # Candidate 4: P = (x^6+1)(x^5-1) = x^11 + x^6 - x^5 - 1
    # That's only 4 terms. Need 12.
    
    # Let's try P = (x^4 - x^3 + x^2 - x + 1)(x^2 - 1) 
    # = x^6 - x^5 + x^4 - x^3 + x^2 - x^4 + x^3 - x^2 + x - 1
    # = x^6 - x^5 + 0*x^4 + 0*x^3 + 0*x^2 + x - 1
    # That's 4 terms.
    
    # Let me try a more systematic approach. We want 12 terms in P.
    # Let's try P with coefficients that create cancellations in P^2.
    
    # What about P = x^11 - x^10 + x^9 - x^8 + x^7 - x^6 + x^5 - x^4 + x^3 - x^2 + x - 1?
    # That's the same as candidate 2.
    
    # Let's try: P = 1 + 2x + 3x^2 + 4x^3 + 5x^4 + 6x^5 + 7x^6 + 8x^7 + 9x^8 + 10x^9 + 11x^10 + 12x^11
    p4 = [i+1 for i in range(12)]
    sq4 = square_coeffs(p4)
    nz4 = count_nonzero_terms(sq4)
    
    # Let's try: P with a gap. P = x^20 + x^19 + x^18 + x^17 + x^16 + x^15 + x^14 + x^13 + x^12 + x^11 + x^10 + x^9
    # But length would be 21, which is <= 41, so OK.
    p5 = [0]*21
    for i in range(9, 21):
        p5[i] = 1
    sq5 = square_coeffs(p5)
    nz5 = count_nonzero_terms(sq5)
    
    # Let's try: P = x^30 + x^29 + ... + x^19 (12 terms from x^19 to x^30)
    p6 = [0]*31
    for i in range(19, 31):
        p6[i] = 1
    sq6 = square_coeffs(p6)
    nz6 = count_nonzero_terms(sq6)
    
    # The key insight: if all exponents in P are distinct and the sums e_i + e_j are all distinct 
    # for i <= j, then P^2 has exactly C(12+1,2) = 78 terms. But we can have collisions.
    
    # If P has consecutive exponents 0,1,...,11, then e_i + e_j ranges from 0 to 22,
    # and each sum appears multiple times. The number of distinct sums is 23.
    
    # To minimize the number of distinct sums, we want the exponents to be as "clustered" as possible,
    # but we also need cancellations.
    
    # Wait, the number of nonzero terms in P^2 is at most the number of distinct sums e_i+e_j.
    # For consecutive exponents, that's 23. Can we get fewer?
    
    # If we use exponents that are all multiples of 2: 0,2,4,...,22, then sums are 0,2,...,44, 
    # which is 23 distinct sums. Same count.
    
    # If we use a geometric progression: 1,2,4,8,...,1024, then all sums are distinct, giving 78 terms.
    
    # So consecutive exponents give the minimum number of distinct sums (23) for 12 terms.
    # But we can get cancellations! If coefficients are chosen so that some sums cancel.
    
    # For P = sum_{i=0}^{11} c_i x^i, the coefficient of x^k in P^2 is sum_{i+j=k} c_i c_j.
    # For k=0: c_0^2
    # For k=1: 2*c_0*c_1
    # For k=2: c_1^2 + 2*c_0*c_2
    # etc.
    
    # To get cancellation, we need these sums to be zero.
    
    # Let's try to make P^2 = (x^6 + 1)^2 = x^12 + 2x^6 + 1. That's 3 terms!
    # But P = x^6 + 1 has only 2 terms. We need 12 terms.
    
    # What if P = (x^6 + 1) * Q where Q has 6 terms? Then P has up to 12 terms,
    # and P^2 = (x^6+1)^2 * Q^2. If Q^2 is sparse, so is P^2.
    
    # Let Q = x^5 + x^4 + x^3 + x^2 + x + 1. Q^2 has 11 terms.
    # P = (x^6+1)Q = x^11 + x^10 + x^9 + x^8 + x^7 + x^6 + x^5 + x^4 + x^3 + x^2 + x + 1.
    # That's 12 terms! And P^2 = (x^6+1)^2 * Q^2.
    # (x^6+1)^2 = x^12 + 2x^6 + 1.
    # Q^2 = (x^5+...+1)^2.
    # P^2 = (x^12 + 2x^6 + 1) * Q^2 = x^12*Q^2 + 2x^6*Q^2 + Q^2.
    
    # Q = (x^6-1)/(x-1). Q^2 = (x^6-1)^2/(x-1)^2.
    # This is getting complicated. Let me just compute it.
    
    p7 = [1]*12  # This is P = (x^6+1)*Q where Q = 1+x+...+x^5
    # Wait, that's the same as p1! Because (x^6+1)(1+x+...+x^5) = x^11+...+x^6+...+1 = 1+x+...+x^11.
    
    # OK so let me try a different Q. Let Q = x^5 - x^4 + x^3 - x^2 + x - 1.
    # Q has 6 terms. P = (x^6+1)Q = x^11 - x^10 + x^9 - x^8 + x^7 - x^6 + x^5 - x^4 + x^3 - x^2 + x - 1.
    # That's 12 terms with alternating signs!
    
    p8 = [1 if i % 2 == 0 else -1 for i in range(12)]
    sq8 = square_coeffs(p8)
    nz8 = count_nonzero_terms(sq8)
    
    # Let's check what nz8 is.
    # P = (x^6+1)(x^5-x^4+x^3-x^2+x-1)
    # P^2 = (x^6+1)^2 (x^5-x^4+x^3-x^2+x-1)^2
    # (x^5-x^4+x^3-x^2+x-1)^2 ... let's compute.
    
    # Actually, let me just run the computation.
    
    # Let me try yet another approach. What if we use P = x^11 + x^10 + x^9 + x^8 + x^7 + x^6 + x^5 + x^4 + x^3 + x^2 + x + 1
    # but with some coefficients set to make cancellations?
    
    # Let's try P = [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1]
    p9 = [1]*6 + [-1]*6
    sq9 = square_coeffs(p9)
    nz9 = count_nonzero_terms(sq9)
    
    # Let's try P = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]
    p10 = [1 if i % 2 == 0 else -1 for i in range(12)]
    sq10 = square_coeffs(p10)
    nz10 = count_nonzero_terms(sq10)
    
    # Let's try to find a good candidate by trying a few patterns.
    candidates = [
        ("all_ones", [1]*12),
        ("alternating", [1 if i%2==0 else -1 for i in range(12)]),
        ("half_pos_half_neg", [1]*6 + [-1]*6),
        ("linear", [i+1 for i in range(12)]),
        ("quadratic", [(i+1)**2 for i in range(12)]),
        ("cubic", [(i+1)**3 for i in range(12)]),
        ("sin_like", [1 if (i//3)%2==0 else -1 for i in range(12)]),
        ("pattern_111_000", [1,1,1,0,0,0,1,1,1,0,0,0]),  # Only 6 nonzero terms, invalid
    ]
    
    # Filter out invalid candidates (need exactly 12 nonzero terms)
    valid_candidates = []
    for name, p in candidates:
        if sum(1 for c in p if c != 0) == 12:
            sq = square_coeffs(p)
            nz = count_nonzero_terms(sq)
            valid_candidates.append((name, p, nz, sq))
    
    # Sort by number of nonzero terms in square
    valid_candidates.sort(key=lambda x: x[2])
    
    # Pick the best one
    if valid_candidates:
        best_name, best_p, best_nz, best_sq = valid_candidates[0]
        return best_p
    else:
        return [1]*12
# EVOLVE-BLOCK-END
