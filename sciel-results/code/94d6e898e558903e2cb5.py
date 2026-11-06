# EVOLVE-BLOCK-START
def construct():
    """
    Return coefficient list in ascending degree, length<=41, nonzero final coefficient,
    entries integers in [-64,64], exactly 12 nonzero terms.
    Minimize nonzero terms in the EXACT integer square.
    """
    
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

    def normalize_poly(p):
        """Remove trailing zeros to ensure nonzero final coefficient and minimal length."""
        while len(p) > 0 and p[-1] == 0:
            p.pop()
        if not p:
            return [0]
        return p

    candidates = []

    # Candidate 1: All ones
    candidates.append([1]*12)
    
    # Candidate 2: Alternating signs
    candidates.append([1 if i % 2 == 0 else -1 for i in range(12)])
    
    # Candidate 3: Half positive, half negative
    candidates.append([1]*6 + [-1]*6)
    
    # Candidate 4: P = (x^6 - 1)(x^5 + x^4 + x^3 + x^2 + x + 1)
    # = x^11 + x^10 + x^9 + x^8 + x^7 + x^6 - x^5 - x^4 - x^3 - x^2 - x - 1
    candidates.append([1]*6 + [-1]*6) # Same as 3
    
    # Candidate 5: P = (x^6 + 1)(x^5 - x^4 + x^3 - x^2 + x - 1)
    # = x^11 - x^10 + x^9 - x^8 + x^7 - x^6 + x^5 - x^4 + x^3 - x^2 + x - 1
    candidates.append([1 if i % 2 == 0 else -1 for i in range(12)]) # Same as 2

    # Candidate 6: P = (x^2 + x + 1)(x^9 + x^8 - x^7 - x^6 + x^5 + x^4 - x^3 - x^2 + x + 1)
    # Let's construct this explicitly
    A = [1, 1, 1]
    B = [1, 1, -1, -1, 1, 1, -1, -1, 1, 1] # 10 terms? No, let's make B 6 terms to get 12?
    # If A has 3 terms and B has 4 terms, product has up to 12 terms.
    # Let A = 1 + x + x^2
    # Let B = 1 - x + x^4 - x^5
    # P = (1+x+x^2)(1-x+x^4-x^5) = 1 - x + x^4 - x^5 + x - x^2 + x^5 - x^6 + x^2 - x^3 + x^6 - x^7
    # = 1 + x^4 - x^3 - x^7
    # This has 4 terms. Not 12.
    
    # Let's try A = 1 + x + x^2 + x^3 (4 terms)
    # B = 1 - x + x^4 - x^5 (4 terms)
    # P = (1+x+x^2+x^3)(1-x+x^4-x^5)
    # = 1 - x + x^4 - x^5 + x - x^2 + x^5 - x^6 + x^2 - x^3 + x^6 - x^7 + x^3 - x^4 + x^7 - x^8
    # = 1 - x^8
    # 2 terms.
    
    # We need exactly 12 terms.
    # Let's try A = 1 + x + x^2 + x^3 + x^4 + x^5 (6 terms)
    # B = 1 - x + x^2 - x^3 + x^4 - x^5 (6 terms)
    # P = A B = (1+x^6)(1-x^2+x^4) = (1+x^6)(1+x^4)(1-x^2)
    # = (1 + x^4 + x^6 + x^{10})(1-x^2)
    # = 1 - x^2 + x^4 - x^6 + x^6 - x^8 + x^{10} - x^{12}
    # = 1 - x^2 + x^4 - x^8 + x^{10} - x^{12}
    # 6 terms.
    
    # To get 12 terms, we can shift B.
    # A = 1 + x + x^2 + x^3 + x^4 + x^5
    # B = x^6 - x^7 + x^8 - x^9 + x^{10} - x^{11}
    # P = A + B (Sum, not product)
    # P = 1 + x + x^2 + x^3 + x^4 + x^5 + x^6 - x^7 + x^8 - x^9 + x^{10} - x^{11}
    # 12 terms.
    candidates.append([1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1])
    
    # Candidate 7: P = 1 - x + x^2 - x^3 + x^4 - x^5 + x^6 - x^7 + x^8 - x^9 + x^{10} - x^{11}
    # Already in list.
    
    # Candidate 8: P = 1 + x - x^2 - x^3 + x^4 + x^5 - x^6 - x^7 + x^8 + x^9 - x^{10} - x^{11}
    # Pattern: 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1
    candidates.append([1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1])
    
    # Candidate 9: P = 1 - x - x^2 + x^3 + x^4 - x^5 - x^6 + x^7 + x^8 - x^9 - x^{10} + x^{11}
    # Pattern: 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1
    candidates.append([1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1])

    best_p = None
    best_nz = float('inf')
    
    for p in candidates:
        # Check validity: exactly 12 nonzero terms
        nz_p = count_nonzero_terms(p)
        if nz_p != 12:
            continue
        
        # Check coefficients in [-64, 64]
        if any(c < -64 or c > 64 for c in p):
            continue
            
        sq = square_coeffs(p)
        nz_sq = count_nonzero_terms(sq)
        
        if nz_sq < best_nz:
            best_nz = nz_sq
            best_p = p[:]
            
    if best_p is not None:
        return best_p
    else:
        return [1]*12
# EVOLVE-BLOCK-END
