import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import time
import sys

def construct():
    N = 17
    MIN_DIST = 6
    deadline = time.time() + 55.0  # 55s safety margin within 60s limit
    
    # Start with the seed: 32 words from first-order Reed-Muller on 16 coords + 1 zero bit
    seed_words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            # bit 16 is 0 (the extra coordinate)
            seed_words.append(word)
    
    seed_set = set(seed_words)
    
    # Generate candidate words: all vectors at distance >= MIN_DIST from all seed words
    # But that's 2^17 = 131072 candidates. Instead, let's be smarter.
    
    # Strategy: 
    # 1. Build the seed code
    # 2. Try to add more words via greedy/local search
    # 3. Use SDP relaxation for upper bound (but we need to maximize, not just bound)
    
    # Actually, let's try a different approach:
    # The [18,9,6] extended quadratic residue code shortened by 1 coordinate gives 256 words.
    # Let's try to construct a BCH code of length 17 with designed distance 6.
    
    # BCH code construction over GF(2^8):
    # Length n=17, primitive 17th root of unity in GF(256)
    # 2^8 = 256 = 15*17 + 1, so alpha^15 is a primitive 17th root of unity
    
    # Find primitive polynomial of degree 8 over GF(2)
    # x^8 + x^4 + x^3 + x^2 + 1 is primitive
    def gf2_mult(a, b, poly=0b100111001):  # 1 + x^2 + x^3 + x^4 + x^8
        """Multiply two elements in GF(2^8) using primitive polynomial x^8+x^4+x^3+x^2+1"""
        result = 0
        while b:
            if b & 1:
                result ^= a
            b >>= 1
            a <<= 1
            if a & (1 << 8):
                a ^= poly & 0xFF  # reduce mod primitive polynomial (low 8 bits)
        return result & 0xFF
    
    def gf2_pow(a, e, poly=0b100111001):
        """Compute a^e in GF(2^8)"""
        result = 1
        while e:
            if e & 1:
                result = gf2_mult(result, a, poly)
            a = gf2_mult(a, a, poly)
            e >>= 1
        return result
    
    # alpha is a primitive element of GF(256) (root of primitive poly)
    alpha = 2  # x in GF(2^8) representation (binary 10)
    # beta = alpha^15 is primitive 17th root of unity
    beta = gf2_pow(alpha, 15)
    
    # Verify: beta^17 should be 1
    assert gf2_pow(beta, 17) == 1, "beta is not a 17th root of unity"
    
    # Compute minimal polynomials of beta^i for i=1,...,5
    # Minimal polynomial of beta^i: find smallest degree d such that beta^i satisfies a degree-d poly over GF(2)
    # Use the fact that the minimal polynomial has degree equal to the order of 2 mod 17/gcd(i,17)
    
    # Order of 2 mod 17: 2^8=256=256 mod 17 = 256-15*17=1, so order is 8
    # For i coprime to 17, minimal polynomial has degree 8
    # All i=1..16 are coprime to 17, so each beta^i has minimal polynomial of degree 8
    
    # But we need to compute the actual minimal polynomials
    # Minimal poly of beta^i: m_i(x) = prod_{j=0}^{7} (x - beta^{i*2^j})
    
    def minimal_poly(b, n=17):
        """Compute minimal polynomial of b (a primitive n-th root) over GF(2)
        Returns polynomial as list of coefficients [c0, c1, ..., cd] where c0 is constant term
        """
        # The conjugates are b, b^2, b^4, ..., b^(2^7) mod n
        conjugates = []
        val = b
        for _ in range(8):
            conjugates.append(val)
            val = gf2_mult(val, val)
        
        # m(x) = prod (x - c_j) over GF(2)
        # Start with m(x) = x, then multiply by (x - c_j)
        # In GF(2), x - c = x + c
        poly_coeffs = [1]  # start with 1 (constant)
        for c in conjugates:
            # multiply by (x + c)
            new_len = len(poly_coeffs) + 1
            new_poly = [0] * new_len
            for j in range(len(poly_coeffs)):
                new_poly[j] ^= poly_coeffs[j]      # x * poly_coeffs
                new_poly[j+1] ^= gf2_mult(poly_coeffs[j], c)  # c * poly_coeffs
            poly_coeffs = new_poly
        return poly_coeffs
    
    # Compute minimal polynomials for beta^1, beta^2, beta^3, beta^4, beta^5
    min_polys = {}
    for i in range(1, 6):
        beta_i = gf2_pow(beta, i)
        min_polys[i] = minimal_poly(beta_i)
    
    # The generator polynomial g(x) = lcm of minimal polynomials of beta^1,...,beta^5
    # Since they may share factors, we need to compute lcm
    # In GF(2)[x], lcm(a,b) = a*b/gcd(a,b)
    
    def poly_gcd(p, q):
        """Compute gcd of two polynomials over GF(2)
        p, q are lists of coefficients [c0, c1, ...] with c0 constant term
        """
        def poly_divmod(a, b):
            """Divide polynomial a by b, return (quotient, remainder)"""
            if len(a) < len(b):
                return [0], a
            a = a[:]  # copy
            b = b[:]
            deg_a = len(a) - 1
            deg_b = len(b) - 1
            if deg_b == 0:
                # b is a constant
                c = b[0]
                if c == 0:
                    raise ValueError("Division by zero polynomial")
                # In GF(2), dividing by 1 is just the polynomial itself
                # If c != 1, this shouldn't happen for monic polynomials
                return a[:], [0]
            q = [0] * len(a)
            while len(a) >= len(b) and any(a):
                deg_diff = len(a) - 1 - (len(b) - 1)
                # a's leading term is 1 (assumed monic)
                # Subtract x^deg_diff * b from a
                for j in range(len(b)):
                    a[j + deg_diff] ^= b[j]
                q[deg_diff] = 1
                # Remove leading zeros
                while len(a) > 0 and a[-1] == 0:
                    a.pop()
                if not a:
                    a = [0]
            # Trim q
            while len(q) > 0 and q[-1] == 0:
                q.pop()
            if not q:
                q = [0]
            return q, a
        
        def poly_lcm(p, q):
            """Compute lcm of two polynomials over GF(2)"""
            g = poly_gcd(p, q)
            if not g or g[0] == 0 and len(g) > 1:
                return p
            # lcm = p * q / gcd
            # First divide p by g
            q1, r1 = poly_divmod(p, g)
            # Then multiply by q
            # Polynomial multiplication
            result = [0] * (len(q1) + len(q) - 1)
            for i, ci in enumerate(q1):
                if ci:
                    for j, cj in enumerate(q):
                        result[i+j] ^= ci * cj
            while len(result) > 0 and result[-1] == 0:
                result.pop()
            if not result:
                result = [0]
            return result
        
        a, b = p[:], q[:]
        # Ensure a has higher or equal degree
        if len(a) < len(b):
            a, b = b, a
        while any(b):
            deg_diff = len(a) - len(b)
            if deg_diff < 0:
                break
            for j in range(len(b)):
                a[j + deg_diff] ^= b[j]
            while len(a) > 0 and a[-1] == 0:
                a.pop()
            if not a:
                a = [0]
            a, b = b, a
            if len(a) < len(b):
                a, b = b, a
        while len(a) > 0 and a[-1] == 0:
            a.pop()
        if not a:
            a = [0]
        return a
    
    def poly_lcm(p, q):
        """Compute lcm of two polynomials over GF(2)"""
        g = poly_gcd(p, q)
        if g == [0] or (len(g) > 1 and g[-1] == 0):
            # One polynomial is zero
            return p if any(p) else q
        # Divide p by g
        # We need polynomial division
        def poly_div(a, b):
            """Divide polynomial a by monic polynomial b, return quotient"""
            a = a[:]
            b = b[:]
            while len(a) >= len(b) and any(a):
                deg_diff = len(a) - 1 - (len(b) - 1)
                if deg_diff < 0:
                    break
                for j in range(len(b)):
                    a[j + deg_diff] ^= b[j]
                while len(a) > 0 and a[-1] == 0:
                    a.pop()
                if not a:
                    a = [0]
            return a
        
        # This is getting complex. Let me use a simpler approach.
        pass
    
    # This is getting too complex. Let me try a different approach.
    # Let's just use the seed and try to improve it with local search.
    
    best_code = list(seed_set)
    best_size = len(best_code)
    
    # Try to add words one by one using greedy approach
    # Generate all 2^17 = 131072 candidate words
    # For each candidate, check if it's at distance >= 6 from all current codewords
    
    all_candidates = list(range(1 << N))
    np.random.shuffle(all_candidates)
    
    current_code = list(best_code)
    current_set = set(current_code)
    
    def hamming_dist(a, b):
        return (a ^ b).bit_count()
    
    def can_add(word, code_set):
        for w in code_set:
            if hamming_dist(word, w) < MIN_DIST:
                return False
        return True
    
    # Greedy: try to add each candidate in random order
    for word in all_candidates:
        if time.time() > deadline - 10:  # Leave 10s for cleanup
            break
        if word not in current_set and can_add(word, current_set):
            current_set.add(word)
            current_code.append(word)
    
    best_code = current_code
    best_size = len(best_code)
    
    # Local search: try to swap words to improve
    # This is a simplified version; a full local search would be more complex
    
    return best_code
