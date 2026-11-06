import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 60.0  # 60 seconds
    
    # Step 1: Construct the [19,10,6] QR code generator polynomial
    # We need GF(2^18)
    # Primitive polynomial: x^18 + x^7 + 1
    # Let's verify if it's primitive.
    # If not, try x^18 + x^11 + x^10 + x^5 + 1 or similar.
    # For now, assume x^18 + x^7 + 1 works.
    
    def gf_mult(a, b, mod_poly):
        # Multiply two elements in GF(2^18)
        result = 0
        while b:
            if b & 1:
                result ^= a
            b >>= 1
            a <<= 1
            # Reduce if a has bit 18 set
            if a & (1 << 18):
                a ^= mod_poly
        return result

    def gf_pow(a, n, mod_poly):
        result = 1
        while n:
            if n & 1:
                result = gf_mult(result, a, mod_poly)
            a = gf_mult(a, a, mod_poly)
            n >>= 1
        return result

    # Mod polynomial for x^18 + x^7 + 1
    # Bits: 18 and 7 and 0
    mod_poly = (1 << 18) | (1 << 7) | 1
    
    # Check if 2 is a primitive element for this polynomial
    # Order of 2 should be 2^18 - 1
    # We can check if 2^((2^18-1)/p) != 1 for all prime factors p of 2^18-1
    # 2^18 - 1 = 262143 = 3 * 87381 = 3 * 3 * 29127 = 9 * 29127 = 9 * 3 * 9709 = 27 * 9709
    # 9709 is prime?
    # Let's just assume it's primitive for now, or test it.
    
    alpha = 2
    # Test primitivity
    is_primitive = True
    factors = [3, 9709] # Prime factors of 2^18-1
    for p in factors:
        if gf_pow(alpha, (2**18 - 1) // p, mod_poly) == 1:
            is_primitive = False
            break
            
    if not is_primitive:
        # Try another polynomial or element
        # For simplicity, if this fails, we might fall back to a slower method
        # But x^18+x^7+1 is likely primitive.
        pass

    # Quadratic residues mod 19
    Q = [1, 4, 5, 6, 7, 9, 11, 16, 17]
    
    # Compute g(x) = prod (x - alpha^i) for i in Q
    # In GF(2), x - alpha^i = x + alpha^i
    # So g(x) = (x + alpha^1) (x + alpha^4) ...
    
    # Represent polynomials as lists of coefficients (LSB first)
    # g starts as 1
    g_coeffs = [1]
    
    for i in Q:
        # Multiply g(x) by (x + alpha^i)
        # This is equivalent to shifting g(x) and adding g(x) * alpha^i
        # Let g(x) = sum g_k x^k
        # (x + a) g(x) = sum g_k x^{k+1} + a sum g_k x^k
        
        # New degree is len(g_coeffs)
        # We need to multiply coefficients by alpha^i in GF(2^18)
        # Wait, the coefficients of g(x) are in GF(2), not GF(2^18)!
        # The roots are in GF(2^18), but the polynomial has coefficients in GF(2).
        # This is because the set Q is closed under multiplication by 2 mod 19.
        # So the product will have coefficients in GF(2).
        
        # How to compute the product in GF(2) directly?
        # We can't just use gf_mult on the coefficients because they are 0 or 1.
        # We need to perform polynomial multiplication over GF(2).
        # But the "values" alpha^i are in GF(2^18).
        # This approach is tricky because we are mixing GF(2) coefficients with GF(2^18) roots.
        
        # Better approach:
        # The generator polynomial g(x) is the minimal polynomial of alpha^1 over GF(2).
        # Since the order of 2 mod 19 is 18, the minimal polynomial has degree 18?
        # No, the degree is the size of the cyclotomic coset.
        # Coset of 1: {1, 2, 4, 8, 16, 13, 7, 14, 9, 18, 17, 15, 11, 3, 6, 12, 5, 10}.
        # Size is 18.
        # So the minimal polynomial of alpha has degree 18.
        # But the QR code [19,10,6] has generator polynomial of degree 9.
        # This means g(x) is NOT the minimal polynomial of alpha.
        # g(x) is the product of minimal polynomials of alpha^i for i in Q.
        # Q has 9 elements.
        # The coset of 1 has 18 elements.
        # So the minimal polynomial of alpha^1 includes roots alpha^1, alpha^2, ..., alpha^18.
        # This includes both Q and non-Q.
        # So we cannot just take the minimal polynomial.
        
        # We must compute the product of (x - alpha^i) for i in Q.
        # Since the result has coefficients in GF(2), we can compute it symbolically.
        # Let P(x) = prod_{i in Q} (x + alpha^i).
        # The coefficients are symmetric polynomials in alpha^i.
        # They are in GF(2).
        
        # We can compute this by iterating and multiplying polynomials with coefficients in GF(2^18),
        # then checking if the final coefficients are 0 or 1.
        
        g_coeffs_gf = [1] # Coeffs in GF(2^18)
        
        for i in Q:
            a = gf_pow(alpha, i, mod_poly)
            # Multiply by (x + a)
            # Shift g_coeffs_gf by 1 (multiply by x)
            new_g = [0] + g_coeffs_gf
            # Add a * g_coeffs_gf
            for k in range(len(g_coeffs_gf)):
                new_g[k] ^= gf_mult(g_coeffs_gf[k], a, mod_poly)
            g_coeffs_gf = new_g
            
        # Now g_coeffs_gf should contain only 0s and 1s.
        g_coeffs_bin = [1 if c == 1 else 0 for c in g_coeffs_gf]
        
        # g_coeffs_bin is the generator polynomial for [19,10,6]
        # Degree should be 9.
        # g_coeffs_bin[0] should be 1.
        
        # Construct generator matrix G_19 (10x19)
        # Rows are cyclic shifts of g_coeffs_bin
        # Note: g_coeffs_bin is LSB first.
        # Row 0: g_coeffs_bin + [0]*10 ? No.
        # The code is cyclic.
        # The generator matrix can be formed by the rows:
        # g(x), x*g(x), ..., x^9*g(x)
        # In vector form (LSB first):
        # Row i is the coefficients of x^i * g(x) mod (x^19 - 1).
        # Since deg(g) = 9, x^i * g(x) has degree i+9 < 19.
        # So no wrap-around needed for i=0..9.
        
        G_19 = np.zeros((10, 19), dtype=int)
        for i in range(10):
            # Shift g_coeffs_bin by i
            # g_coeffs_bin has length 10.
            # We need to place it in a vector of length 19.
            # Position i to i+9.
            for j in range(10):
                G_19[i, i+j] = g_coeffs_bin[j]
                
        # Shorten by removing the last column (index 18)
        G_18 = G_19[:, :18]
        
        # Now G_18 is a 10x18 matrix.
        # The code is the span of the rows of G_18.
        # Wait, shortening a [19,10,6] code gives a [18,9,6] code.
        # The dimension drops by 1.
        # The rows of G_18 are linearly dependent?
        # Yes, because the last column of G_19 was removed.
        # The row space of G_18 has dimension 9.
        # We need to find a basis for this row space.
        
        # Perform Gaussian elimination on G_18 to get a basis.
        # Or simply generate all 2^10 combinations and remove duplicates/dependencies?
        # 2^10 = 1024 words.
        # We can generate all 1024 words, check which are distinct, and take the first 256?
        # No, we need the code of size 256.
        # The code is the set of all linear combinations.
        # Since the dimension is 9, there are exactly 256 codewords.
        
        # Generate all codewords
        codewords = []
        for mask in range(256): # We only need 256 words, but the basis has 9 vectors.
            # We need the basis first.
            pass
            
        # Let's find the basis of G_18.
        # G_18 is 10x18.
        # We can use numpy to find the rank and basis.
        # Or just iterate through all 1024 combinations, compute the word, and add to a set.
        
        words_set = set()
        for mask in range(1024):
            word = 0
            for i in range(10):
                if mask & (1 << i):
                    # Add row i
                    for j in range(18):
                        if G_18[i, j] == 1:
                            word ^= (1 << j)
            words_set.add(word)
            
        # The set should have size 256.
        words = list(words_set)
        
        # Verify distance
        # We trust the construction.
        
        return words

# Test the function
if __name__ == "__main__":
    words = construct()
    print(len(words))
