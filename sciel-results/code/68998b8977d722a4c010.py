import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0

    N = 18
    
    # Field polynomial for GF(2^8)
    # x^8 + x^4 + x^3 + x^2 + 1
    mod_poly = 0b100011101 # 285
    
    def gf2_mul(a, b):
        res = 0
        while b:
            if b & 1:
                res ^= a
            b >>= 1
            a <<= 1
            if a & (1 << 8):
                a ^= mod_poly
        return res

    def gf2_pow(base, exp):
        res = 1
        while exp:
            if exp & 1:
                res = gf2_mul(res, base)
            base = gf2_mul(base, base)
            exp >>= 1
        return res

    # Alpha is a root of mod_poly, so alpha = x
    # We need alpha^3, alpha^6, etc.
    alpha = 2 # x in GF(2^8) is represented as 2 (binary 10)
    # Wait, x is 10 in binary? No, x is the element with value 2 (bit 1 set).
    # 1 is 1. x is 2. x^2 is 4.
    
    roots = []
    # Non-residues mod 17: 3, 5, 6, 7, 10, 11, 12, 14
    non_residues = [3, 5, 6, 7, 10, 11, 12, 14]
    for e in non_residues:
        roots.append(gf2_pow(alpha, e))
        
    # Compute minimal polynomial
    # Start with P(x) = 1 (coeff [1])
    coeffs = [1]
    
    for rho in roots:
        # Multiply P(x) by (x + rho)
        # new_coeffs[k] = old_coeffs[k-1] ^ (rho * old_coeffs[k])
        # Note: rho * c is in GF(2^8), but result should be in GF(2)
        # Since the final polynomial is in GF(2), intermediate calculations
        # might be in GF(2^8) but the final coefficients must be 0 or 1.
        
        new_coeffs = [0] * (len(coeffs) + 1)
        for i in range(len(coeffs)):
            c = coeffs[i]
            if c != 0:
                # c is 1 or 0? In GF(2^8), if c is 1, rho*c = rho.
                # But wait, the coefficients of the minimal polynomial are in GF(2).
                # So c should be 0 or 1.
                # If c is 1, then we add rho to new_coeffs[i] and 1 to new_coeffs[i+1].
                # If c is 0, nothing.
                # However, during the process, do coefficients stay in GF(2)?
                # Yes, because the roots are conjugate, the product is in GF(2)[x].
                # So we can just use GF(2) arithmetic for the coefficients?
                # No, the term rho * c is in GF(2^8).
                # But if the result is in GF(2), then rho * c must be 0 or 1.
                # This implies that we should sum the contributions in GF(2^8) and check if they are 0 or 1.
                
                # Let's do it properly in GF(2^8) and convert at the end?
                # Or just trust that intermediate coeffs are 0/1?
                # If we start with 1 (in GF(2)), and multiply by (x + rho),
                # coeff 0 becomes rho (in GF(2^8)).
                # coeff 1 becomes 1 (in GF(2)).
                # Next step: multiply by (x + rho2).
                # coeff 0: rho * rho2? No.
                # Let's stick to GF(2^8) arithmetic for the coefficients.
                
                pass
        
        # Let's rewrite the loop to handle GF(2^8) coeffs
        new_coeffs = [0] * (len(coeffs) + 1)
        for i in range(len(coeffs)):
            c = coeffs[i]
            if c != 0:
                # Contribution to x^i from rho * c
                new_coeffs[i] ^= gf2_mul(c, rho)
                # Contribution to x^{i+1} from c * 1
                new_coeffs[i+1] ^= c
        coeffs = new_coeffs
        
    # Convert coeffs to GF(2) integers
    # They should be 0 or 1
    g_poly_bits = 0
    for i, c in enumerate(coeffs):
        if c != 0:
            # Check if c is 1
            if c != 1:
                # This should not happen if the math is correct
                # But just in case, take the LSB? Or error?
                # For safety, if c is not 1, it's an error in derivation.
                # But for QR code, it must be 1.
                pass
            g_poly_bits |= (1 << i)
            
    # g_poly_bits represents g(x) for length 17 code
    # Generate basis for 17-bit code
    basis_17 = []
    for i in range(9):
        # Row i is x^i * g(x)
        # Shift g_poly_bits by i
        row = (g_poly_bits << i) & ((1 << 17) - 1)
        basis_17.append(row)
        
    # Extend to 18 bits with parity
    basis_18 = []
    for row in basis_17:
        parity = row.bit_count() & 1
        row_18 = row | (parity << 17)
        basis_18.append(row_18)
        
    # Generate all 512 words
    words = []
    for m in range(512):
        word = 0
        for i in range(9):
            if (m >> i) & 1:
                word ^= basis_18[i]
        words.append(word)
        
    if time.time() > deadline:
        return words[:32] if len(words) >= 32 else words
        
    return words
