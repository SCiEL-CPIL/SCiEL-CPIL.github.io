"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Construct the extended quadratic residue code of length 18.
    # This is a [18, 9, 6] linear code with 512 codewords.
    
    # GF(16) arithmetic with primitive polynomial x^4 + x + 1
    # alpha is a root of x^4 + x + 1
    # We represent elements as integers 0-15 representing coefficients of 1, x, x^2, x^3
    
    def gf_mult(a, b):
        res = 0
        for i in range(4):
            if (b >> i) & 1:
                res ^= a
            a = (a << 1) ^ (0x3 if (a >> 3) & 1 else 0) # x^4 = x + 1 -> 0b0011
            # Wait, x^4 + x + 1 = 0 => x^4 = x + 1.
            # If bit 3 is set, x^4 is generated.
            # (a << 1) shifts left. If a had bit 3, it goes to bit 4.
            # We need to reduce mod x^4+x+1.
            # x^4 = x + 1.
            # So if a & 8: res ^= (a << 1) ^ 0b0011? No.
            # Standard reduction:
            # a = a0 + a1 x + a2 x^2 + a3 x^3
            # a x = a0 x + a1 x^2 + a2 x^3 + a3 x^4
            # a3 x^4 = a3 (x + 1)
            # So a x = a0 x + a1 x^2 + a2 x^3 + a3 x + a3
            # = (a0+a3) x + a1 x^2 + a2 x^3 + a3
            # Bits:
            # bit 0: a3
            # bit 1: a0 ^ a3
            # bit 2: a1
            # bit 3: a2
            # This is (a << 1) ^ (0b0011 if a & 8 else 0) ?
            # Let's trace: a=8 (1000). a<<1 = 10000.
            # a & 8 is true.
            # 10000 ^ 00011 = 10011.
            # 10011 is 19. In GF(16) we only care about low 4 bits?
            # The result of multiplication should be in 0-15.
            # My loop logic above was slightly off.
            
    # Let's use a table for multiplication or a better function.
    # Actually, we just need to find minimal polynomials.
    # We can compute them by finding the characteristic polynomial of the linear map x -> alpha*x.
    # Or simpler: iterate polynomials.
    
    # Let's hardcode the generator polynomial if we can derive it reliably.
    # Or just compute it.
    
    # Primitive poly: x^4 + x + 1 (0b01011)
    # alpha = 0b0010 (x)
    
    def gf_mul(a, b):
        p = 0
        while b:
            if b & 1:
                p ^= a
            b >>= 1
            a <<= 1
            if a & 0x10: # x^4
                a ^= 0x13 # x^4 + x + 1
        return p & 0xF

    # Find minimal polynomial of alpha^i
    def min_poly(i):
        # We want poly p(x) = x^k + ... + 1 such that p(alpha^i) = 0
        # Try degrees 1, 2, 4...
        # For GF(16), degrees are 1, 2, or 4.
        # alpha^1 is degree 4.
        # alpha^3 is degree 4.
        
        # Construct polynomial with roots alpha^(i * 2^j)
        # Start with (x - alpha^i)
        # poly = [1, alpha^i] (coeffs of x^1, x^0)
        # Multiply by (x - alpha^(i*2))
        
        roots = []
        r = pow(alpha, i, 16) # This is not standard pow for GF
        # Let's just compute powers
        r = 1
        for _ in range(i):
            r = gf_mul(r, alpha)
        
        # Conjugates: r, r^2, r^4, r^8
        c1 = r
        c2 = gf_mul(c1, c1)
        c3 = gf_mul(c2, c2)
        c4 = gf_mul(c3, c3)
        
        # Poly = (x - c1)(x - c2)(x - c3)(x - c4)
        # In GF(2), x - c = x + c
        # Start with p = [1, c1] (x + c1)
        # p = [1, c1]
        # Multiply by (x + c2):
        # (x + c1)(x + c2) = x^2 + (c1+c2)x + c1c2
        # p = [1, c1+c2, gf_mul(c1, c2)]
        
        p = [1, c1]
        # Multiply by (x + c2)
        new_p = [0] * (len(p) + 1)
        for k in range(len(p)):
            new_p[k] ^= p[k]
            new_p[k+1] ^= gf_mul(p[k], c2)
        p = new_p
        
        # Multiply by (x + c3)
        new_p = [0] * (len(p) + 1)
        for k in range(len(p)):
            new_p[k] ^= p[k]
            new_p[k+1] ^= gf_mul(p[k], c3)
        p = new_p
        
        # Multiply by (x + c4)
        new_p = [0] * (len(p) + 1)
        for k in range(len(p)):
            new_p[k] ^= p[k]
            new_p[k+1] ^= gf_mul(p[k], c4)
        p = new_p
        
        # p is in GF(16). We need to verify it's in GF(2).
        # It should be.
        # Convert to integer representation of polynomial over GF(2)
        # p[0] is coeff of x^4, p[3] is coeff of x^0.
        # Check if all coeffs are in GF(2) (0 or 1)
        # If not, something is wrong.
        
        res = 0
        for k, coeff in enumerate(reversed(p)): # p is [x^4, x^3, x^2, x, 1]
            # coeff should be 0 or 1
            if coeff == 1:
                res |= (1 << k)
            elif coeff != 0:
                # Error
                pass
        return res

    alpha = 0b0010 # x
    
    m1 = min_poly(1)
    m3 = min_poly(3)
    
    # g(x) = m1(x) * m3(x)
    # Polynomial multiplication in GF(2)
    def poly_mult(a, b):
        res = 0
        while b:
            if b & 1:
                res ^= a
            a <<= 1
            b >>= 1
        return res

    g = poly_mult(m1, m3)
    
    # g is an integer representing the polynomial.
    # e.g. if g = 0b10011, it's x^4 + x + 1.
    # The code is [17, 9, 5].
    # Generator matrix G is formed by rows of g(x), x*g(x), ..., x^8*g(x).
    # We can just generate codewords by multiplying message polynomial by g(x).
    
    words = []
    for msg in range(512):
        # msg is 9 bits.
        # c(x) = msg(x) * g(x)
        # msg(x) is degree < 9.
        # g(x) is degree 8.
        # c(x) is degree < 17.
        
        c = 0
        m = msg
        # Multiply m(x) by g(x)
        # m is an integer.
        # We can do bitwise multiplication.
        # c = 0
        # while m:
        #    if m & 1: c ^= g
        #    g <<= 1
        #    m >>= 1
        
        # Wait, this is standard integer multiplication, which is correct for GF(2) polynomials.
        c = 0
        temp_g = g
        temp_m = msg
        while temp_m:
            if temp_m & 1:
                c ^= temp_g
            temp_g <<= 1
            temp_m >>= 1
            
        # c is an integer < 2^17.
        # This is the 17-bit codeword.
        
        # Extend with parity bit.
        # Parity bit is the parity of the weight of c.
        parity = c.bit_count() % 2
        # We want total weight even.
        # So if c has odd weight, add 1. If even, add 0.
        # Wait, standard extension: c_ext = c || p, where p = sum(c_i) mod 2.
        # Then weight(c_ext) = weight(c) + p.
        # If weight(c) is odd, p=1, weight is even.
        # If weight(c) is even, p=0, weight is even.
        # So p = weight(c) % 2.
        
        word = (c << 1) | parity
        
        words.append(word)
        
    return words
# EVOLVE-BLOCK-END
