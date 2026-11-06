import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 60.0
    
    # 1. Construct GF(2^9)
    # Primitive polynomial: x^9 + x^4 + 1
    # Coefficients from high to low: 1, 0, 0, 0, 1, 0, 0, 0, 0, 1
    # In bit representation (LSB first): 1 (x^0), 0, 0, 0, 1 (x^4), 0, 0, 0, 0, 1 (x^9)
    # poly = 1 + 16 + 512 = 529
    
    m = 9
    poly = 1 + (1 << 4) + (1 << 9) # 1 + x^4 + x^9
    
    # We need a primitive element alpha. Let alpha = 2 (x).
    # We need to verify alpha is primitive.
    # Order of alpha should be 2^9 - 1 = 511.
    # Since 511 is prime, any non-zero element other than 1 is primitive.
    # So alpha = 2 is primitive.
    
    # Function to multiply in GF(2^9)
    def gf_mult(a, b):
        res = 0
        while b:
            if b & 1:
                res ^= a
            b >>= 1
            a <<= 1
            if a & (1 << m):
                a ^= poly
        return res

    # Function to compute power alpha^k
    # alpha = 2
    # alpha^k can be computed by repeated squaring or just a loop since k < 511
    
    # We need the generator polynomial g(x) of the QR code.
    # Roots are alpha^(2i) for i in Q (quadratic residues mod 19).
    # Q = {1, 4, 5, 6, 7, 9, 11, 16, 17}
    
    # We need the minimal polynomial of each root.
    # Since 2 is a primitive root mod 19?
    # 2^1=2, 2^2=4, 2^3=8, 2^4=16=-3, 2^5=32=13, 2^6=26=7, 2^7=14, 2^8=28=9, 2^9=18=-1.
    # So 2 has order 18 mod 19.
    # This means the cyclotomic cosets mod 19 under multiplication by 2 are of size 18?
    # No, the order of 2 mod 19 is 18.
    # The cosets are orbits of x -> 2x mod 19.
    # Since gcd(2, 19)=1, the orbits partition {1, ..., 18}.
    # Size of orbit of 1 is 18.
    # This implies there is only ONE cyclotomic coset of size 18?
    # Wait, 2 is a primitive root mod 19.
    # So the coset of 1 is {1, 2, 4, 8, 16, 13, 7, 14, 9, 18, 17, 15, 11, 3, 6, 12, 5, 10}.
    # This covers all non-zero residues.
    # This means the minimal polynomial of alpha has degree 18?
    # But we are in GF(2^9). The degree of the minimal polynomial of alpha^k must divide 9?
    # No, alpha is in GF(2^9). The minimal polynomial of alpha over GF(2) has degree 9.
    # The roots of the minimal polynomial of alpha are alpha, alpha^2, alpha^4, alpha^8, alpha^16, alpha^32, ...
    # The exponents are $1 \cdot 2^j \pmod{511}$.
    # This is different from the QR code definition which uses roots $\alpha^{2i}$ where $\alpha$ is a 19th root of unity.
    
    # Ah, the QR code is a cyclic code of length $n=19$.
    # The field containing the 19th roots of unity is $GF(2^9)$ because $2^9 \equiv 1 \pmod{19}$?
    # $2^9 = 512 = 19 \times 26 + 18 \equiv -1 \pmod{19}$.
    # So $2^{18} \equiv 1 \pmod{19}$.
    # The order of 2 mod 19 is 18.
    # So the 19th roots of unity are in $GF(2^{18})$?
    # No, if $2^9 \equiv -1$, then $2^{18} \equiv 1$.
    # The smallest $k$ such that $2^k \equiv 1 \pmod{19}$ is 18.
    # So the 19th roots of unity lie in $GF(2^{18})$.
    # But we can represent them in $GF(2^9)$?
    # If $\beta$ is a 19th root of unity, $\beta^{19}=1$.
    # The minimal polynomial of $\beta$ has degree 18?
    # If so, we need $GF(2^{18})$.
    
    # However, QR codes are often defined using the field where the roots exist.
    # For $p=19$, the degree of the minimal polynomial of a primitive 19th root is 18.
    # So we need $GF(2^{18})$.
    
    # Let's switch to $GF(2^{18})$.
    # Find a primitive polynomial of degree 18.
    # This is getting complicated to implement from scratch reliably in 60s.
    
    # Alternative:
    # Use a known generator polynomial for the QR(19) code.
    # Or, use a simpler construction.
    
    # What about the **Extended Golay Code** [24, 12, 8]?
    # We can shorten it by 6 positions to get [18, 6, 8] -> Size 64. Score 2.0.
    # This is safer and easier to implement correctly.
    
    # Can we do better than 64?
    # Shortening by 4 positions: [20, 8, 8]? Size 256.
    # If we shorten [24, 12, 8] by 4 positions, we get [20, 8, 8].
    # Then shorten by 2 more to get [18, 6, 8]? No, shortening preserves distance.
    # [24, 12, 8] -> Shorten 6 -> [18, 6, 8] (Size 64).
    # [24, 12, 8] -> Shorten 4 -> [20, 8, 8] (Size 256).
    # [20, 8, 8] -> Shorten 2 -> [18, 6, 8] (Size 64).
    # Wait, shortening reduces dimension.
    # If we shorten [24, 12, 8] by 6, dim drops by 6 (if independent) to 6. Size 64.
    # If we shorten by 4, dim drops by 4 to 8. Size 256. Length 20.
    # We need length 18.
    # We can take the [20, 8, 8] code and PUNCTURE it by 2?
    # Puncturing reduces distance. $d_{punctured} \ge d - 1 = 7$.
    # So [18, 8, 7] code.
    # Distance 7 is $> 6$.
    # So a [18, 8, 7] code is a valid code for $D=6$.
    # Size 256.
    
    # So:
    # 1. Construct Extended Binary Golay Code [24, 12, 8].
    # 2. Shorten by 4 positions (e.g., 0, 1, 2, 3). Get [20, 8, 8].
    # 3. Puncture by 2 positions (e.g., 0, 1 of the shortened code). Get [18, 8, 7].
    # 4. Return the 256 words.
    
    # This is a very strong candidate.
    
    # Implementation of Golay Code:
    # Generator polynomial for [23, 12, 7] Golay code:
    # g(x) = x^11 + x^10 + x^6 + x^5 + x^4 + x^2 + 1
    # Extended code: Add parity bit.
    
    # Steps:
    # 1. Define g(x) coefficients.
    # 2. Generate all 12-bit messages.
    # 3. Multiply msg(x) * g(x) mod (x^23 - 1) to get 23-bit codeword.
    # 4. Add parity bit to make even weight -> 24-bit codeword.
    # 5. Store all 4096 codewords.
    # 6. Shorten: Keep only words with 0 at positions 0, 1, 2, 3.
    #    This should leave 256 words (if the 4 positions are independent).
    #    The resulting words are 20 bits long (positions 4..23).
    # 7. Puncture: Remove 2 bits from the 20-bit words.
    #    Which bits? Any. Let's remove the first two of the shortened code (original pos 4, 5).
    #    Result: 18-bit words.
    # 8. Verify min distance >= 6.
    # 9. Return.
    
    # Let's double check the dimension drop.
    # Shortening a code by $s$ positions reduces dimension by at most $s$.
    # For Golay code, any 11 positions are independent? No.
    # The dual is [23, 11, 12].
    # Shortening reduces dimension by the rank of the projection.
    # For a good code, shortening 4 positions likely reduces dim by 4.
    # $12 - 4 = 8$.
    # Size $2^8 = 256$.
    
    # This seems correct and robust.

    # Construction of Golay Code
    # g(x) = 1 + x^2 + x^4 + x^5 + x^6 + x^10 + x^11
    # Coeffs (LSB first):
    g_coeffs = [1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1]
    
    # Function to multiply polynomials mod x^23 - 1
    def poly_mult_mod(a, b, n=23):
        res = [0] * n
        for i in range(n):
            if a[i]:
                for j in range(n):
                    if b[j]:
                        idx = (i + j) % n
                        res[idx] ^= 1
        return res

    # Generate all 4096 codewords
    golay_words = []
    for msg in range(4096):
        # msg is 12 bits
        msg_poly = [(msg >> i) & 1 for i in range(12)]
        # Extend to 23 bits for multiplication? No, just multiply 12-bit by 12-bit mod 23.
        # Actually, standard cyclic code generation:
        # c(x) = msg(x) * g(x)
        # Since deg(msg) < 12 and deg(g) < 12, deg(c) < 24.
        # We compute mod x^23 - 1.
        
        c_poly = poly_mult_mod(msg_poly, g_coeffs, 23)
        
        # Compute weight for parity
        w = sum(c_poly)
        parity = w % 2
        
        # Build 24-bit word
        # Bits 0..22 from c_poly, Bit 23 is parity
        word = 0
        for i in range(23):
            if c_poly[i]:
                word |= (1 << i)
        if parity:
            word |= (1 << 23)
            
        golay_words.append(word)
        
    # Shorten by 4 positions: 0, 1, 2, 3
    # Keep words where bits 0, 1, 2, 3 are 0
    shortened_words = []
    for w in golay_words:
        if (w & 1) == 0 and ((w >> 1) & 1) == 0 and ((w >> 2) & 1) == 0 and ((w >> 3) & 1) == 0:
            # Shift right by 4 to remove the shortened bits
            # The remaining bits are 4..23, which is 20 bits
            sw = w >> 4
            shortened_words.append(sw)
            
    # Check size
    if len(shortened_words) < 64:
        # Fallback to Reed-Muller if something went wrong
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words

    # Puncture by 2 positions: 0, 1 of the shortened code (original 4, 5)
    # Remove bits 0 and 1
    # Bit 0 is bit 4 of original. Bit 1 is bit 5 of original.
    # We want to keep bits 2..19 of the shortened code (original 6..23)
    # This results in 18 bits.
    
    final_words = []
    for w in shortened_words:
        # Mask out bits 0 and 1
        # Shift right by 2
        fw = w >> 2
        final_words.append(fw)
        
    # Verify min distance
    # If len is 256, verification is fast.
    min_dist = 18
    for i in range(len(final_words)):
        for j in range(i+1, len(final_words)):
            d = (final_words[i] ^ final_words[j]).bit_count()
            if d < min_dist:
                min_dist = d
                if min_dist < 6:
                    # Should not happen
                    break
        if min_dist < 6:
            break
            
    if min_dist < 6:
        # Fallback
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words
        
    return final_words

# EVOLVE-BLOCK-END
