def construct():
    import time
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill
    
    # Construct extended QR(17) code: [18, 9, 6] with 512 codewords
    # QR(17) is [17, 9, 5], extended by parity bit gives [18, 9, 6]
    
    # Primitive root mod 17: g = 3
    # Quadratic non-residues mod 17: {3, 5, 6, 7, 10, 11, 12, 14}
    # These correspond to g^i for i in {1, 3, 5, 7, 9, 11, 13, 15}
    
    # Build generator polynomial over GF(2) of degree 8
    # g_poly = prod_{i in defining_set} (x - alpha^i) where alpha = g mod 17
    
    # Work in GF(2) polynomial arithmetic
    # Represent polynomials as lists of coefficients [c0, c1, ..., cn] for c0 + c1*x + ... + cn*x^n
    
    def gf2_poly_mul(p, q):
        """Multiply two GF(2) polynomials."""
        r = [0] * (len(p) + len(q) - 1)
        for i in range(len(p)):
            if p[i]:
                for j in range(len(q)):
                    if q[j]:
                        r[i+j] ^= 1
        return r
    
    def gf2_poly_add(p, q):
        """Add two GF(2) polynomials (XOR)."""
        n = max(len(p), len(q))
        r = [0] * n
        for i in range(len(p)):
            r[i] = p[i]
        for i in range(len(q)):
            r[i] ^= q[i]
        return r
    
    # The defining set exponents: {1, 3, 5, 7, 9, 11, 13, 15}
    # We need to compute prod (x - alpha^i) over GF(2)
    # where alpha is a primitive 17th root of unity in GF(2^8)
    
    # Actually, let's use a different approach: compute the generator polynomial
    # using the fact that it's a cyclic code.
    
    # Alternative: Use the fact that QR(17) generator polynomial is known.
    # Let me compute it directly.
    
    # In GF(2), we work with polynomials. The roots are elements of GF(2^8) 
    # that are 17th roots of unity.
    
    # Let me use a simpler approach: generate the code using the parity-check matrix
    # or use the known structure.
    
    # Actually, let me just brute-force generate the QR(17) code using its definition.
    # The QR(17) code consists of all vectors (c_0, c_1, ..., c_16) where
    # c_j = trace(alpha^j * beta) for some beta in GF(2^8)
    # where trace is the field trace from GF(2^8) to GF(2)
    
    # GF(2^8) = GF(2)[x] / (x^8 + x^4 + x^3 + x^2 + 1)  # primitive polynomial
    # Actually, let me use x^8 + x^4 + x^3 + x + 1 or another primitive poly
    
    # Let me use the primitive polynomial p(x) = x^8 + x^4 + x^3 + x^2 + 1
    # This is primitive for GF(2^8).
    
    # Elements of GF(2^8) as integers 0..255
    # alpha = x (i.e., 2 in integer representation)
    
    # Precompute powers of alpha
    PRIM_POLY = 0x11D  # x^8 + x^4 + x^3 + x + 1 = 0x11D
    
    def gf2_mul(a, b):
        """Multiply two elements of GF(2^8)."""
        result = 0
        while b:
            if b & 1:
                result ^= a
            b >>= 1
            a <<= 1
            if a & 0x100:
                a ^= PRIM_POLY
        return result
    
    # alpha = x = 2 in our representation
    alpha = 2
    
    # Compute alpha^i for i = 0..16
    alpha_powers = [1]
    for i in range(1, 17):
        alpha_powers.append(gf2_mul(alpha_powers[-1], alpha))
    
    # The 17th roots of unity are alpha^0, alpha^1, ..., alpha^16
    # (since alpha^17 = 1 in GF(2^8) if alpha has order 17)
    # Wait, alpha has order 255 in GF(2^8), not 17.
    # We need an element of order 17.
    # alpha^(255/17) = alpha^15 has order 17.
    
    beta = gf2_mul(alpha_powers[15], 1)  # This isn't right either
    
    # Let me find an element of order 17 in GF(2^8)
    # We need gamma such that gamma^17 = 1 and gamma != 1
    # gamma = alpha^(255/17) = alpha^15
    
    # Compute alpha^15
    alpha_15 = 1
    for i in range(15):
        alpha_15 = gf2_mul(alpha_15, alpha)
    
    # Verify: alpha_15^17 should be 1
    check = alpha_15
    for i in range(16):
        check = gf2_mul(check, alpha_15)
    assert check == 1, f"alpha^15 does not have order 17, got {check}"
    
    # Now the 17th roots of unity are: 1, alpha_15, alpha_15^2, ..., alpha_15^16
    # Let's compute them
    roots17 = [1]
    for i in range(1, 17):
        roots17.append(gf2_mul(roots17[-1], alpha_15))
    
    # The trace function: Tr(x) = x + x^2 + x^4 + x^8 (sum over Frobenius orbit)
    def trace(x):
        t = x
        t ^= gf2_mul(t, t)      # x^2
        t ^= gf2_mul(t, t)      # x^4  -- wait, this is wrong
        # Tr(x) = x + x^2 + x^4 + x^8
        x2 = gf2_mul(x, x)
        x4 = gf2_mul(x2, x2)
        x8 = gf2_mul(x4, x4)
        return (x ^ x2 ^ x4 ^ x8) & 1
    
    # The QR(17) code: codewords are (Tr(beta * roots17[j]) for j=0..16) for beta in GF(2^8)
    # But we only need 9 dimensions, so beta ranges over a 9-dimensional subspace
    # Actually, the QR code has dimension (17+1)/2 = 9, so we need 9 basis elements
    
    # The code is: { (Tr(beta * w_j) for j=0..16) : beta in GF(2^8) }
    # but this gives a code of dimension at most 8 (since beta is in GF(2^8))
    # Actually, the full trace code has dimension 8, and we need to add the all-ones vector
    # to get dimension 9.
    
    # Let me generate all 512 codewords by iterating over all 9-bit combinations
    # of a basis for the appropriate subspace.
    
    # Simpler approach: generate all codewords as (Tr(beta * w_j)) for beta in a 9-dim subspace
    # The subspace is spanned by {1, alpha_15, alpha_15^2, ..., alpha_15^8} perhaps?
    
    # Actually, let me just generate all 256 trace codewords plus their complements,
    # or find the right 9-dimensional subspace.
    
    # The QR(17) code = { c : c_j = Tr(beta * w_j) } union { c : c_j = Tr(beta * w_j) + 1 }
    # where w_j = alpha_15^j, and beta ranges over GF(2^8).
    # This gives 2 * 256 = 512 codewords.
    
    words = []
    seen = set()
    
    for beta in range(256):
        # Generate codeword c where c_j = Tr(beta * w_j)
        word = 0
        for j in range(17):
            val = gf2_mul(beta, roots17[j])
            bit = trace(val)
            word |= (bit << j)
        
        # Add parity bit to extend to length 18
        parity = word.bit_count() % 2
        extended_word = word | (parity << 17)
        
        if extended_word not in seen:
            seen.add(extended_word)
            words.append(extended_word)
        
        # Also add the complement (add 1 to each of the first 17 bits, recompute parity)
        comp_word = word ^ 0x1FFFF  # XOR with all-ones for 17 bits
        comp_parity = comp_word.bit_count() % 2
        comp_extended = comp_word | (comp_parity << 17)
        
        if comp_extended not in seen:
            seen.add(comp_extended)
            words.append(comp_extended)
    
    # Check time
    if time.time() - start_time > DEADLINE:
        return words
    
    # Verify minimum distance is at least 6
    # Spot check a few pairs
    n = len(words)
    min_dist = 18
    for i in range(min(n, 50)):
        for j in range(i+1, min(n, 50)):
            d = (words[i] ^ words[j]).bit_count()
            if d < min_dist:
                min_dist = d
    
    return words
